"""Combine K independently trained member checkpoints into one ensemble checkpoint.

The fixed scorer takes a single `--checkpoint` file, so the whole ensemble has to
travel inside one state dict. `nn.ModuleList` gives the members the `members.<k>.`
key prefix, and torch.save deduplicates the tied embedding that each member stores
under both `token.weight` and `head.weight`, so the file is close to
`K * (unique parameters) * 4 bytes` rather than twice that.

README section 3 requires logging "checkpoint ancestry and search costs" and warns
that "reusing a checkpoint does not erase its training cost", so this tool writes a
metrics.json naming every member run, its checkpoint digest, its seed and its
processed-target count. `--search-cost` carries the totals forward when the member
runs were themselves fine-tuned from earlier checkpoints.

Usage
-----
    python pack_ensemble.py --out runs/ens_k4 \
        --member runs/a/checkpoint.pt --member runs/b/checkpoint.pt \
        --member runs/c/checkpoint.pt --member runs/d/checkpoint.pt
"""
import argparse
import json
from pathlib import Path

import torch

from common import PROTOCOL, sha


def load_member(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['protocol'] != PROTOCOL:
        raise ValueError('%s belongs to a different course protocol' % path)
    return payload


def prefix_state_dict(state_dict, rank):
    """Rewrite `token.weight` to `members.<rank>.token.weight`.

    Tensor identity is preserved, so the tied embedding a member stores under both
    `token.weight` and `head.weight` stays shared and is serialized once.
    """
    return {'members.%d.%s' % (rank, key): value for key, value in state_dict.items()}


def member_state_dict(payload):
    """The member's own weights, with a size-1 ensemble wrapper removed.

    `train_v2.py` saves `make_model(args.implementation, config).state_dict()`
    verbatim, and the key prefix follows from which factory that was:

      implementation=model    -> a bare `GPT` (53 keys, no prefix)      -- measured
      implementation=student  -> a `VariantEnsemble` of size 1, whose keys
                                 carry a `members.0.` prefix

    This tool adds `members.<rank>.` on top, so the second form would become
    `members.<rank>.members.0....` and the evaluator could not load it.  Measured
    on exactly that file: `RuntimeError ... 106 unexpected key(s)`, raised by
    `load_state_dict` -- i.e. at scoring time, after submission, unless
    `_verify_rebuild` catches it here first (it does: this is what it was for).

    One level is removed, but only when the payload really IS a size-1 ensemble.
    A K>1 member is a different object; stripping it would corrupt the file.
    """
    state = payload['model']
    if int(payload['config'].get('ensemble_size', 1)) != 1:
        return state
    if not all(key.startswith('members.0.') for key in state):
        return state
    return {key[len('members.0.'):]: value for key, value in state.items()}


def pack(member_paths, out_dir, search_cost=None, overwrite=False):
    out_dir = Path(out_dir)
    checkpoint_path = out_dir / 'checkpoint.pt'
    if checkpoint_path.exists() and not overwrite:
        raise SystemExit('%s exists; pass --overwrite to replace it' % checkpoint_path)

    members = [load_member(path) for path in member_paths]
    shared = dict(members[0]['config'])
    shared.pop('ensemble_size', None)
    shared.pop('member_configs', None)
    member_configs = []
    model = {}
    for rank, payload in enumerate(members):
        config = dict(payload['config'])
        config.pop('ensemble_size', None)
        config.pop('member_configs', None)
        for field in ('vocab', 'context'):
            if config[field] != shared[field]:
                raise ValueError('Member %d disagrees on %s; the protocol fixes it.'
                                 % (rank, field))
        member_configs.append(config)
        model.update(prefix_state_dict(member_state_dict(payload), rank))

    config = dict(shared, ensemble_size=len(members), member_configs=member_configs)
    implementation = _implementation_for(members, member_configs)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol': PROTOCOL, 'implementation': implementation, 'config': config,
                'model': model, 'ensemble_size': len(members)}, checkpoint_path)

    # Unique parameters: `parameters()` deduplicates the tied embedding, and the
    # members are distinct objects, so this is K * (per-member unique count).
    per_member = sum(p.numel() for p in
                     _probe(config, member_configs, implementation).parameters())
    _verify_rebuild(checkpoint_path, implementation, per_member)
    metrics = {
        'protocol': PROTOCOL,
        'implementation': 'ensemble',
        'ensemble_size': len(members),
        'parameters': per_member,
        'parameters_per_member': per_member // len(members),
        'member_configs': member_configs,
        'members': [
            {
                'rank': rank,
                'checkpoint': str(Path(path)),
                'checkpoint_sha256': sha(path),
                'seed': payload.get('seed'),
                'train_tokens': payload.get('train_tokens'),
                'parameters': sum(p.numel() for p in
                                  _probe(dict(payload['config']), [payload['config']],
                                         payload['implementation']).parameters()),
            }
            for rank, (path, payload) in enumerate(zip(member_paths, members))
        ],
        'ancestry': [str(Path(path)) for path in member_paths],
        'search_cost': search_cost,
        'checkpoint_bytes': checkpoint_path.stat().st_size,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return checkpoint_path, metrics


def _probe(config, member_configs, implementation='ensemble'):
    """Build the module the EVALUATOR would build, just to count parameters.

    `implementation` is the checkpoint's own dispatch key: `evaluate.py:54` calls
    `make_model(checkpoint['implementation'], checkpoint['config'], device)`.  A
    probe that hard-codes `ensemble.Ensemble` therefore measures a model this
    submission does not contain.  Measured: a member with `mixer_mode=1` has
    824,064 parameters, and the hard-coded probe reports 1,088,256.
    """
    from common import make_model
    model, _ = make_model(implementation,
                          dict(config, member_configs=member_configs),
                          torch.device('cpu'))
    return model


def _implementation_for(members, member_configs):
    """The module the evaluator must import to rebuild THIS ensemble.

    `ensemble.py` is the right answer only while no member switches a variant
    module on.  `Ensemble` builds plain `GPT` members and ignores every switch in
    `student.FLAG_SPEC`, so a member carrying one would be reconstructed as a
    different model -- a strict `load_state_dict` away from either an exception
    or a silently wrong score.

    `student.py`'s `VariantEnsemble` is a plain `Ensemble` when no switch is set
    (the dispatch is bit-identical, and the launch gate proves the golden path),
    so for golden ensembles both answers are correct and the recorded value stays
    `ensemble` -- the value whose end-to-end deployment gate has already passed.
    Only the case `ensemble` provably cannot express changes it.
    """
    from student import variant_active
    if not any(variant_active(member) for member in member_configs):
        return 'ensemble'
    impls = sorted({member['implementation'] for member in members})
    if len(impls) != 1:
        raise ValueError('members came from different implementations %s; the '
                         'evaluator can import only one' % impls)
    if impls[0] == 'ensemble':
        raise ValueError('a member config switches on a variant module but was '
                         'trained with implementation=ensemble, which cannot '
                         'rebuild it')
    return impls[0]


def _verify_rebuild(checkpoint_path, implementation, reported):
    """Rebuild the file we just wrote, exactly as `evaluate.py:54` does.

    The parameter count above is evidence about the SUBMISSION only if it
    describes the model inside the checkpoint.  This makes that a checked fact
    instead of an assumption, and it fails loudly at pack time rather than
    producing a number that describes a model nobody trained.
    """
    from common import make_model
    written = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if written['implementation'] != implementation:
        raise ValueError('the checkpoint records implementation=%s but %s was '
                         'probed' % (written['implementation'], implementation))
    rebuilt, _ = make_model(written['implementation'],
                           written['config'], torch.device('cpu'))
    rebuilt.load_state_dict(written['model'])
    n = sum(p.numel() for p in rebuilt.parameters())
    if n != reported:
        raise ValueError('the packed checkpoint rebuilds to %d parameters but the '
                         'metrics report %d: implementation=%s does not describe '
                         'these members' % (n, reported, implementation))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--member', action='append', required=True, type=Path,
                        help='Member checkpoint; repeat once per member, in rank order.')
    parser.add_argument('--out', required=True, type=Path, help='Output run directory.')
    parser.add_argument('--search-cost', type=float, default=None,
                        help='Total processed training targets across all members.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    checkpoint_path, metrics = pack(args.member, args.out,
                                    search_cost=args.search_cost, overwrite=args.overwrite)
    size = metrics['checkpoint_bytes']
    print(json.dumps({'checkpoint': str(checkpoint_path),
                      'bytes': size,
                      'mib': round(size / 2**20, 3),
                      'ensemble_size': metrics['ensemble_size'],
                      'parameters': metrics['parameters'],
                      'checkpoint_sha256': sha(checkpoint_path)}, indent=2))


if __name__ == '__main__':
    main()

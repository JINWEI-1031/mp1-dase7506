"""Configurable fork of the classroom trainer (protocol 7506-mp1-wt2-v2).

Fork of ``code/train.py``. The checkpoint stays **byte-compatible** with the
course evaluator, i.e. it still saves exactly:

    torch.save({'protocol': PROTOCOL, 'implementation': ..., 'config': ...,
                'model': state_dict, 'seed': ..., 'train_tokens': ...}, path)

so ``evaluate.py --checkpoint <this>`` loads it without any change.

With none of the new flags set the run is a bit-exact reproduction of
``code/train.py`` (same LR sequence, same loss, same checkpoint bytes); this is
asserted by the self-test recorded in the delivery notes.

New knobs (all default to the classroom behaviour):

  --lr, --weight-decay, --beta2        optimizer hyper-parameters
  --decay-scope {all,matrices,         which parameters get --weight-decay; both
                 matrices_no_embed}    groups always share one lr (see build_optimizer)
  --schedule {cosine,d2z}              cosine = classroom; d2z = warmup, hold
                                       peak, linear decay to exactly 0
  --warmup-steps, --min-lr-frac        warmup length / cosine floor
  --d2z-decay-frac                     fraction of steps used by the d2z cooldown
  --dropout                            residual + embedding dropout (forward level)
  --arch                               depth/width/heads override of --config
  --eval-split {validation,test}       split used by --eval-every and the final score
  --prefix-weight                      weight of the prefix-token auxiliary loss
  --prefix-warmdown-frac               last fraction of steps where it decays to 0
  --prefix-schedule {warmdown,pr}      warmdown = local rule; pr = reuse
                                       prefix.prefix_weight() (PR #337 schedule)
  --ema-decay                          >0 enables within-trajectory EMA averaging
  --log-every, --run-dir, --seed, --steps, --batch-size, --device, --precision,
  --threads, --eval-every              (as in train.py)
  --overwrite-run-dir                  allow a non-empty run directory (off by
                                       default, exactly like train.py)

Prefix auxiliary loss. The helper lives in a separate module (``prefix.py``,
implemented by another workstream); this trainer only calls its documented
interface::

    from prefix import build_prefix_table, prefix_ce_loss
    table = build_prefix_table()                          # LongTensor, len 2048, -1 = ignore
    loss_prefix = prefix_ce_loss(logits, targets, table)  # logits [B,T,2048] f32, targets [B,T] long

If ``prefix.py`` is missing, the import falls back to a stub that raises only
when ``--prefix-weight > 0`` is actually used, so the default path still runs.
"""
import argparse
import json
import math
import os
import re
import sys
import time
import types
from pathlib import Path

import torch
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Path bootstrap: this file lives in _plans/, the course package in
# ../MP1_student_starter/code (locally) or anywhere on the instance. Resolve the
# directory that holds common.py / model.py / student.py before importing them.
# --------------------------------------------------------------------------- #
def find_code_dir():
    """Locate the classroom ``code/`` directory (contains common.py + model.py)."""
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get('MP1_CODE_DIR')
    if env:
        candidates.append(Path(env))
    candidates += [
        here / 'code',
        here / '..' / 'MP1_student_starter' / 'code',
        here / '..' / 'code',
        Path.cwd(),
        Path.cwd() / 'code',
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / 'common.py').is_file() and (candidate / 'model.py').is_file():
            return candidate
    raise SystemExit(
        'train_v2.py: cannot find the course code directory (needs common.py and model.py).\n'
        'Searched: ' + ', '.join(str(c) for c in candidates) + '\n'
        'Set MP1_CODE_DIR=/path/to/code to override.')


CODE_DIR = find_code_dir()
for _p in (str(CODE_DIR), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha  # noqa: E402
from evaluate import score  # noqa: E402

# --------------------------------------------------------------------------- #
# Optional prefix-token auxiliary loss (implemented in a separate module).
# --------------------------------------------------------------------------- #
PREFIX_IMPORT_ERROR = None
PREFIX_WEIGHT_HELPER = None
try:
    from prefix import build_prefix_table, prefix_ce_loss  # type: ignore
    HAVE_PREFIX = True
    try:  # optional: prefix.py also ships the PR's staged schedule
        from prefix import prefix_weight as PREFIX_WEIGHT_HELPER  # type: ignore
    except ImportError:
        PREFIX_WEIGHT_HELPER = None
except Exception as exc:  # noqa: BLE001 - keep the default path runnable
    HAVE_PREFIX = False
    PREFIX_IMPORT_ERROR = exc

    def build_prefix_table(*args, **kwargs):
        raise ImportError('prefix.py is unavailable: %r' % (PREFIX_IMPORT_ERROR,))

    def prefix_ce_loss(*args, **kwargs):
        raise ImportError('prefix.py is unavailable: %r' % (PREFIX_IMPORT_ERROR,))

ARCH_KEYS = ('depth', 'width', 'heads')
DECAY_SCOPES = ('all', 'matrices', 'matrices_no_embed')


def _switch_spec():
    """`student.FLAG_SPEC` (name -> (default, bare-name value)), imported lazily.

    Lazy so that `train_v2.py` still imports and runs when `student.py` is
    swapped back for a plain baseline. Note that a spec file needs none of this:
    module switches go straight into the variant's `config` block, which the
    runner already copies verbatim into `config.json`.
    """
    try:
        import student
        return dict(student.FLAG_SPEC)
    except Exception:
        return {}


def _coerce_switch(spec, key, value):
    return float(value) if isinstance(spec[key][0], float) else int(value)


def parse_arch(text):
    """Parse an --arch override: ``4x128x4``, ``depth=4,width=256,heads=4`` or JSON.

    Module switches (`student.py`, see `_plans/arch_modules_notes.md`) may be
    appended to the shorthand: ``4x128x4,softcap=15,v_res=1`` or, for the switches
    whose "on" value is fixed, named bare: ``4x128x4+swiglu+bigram_dim=64``.
    """
    switches = _switch_spec()
    text = text.strip()
    if text.startswith('{'):
        parsed = json.loads(text)
    else:
        parsed = {}
        for part in re.split(r'[,;+ ]+', text):
            part = part.strip()
            if not part:
                continue
            key, sep, value = part.partition('=')
            key = key.strip()
            if sep:
                if key in ARCH_KEYS:
                    parsed[key] = int(value)
                elif key in switches:
                    parsed[key] = _coerce_switch(switches, key, value)
                else:
                    raise ValueError('--arch: unknown key %r; allowed: %s + %s'
                                     % (key, list(ARCH_KEYS), sorted(switches)))
            elif re.fullmatch(r'\d+[xX]\d+[xX]\d+', part):
                parsed.update(zip(ARCH_KEYS, (int(n) for n in re.split(r'[xX]', part))))
            elif key in switches and switches[key][1] is not None:
                parsed[key] = _coerce_switch(switches, key, switches[key][1])
            elif key in switches:
                raise ValueError('--arch: %r needs an explicit value, e.g. %s=32' % (key, key))
            else:
                raise ValueError('--arch: expected key=value, depth x width x heads or a '
                                 'switch name, got %r' % (part,))
    unknown = sorted(set(parsed) - set(ARCH_KEYS) - set(switches))
    if unknown:
        raise ValueError('--arch: unknown key(s) %s; allowed: %s + %s'
                         % (unknown, list(ARCH_KEYS), sorted(switches)))
    for key in ARCH_KEYS:
        if key not in parsed:
            raise ValueError('--arch: missing %r' % (key,))
        if int(parsed[key]) < 1:
            raise ValueError('--arch: %s must be >= 1' % (key,))
    if int(parsed['width']) % int(parsed['heads']):
        raise ValueError('--arch: width must be divisible by heads')
    return {k: (int(v) if k in ARCH_KEYS else v) for k, v in parsed.items()}


def variant_keys_of(config):
    """The subset of `config` that names `student.py` module switches."""
    switches = _switch_spec()
    return {k: v for k, v in config.items() if k in switches}


def learning_rate_at(step, args, total_steps):
    """LR for 0-based ``step``.

    Defaults reproduce train.py:52 exactly (bit-for-bit, verified off-line):
        .001 * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/steps)))
    """
    warmup = max(1, args.warmup_steps)
    ramp = min(1., (step + 1) / warmup)
    if args.schedule == 'cosine':
        floor = args.min_lr_frac
        shape = floor + (1.0 - floor) * .5 * (1 + math.cos(math.pi * step / total_steps))
        return args.lr * ramp * shape
    if args.schedule == 'd2z':
        decay_steps = max(1, int(round(total_steps * args.d2z_decay_frac)))
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return args.lr * ramp
        progress = min(1.0, (step - decay_start) / decay_steps)
        return args.lr * (1.0 - progress)
    raise ValueError('unknown schedule %r' % (args.schedule,))


def prefix_weight_at(step, args, total_steps):
    """Prefix-loss weight: constant, then linearly to 0 over the last fraction.

    ``--prefix-schedule pr`` delegates to ``prefix.prefix_weight`` (the stage
    schedule ported from modded-nanogpt PR #337) instead of the local warmdown.
    """
    if args.prefix_weight <= 0.:
        return 0.
    if args.prefix_schedule == 'pr':
        if PREFIX_WEIGHT_HELPER is None:
            raise RuntimeError('--prefix-schedule pr needs prefix.prefix_weight()')
        return float(PREFIX_WEIGHT_HELPER(step, total_steps, peak=args.prefix_weight))
    hold = int(round(total_steps * (1. - args.prefix_warmdown_frac)))
    if step < hold:
        return args.prefix_weight
    remaining = max(1, total_steps - hold)
    return args.prefix_weight * max(0., (total_steps - step) / remaining)


def add_dropout(model, p):
    """Attach dropout to the residual and embedding paths of a model.py GPT.

    model.py must stay untouched, so the two forward methods are rebound on the
    *instance* with identical arithmetic plus F.dropout. Only called when p > 0,
    so the default path keeps the original (bit-exact) code path.

    Rebinding replaces `Block.forward` wholesale, so on a `student.GPTVariant`
    block it would silently discard every architecture module the run switched
    on. Refuse instead of producing a run whose config no longer describes it.
    """
    variant_blocks = [b for b in model.modules() if getattr(b, '_variant', False)]
    if variant_blocks:
        raise SystemExit('--dropout is incompatible with the student.py module '
                         'switches (%d variant block(s) found): add_dropout rebinds '
                         'Block.forward and would silently drop them. Use droppath '
                         'instead, or run the arm without --dropout.'
                         % len(variant_blocks))

    def block_forward(self, x, p=p):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)), p, self.training)
        return x + F.dropout(self.mlp(self.norm2(x)), p, self.training)

    def features(self, ids, p=p):
        x = self.token(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = F.dropout(x, p, self.training)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    for block in model.blocks:
        block.forward = types.MethodType(block_forward, block)
    model.features = types.MethodType(features, model)
    return model


# --------------------------------------------------------------------------- #
# Weight-decay grouping (--decay-scope)
# --------------------------------------------------------------------------- #
def unique_parameters(model):
    """``(params, duplicates)``: every parameter tensor once, deduplicated by identity.

    ``model.parameters()`` already dedupes, but that is exactly the trap here: the
    course ``model.py`` ties the output head to the embedding table
    (``self.head.weight = self.token.weight`` -- the *same* ``Parameter`` object
    registered under two module paths, not a copy).  Any traversal that goes through
    ``named_modules()``/``named_parameters(remove_duplicate=False)``/``state_dict()``
    yields that tensor twice, so a hand-rolled grouping would decay it twice and
    report a parameter count 262,144 too high (== ``vocab*width``, for the default
    config).  Deduplicate on ``id()`` explicitly rather than relying on the default.
    """
    seen = set()
    params = []
    duplicates = []
    for name, param in model.named_parameters(remove_duplicate=False):
        if id(param) in seen:
            duplicates.append(name)
            continue
        seen.add(id(param))
        params.append(param)
    return params, duplicates


def embedding_parameter_ids(model):
    """``id()`` of every parameter owned by an ``nn.Embedding`` module.

    Identity, not name, so the tied ``head.weight`` (an ``nn.Linear`` weight that *is*
    ``token.weight``) is classified as an embedding and not double-classified.
    """
    ids = set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            for param in module.parameters(recurse=False):
                ids.add(id(param))
    return ids


def build_optimizer(model, args):
    """AdamW over ``model``, grouped according to ``args.decay_scope``.

    ``all`` (the default) is literally the pre-existing call -- one group over
    ``model.parameters()`` with ``weight_decay=args.weight_decay`` -- so every run
    recorded before this flag existed stays bit-for-bit comparable.

    ``matrices`` keeps decay on ``p.dim() >= 2`` only and puts everything else
    (LayerNorm gains/biases, every bias, the embedding tables) in a second group with
    ``weight_decay=0.0``.  ``matrices_no_embed`` additionally pulls the embedding
    tables out of the decay group.

    Both groups are constructed with the same ``lr``: the training loop writes
    ``group['lr']`` for *every* group each step, so a group that missed that write
    would silently train at a stale lr.  Keeping ``lr`` in both groups at build time
    means the loop's uniform write is the only thing that ever sets it.
    """
    params, duplicates = unique_parameters(model)
    embed_ids = embedding_parameter_ids(model)
    betas = (0.9, args.beta2)
    decay = []
    no_decay = []
    if args.decay_scope == 'all':
        decay = list(params)
        optimizer = torch.optim.AdamW(decay, lr=args.lr, weight_decay=args.weight_decay,
                                      betas=betas)
    else:
        exclude = embed_ids if args.decay_scope == 'matrices_no_embed' else set()
        for param in params:
            (decay if (param.dim() >= 2 and id(param) not in exclude) else no_decay).append(param)
        optimizer = torch.optim.AdamW([
            {'params': decay, 'lr': args.lr, 'betas': betas, 'weight_decay': args.weight_decay},
            {'params': no_decay, 'lr': args.lr, 'betas': betas, 'weight_decay': 0.},
        ])
    embed_params = [p for p in params if id(p) in embed_ids]
    stats = {
        'decay_scope': args.decay_scope,
        'decay_group_tensors': len(decay),
        'decay_group_numel': int(sum(p.numel() for p in decay)),
        'no_decay_group_tensors': len(no_decay),
        'no_decay_group_numel': int(sum(p.numel() for p in no_decay)),
        'decay_group_weight_decay': float(optimizer.param_groups[0]['weight_decay']),
        'no_decay_group_weight_decay': (float(optimizer.param_groups[1]['weight_decay'])
                                        if len(optimizer.param_groups) > 1 else None),
        'unique_param_tensors': len(params),
        'unique_param_numel': int(sum(p.numel() for p in params)),
        'embedding_param_tensors': len(embed_params),
        'embedding_param_numel': int(sum(p.numel() for p in embed_params)),
        'tied_param_tensors_dropped': len(duplicates),
        'tied_param_names_dropped': duplicates,
    }
    assert (stats['decay_group_numel'] + stats['no_decay_group_numel']
            == stats['unique_param_numel']), 'decay groups must partition the parameters'
    return optimizer, stats


def self_peak_rss_gb():
    """Peak RSS of this process in GiB (POSIX only; ru_maxrss is KiB on Linux)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    return peak / (1024. ** 2) if sys.platform == 'darwin' else peak / (1024. * 1024.)


class ModelEMA:
    """Within-trajectory EMA of the parameters (timm-style decay warmup).

    The shadow starts as a copy of the initial weights and the effective decay
    ramps as min(decay, (1+n)/(10+n)) so that the (long) early phase does not
    pin the average to the random initialisation.
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if value.dtype.is_floating_point:
                shadow.mul_(decay).add_(value.detach().float(), alpha=1. - decay)
            else:
                shadow.copy_(value.detach().float())

    @torch.no_grad()
    def copy_to(self, model):
        reference = model.state_dict()
        model.load_state_dict({k: v.to(reference[k].dtype) for k, v in self.shadow.items()})


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    p.add_argument('--save-every', type=int, default=0,
                   help='Optional intermediate-checkpoint interval; 0 saves only the final '
                        'checkpoint. Files land next to checkpoint.pt as '
                        'checkpoint_step<N>.pt, with the SAME dict layout, so '
                        '`evaluate.py --checkpoint <one of them>` loads it unchanged. '
                        'Pair it with --eval-every <= N: selection must be driven by the '
                        'validation curve, and every snapshot then has a paired reading.')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--overwrite-run-dir', action='store_true',
                   help='Allow training into a run directory that already holds results.')
    # optimizer / schedule
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--decay-scope', choices=list(DECAY_SCOPES), default='all',
                   help='Which parameters receive --weight-decay. all = one group over '
                        'model.parameters() (the pre-existing behaviour, default); '
                        'matrices = only p.dim()>=2, everything else (LayerNorm gains/biases, '
                        'all biases, embedding tables) gets weight_decay=0; '
                        'matrices_no_embed = matrices minus the embedding tables. '
                        'Both groups always share one lr. Group sizes are recorded in '
                        'metrics.json under decay_group_*/no_decay_group_*.')
    p.add_argument('--schedule', choices=['cosine', 'd2z'], default='cosine')
    p.add_argument('--warmup-steps', type=int, default=100)
    p.add_argument('--min-lr-frac', type=float, default=0.1)
    p.add_argument('--d2z-decay-frac', type=float, default=0.2,
                   help='d2z: fraction of the run spent in the linear decay to zero.')
    # regularisation / architecture
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--arch', default=None,
                   help='depth/width/heads override of --config, e.g. 4x256x4 or depth=4,width=256,heads=4')
    # auxiliary loss
    p.add_argument('--prefix-weight', type=float, default=0.0)
    p.add_argument('--prefix-warmdown-frac', type=float, default=0.2)
    p.add_argument('--prefix-schedule', choices=['warmdown', 'pr'], default='warmdown',
                   help='warmdown = hold peak, then linear to 0 over the last '
                        '--prefix-warmdown-frac; pr = reuse prefix.prefix_weight()')
    # post-processing
    p.add_argument('--ema-decay', type=float, default=0.0)
    p.add_argument('--eval-split', choices=['validation', 'test'], default='validation')
    # provenance — README.md:88 requires "checkpoint ancestry and search costs" to be
    # logged. These two are recorded in metrics.json ONLY; they are deliberately kept
    # out of the checkpoint dict below, whose key list is fixed, so that recording
    # provenance can never change a checkpoint's bytes (cf. ledger check 24: a metadata
    # field DID change the hash once, via `implementation`).
    p.add_argument('--parent-checkpoints', default='',
                   help='Comma-separated run ids this run descends from; empty = trained from scratch.')
    p.add_argument('--search-cost', default='',
                   help='Cost of the search that produced this configuration, as a free-form string.')
    return p


def main():
    total_started = time.perf_counter()
    p = build_parser()
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.log_every < 1:
        p.error('--log-every must be positive.')
    if not 0. <= args.dropout < 1.:
        p.error('--dropout must be in [0, 1).')
    if not 0. <= args.prefix_warmdown_frac < 1.:
        p.error('--prefix-warmdown-frac must be in [0, 1).')
    if not 0. <= args.d2z_decay_frac <= 1.:
        p.error('--d2z-decay-frac must be in [0, 1].')
    if args.prefix_weight > 0. and not HAVE_PREFIX:
        p.error('--prefix-weight %g requires prefix.py on the import path (import failed: %r)'
                % (args.prefix_weight, PREFIX_IMPORT_ERROR))
    if args.prefix_weight > 0. and args.prefix_schedule == 'pr' and PREFIX_WEIGHT_HELPER is None:
        p.error('--prefix-schedule pr requires prefix.prefix_weight(), which this prefix.py '
                'does not export')
    arch = None
    try:
        arch = parse_arch(args.arch) if args.arch else None
    except (ValueError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.overwrite_run_dir:
        p.error('Run directory already contains results. Use a new --run-dir.')

    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    if arch:
        config.update(arch)
    model, implementation_sha = make_model(args.implementation, config, device)
    if args.dropout > 0.:
        add_dropout(model, args.dropout)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer, decay_stats = build_optimizer(model, args)
    prefix_table = build_prefix_table() if args.prefix_weight > 0. else None
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0. else None
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter() - prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    snapshots = []
    for step in range(args.steps):
        starts = torch.randint(len(tokens) - 257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:, None] + torch.arange(257, device=device)]
        learning_rate = learning_rate_at(step, args, args.steps)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            logits = model(batch[:, :-1])
            loss_ce = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())
        # `loss` is the TOTAL objective (CE + prefix aux).  Logging only the total makes
        # runs with different --prefix-weight incomparable: the aux term is >= 0, so a
        # larger weight mechanically shows a larger loss at equal quality.  Keep the CE
        # component separate; it is the only quantity comparable across recipes.
        loss = loss_ce
        prefix_weight = prefix_weight_at(step, args, args.steps)
        if prefix_weight > 0.:
            loss = loss + prefix_weight * prefix_ce_loss(logits.float(), batch[:, 1:], prefix_table)
        log_ce = loss_ce.detach()  # detach is free; .item() (a GPU sync) happens only when logging
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'loss': loss.item(), 'loss_ce': log_ce.item(),
                   'prefix_weight': prefix_weight, 'lr': learning_rate,
                   'seconds': time.perf_counter() - started - intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row), flush=True)
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            intermediate = score(model, *data[args.eval_split], device, 'fp32')
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step': step + 1, **intermediate})
            print(json.dumps({'validation': validation_history[-1]}), flush=True)
        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            # A snapshot must NOT disturb the run: copy the tensors to CPU instead of
            # moving the module (the final save uses model.cpu(), which is only safe
            # because training is over by then).  train_tokens records what THIS
            # snapshot has actually seen, not the run's total -- otherwise a mid-run
            # checkpoint would claim tokens it never processed.
            snap = args.run_dir / ('checkpoint_step%d.pt' % (step + 1))
            torch.save({'protocol': PROTOCOL, 'implementation': args.implementation,
                        'config': config, 'seed': args.seed,
                        'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'train_tokens': (step + 1) * args.batch_size * 256}, snap)
            snapshots.append({'step': step + 1, 'path': snap.name, 'sha256': sha(snap)})
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started - intermediate_validation_seconds
    validation_raw = score(model, *data[args.eval_split], device, 'fp32')
    validation_raw.pop('window_nll_nats')
    validation = validation_raw
    validation_ema = None
    ema_apply_seconds = None
    if ema is not None:
        # The checkpoint carries the averaged weights, so the reported score must
        # be measured on those same weights.
        ema_apply_started = time.perf_counter()
        ema.copy_to(model)
        ema_apply_seconds = time.perf_counter() - ema_apply_started
        validation_ema = score(model, *data[args.eval_split], device, 'fp32')
        validation_ema.pop('window_nll_nats')
        validation = validation_ema
    checkpoint = args.run_dir / 'checkpoint.pt'
    torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                'model': model.cpu().state_dict(), 'seed': args.seed,
                'train_tokens': args.steps * args.batch_size * 256}, checkpoint)
    if ema is not None:
        torch.save({'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config,
                    'model': model.state_dict(), 'seed': args.seed,
                    'train_tokens': args.steps * args.batch_size * 256,
                    'ema_decay': args.ema_decay}, args.run_dir / 'checkpoint_ema.pt')
    result = {'protocol': PROTOCOL, 'implementation': args.implementation, 'config': config, 'seed': args.seed,
              'parameters': sum(p.numel() for p in model.parameters()), 'precision': precision,
              'train_tokens': args.steps * args.batch_size * 256, 'preparation_seconds': preparation_seconds,
              'train_seconds': train_seconds, 'validation': validation,
              'validation_raw': validation_raw, 'validation_ema': validation_ema,
              'eval_split': args.eval_split, 'history': history,
              'validation_history': validation_history,
              'intermediate_validation_seconds': intermediate_validation_seconds,
              'eval_every': args.eval_every, 'save_every': args.save_every,
              'snapshots': snapshots,
              'process_seconds': time.perf_counter() - total_started,
              'torch_version': str(torch.__version__), 'threads': args.threads,
              'checkpoint_sha256': sha(checkpoint), 'implementation_sha256': implementation_sha,
              'lr': args.lr, 'weight_decay': args.weight_decay, 'beta2': args.beta2,
              # decay_scope + the measured group sizes (tensors / elements / wd per group);
              # provenance only, deliberately NOT in the checkpoint dict (its key list is fixed).
              **decay_stats,
              'schedule': args.schedule, 'warmup_steps': args.warmup_steps,
              'min_lr_frac': args.min_lr_frac, 'd2z_decay_frac': args.d2z_decay_frac,
              'dropout': args.dropout, 'arch': arch, 'log_every': args.log_every,
              'prefix_weight': args.prefix_weight,
              'prefix_warmdown_frac': args.prefix_warmdown_frac,
              'prefix_schedule': args.prefix_schedule,
              'prefix_module': getattr(sys.modules.get('prefix'), '__file__', None) if HAVE_PREFIX else None,
              'ema_decay': args.ema_decay, 'ema_apply_seconds': ema_apply_seconds,
              'config_path': str(args.config.resolve()),
              'parent_checkpoints': args.parent_checkpoints,
              'search_cost': args.search_cost,
              'peak_rss_gb': self_peak_rss_gb(),
              **device_metrics(device)}
    (args.run_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result | {'history': []}, indent=2), flush=True)


if __name__ == '__main__':
    main()

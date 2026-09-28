"""Generate text from a frozen checkpoint, single model or ensemble.

The leaderboard scores bits per byte, and BPB is a proper scoring rule over the
whole predictive distribution, so it already measures prediction quality more
faithfully than any handful of sampled continuations. Sampling is therefore not a
metric here. It is a sanity check with two jobs: it shows what the model actually
produces rather than only reporting a scalar, and it corroborates that an
ensemble's lower BPB reflects a genuinely better distribution instead of a
scoring artifact -- probability averaging lowers BPB by construction, so an
independent view of the output is worth having.

Prompts are drawn from the training text. Nothing here touches validation or test
targets, and sampling never influences any selection.

    python sample.py --checkpoint runs_ens/k4/checkpoint.pt --seed 17 --samples 2
"""
import argparse
from pathlib import Path

import torch

from common import ROOT, load_data, make_model, setup
from tokenizers import Tokenizer

CONTEXT = 256


@torch.no_grad()
def generate(model, prompt_ids, new_tokens, temperature, top_k, seed, device):
    """Sample `new_tokens` tokens, re-reading at most the last CONTEXT tokens each step.

    Re-reading the window keeps every prediction strictly causal and stateless,
    which is the same contract the scorer relies on.
    """
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = list(prompt_ids)
    for _ in range(new_tokens):
        window = torch.tensor([ids[-CONTEXT:]], dtype=torch.long, device=device)
        logits = model(window).float()[0, -1]
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            logits = logits / temperature
            if top_k:
                cutoff = torch.topk(logits, top_k).values[-1]
                logits = logits.masked_fill(logits < cutoff, float('-inf'))
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        ids.append(nxt)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--prompt', type=int, default=48, help='Prompt length in tokens.')
    parser.add_argument('--prompt-offset', type=int, default=0,
                        help='Where in the training text the prompt starts.')
    parser.add_argument('--new-tokens', type=int, default=96)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    device, precision = setup('cpu', 'fp32', args.threads)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, implementation_sha = make_model(payload['implementation'], payload['config'], device)
    model.load_state_dict(payload['model'])
    model.eval()

    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    train_tokens = load_data()['train'][0]
    prompt_ids = train_tokens[args.prompt_offset:args.prompt_offset + args.prompt].tolist()

    size = getattr(model, 'size', 1)
    print('checkpoint   %s' % args.checkpoint)
    print('implementation %s (sha256 %s)' % (payload['implementation'], implementation_sha[:16]))
    print('members      %d   temperature %.2f   top_k %d   seed %d'
          % (size, args.temperature, args.top_k, args.seed))
    print('prompt       %r' % tokenizer.decode(prompt_ids))
    for index in range(args.samples):
        ids = generate(model, prompt_ids, args.new_tokens, args.temperature, args.top_k,
                       args.seed + index, device)
        print('\n--- sample %d ---' % (index + 1))
        print(tokenizer.decode(ids[len(prompt_ids):]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

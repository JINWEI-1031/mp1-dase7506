"""Quantify generated-text quality for a frozen checkpoint.

BPB is a proper scoring rule over the whole predictive distribution, so it is the
better measure of prediction quality and it is the only thing the leaderboard
scores. Sampling is kept as a *separate*, descriptive check with two jobs: to show
what the model actually produces rather than only reporting a scalar, and to test
whether a lower BPB corresponds to less degenerate text or is merely an artifact of
averaging probabilities.

The two metrics name the failure modes rather than score fluency, so neither can be
mistaken for a substitute leaderboard:

  replacement_rate  fraction of decoded characters that are U+FFFD, i.e. bytes the
                    tokenizer emitted that do not form valid UTF-8. This is the
                    "mojibake" failure, and it is invisible to BPB.
  repeat_4          distinct 4-grams divided by total 4-grams. Low means the model
                    has fallen into a loop ("the nest of the nest of the nest").

Prompts come from the training text, and nothing here affects any selection.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from common import ROOT, load_data, make_model, setup
from sample import generate

# Offsets into the training text chosen for ordinary English prose; a fair
# generation check should not start from a rare-script title.
PROMPT_OFFSETS = [90000, 300000, 700000, 1100000]
PROMPT_TOKENS = 48


def _longest_run(words, max_period=8):
    """Longest span occupied by a single repeating cycle.

    This is the degeneracy signature greedy decoding falls into: a phrase repeats
    until the token budget runs out. Short coincidental repeats (period 1 served
    twice, say) must not count, so a run only qualifies once it spans more than two
    full periods.
    """
    best = 0
    for start in range(len(words)):
        for period in range(1, max_period + 1):
            end = start
            while (end + period < len(words)
                   and words[end:end + period] == words[end - period:end]):
                end += period
            run = end - start
            if run > best and run > 2 * period:
                best = run
    return best


def metrics(text):
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return {
        'chars': len(text),
        'replacement_rate': text.count('�') / max(1, len(text)),
        'repeat_4_distinct_ratio': len(set(grams)) / len(grams) if grams else 1.0,
        'longest_word_run': _longest_run(words),
    }


def _decode(model, prompt_ids, new_tokens, temperature, top_k, device, seed):
    ids = generate(model, prompt_ids, new_tokens, temperature, top_k, seed, device)
    return TOKENIZER.decode(ids[len(prompt_ids):])


TOKENIZER = None


def main():
    global TOKENIZER
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', action='append', required=True, type=Path)
    parser.add_argument('--new-tokens', type=int, default=128)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    device, _ = setup('cpu', 'fp32', args.threads)
    TOKENIZER = Tokenizer.from_file(str(ROOT / 'data/tokenizer.json'))
    train = load_data()['train'][0]
    prompts = [train[off:off + PROMPT_TOKENS].tolist() for off in PROMPT_OFFSETS]

    report = {}
    for path in args.checkpoint:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model, _ = make_model(payload['implementation'], payload['config'], device)
        model.load_state_dict(payload['model'])
        model.eval()
        entry = {'members': getattr(model, 'size', 1), 'by_temperature': {}}
        for temperature in (0.0, 0.8):
            texts = [_decode(model, prompt, args.new_tokens, temperature, args.top_k,
                             device, args.seed + index)
                     for index, prompt in enumerate(prompts)]
            measured = [metrics(text) for text in texts]
            count = len(measured)
            entry['by_temperature'][str(temperature)] = {
                'replacement_rate_mean': sum(m['replacement_rate'] for m in measured) / count,
                'repeat_4_distinct_ratio_mean': sum(m['repeat_4_distinct_ratio'] for m in measured) / count,
                'longest_word_run_max': max(m['longest_word_run'] for m in measured),
                'samples': measured,
                'texts': texts,
            }
        report[str(path)] = entry
        print('%-48s members=%d' % (path.parent.name + '/' + path.name, entry['members']))
        for temperature, block in entry['by_temperature'].items():
            print('   T=%-4s  U+FFFD %6.4f   distinct-4 %6.4f   longest run %d words'
                  % (temperature, block['replacement_rate_mean'],
                     block['repeat_4_distinct_ratio_mean'], block['longest_word_run_max']))

    out = Path(__file__).resolve().parent / 'gen_quality.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('\nwrote %s' % out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

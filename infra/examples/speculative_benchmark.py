#!/usr/bin/env python3
"""Sequential, non-streaming client for an already running local vLLM server."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Name exposed by /v1/models')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=120.0)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error('Require max-tokens > 0, temperature >= 0, 0 < top-p <= 1')
    if args.output.resolve() == args.dataset.resolve():
        parser.error('Output must differ from the input dataset')
    items = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not items:
        parser.error('Dataset is empty')
    for item in items:
        if not isinstance(item.get('messages'), list) or not item['messages']:
            parser.error('Each JSONL record requires a nonempty messages list')

    def request(item):
        body = dict(model=args.model, messages=item['messages'], n=1, stream=False,
                    temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
        req = Request(args.base_url.rstrip('/') + '/chat/completions',
                      data=json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'})
        start = time.perf_counter()
        with urlopen(req, timeout=args.timeout) as response:
            result = json.load(response)
        return result, time.perf_counter() - start

    # One warmup request, excluded from records; disable global suffix history
    # for isolated experiments or warm every configuration in the same order.
    request(items[0])
    total_a = total_d = total_s = total_tokens = missing_metrics = 0
    total_seconds = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as output:
        for i, item in enumerate(items):
            result, seconds = request(item)
            spec = (result.get('metrics') or {}).get('speculative_decoding')
            tokens = (result.get('usage') or {}).get('completion_tokens')
            if not isinstance(tokens, int):
                raise ValueError('Response lacks usage.completion_tokens')
            if spec is not None:
                total_a += spec['num_accepted_draft_tokens']
                total_d += spec['num_draft_tokens']
                total_s += spec['num_spec_steps']
            else:
                missing_metrics += 1
            total_tokens += tokens
            total_seconds += seconds
            record = dict(id=item.get('id', i), label=args.label, seconds=seconds,
                          completion_tokens=tokens, speculative_decoding=spec,
                          temperature=args.temperature, top_p=args.top_p,
                          max_tokens=args.max_tokens, response=result)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
    summary = dict(label=args.label, requests=len(items), missing_metrics=missing_metrics,
                   accepted_draft_tokens=total_a, proposed_draft_tokens=total_d,
                   spec_steps=total_s, draft_acceptance_rate=total_a / total_d if total_d else None,
                   mean_acceptance_length=1 + total_a / total_s if total_s else None,
                   seconds=total_seconds, completion_tokens=total_tokens,
                   sequential_output_tokens_per_second=total_tokens / total_seconds)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Non-streaming client totals include prefill and HTTP time; these are not TTFT/TPOT.')


if __name__ == '__main__':
    main()

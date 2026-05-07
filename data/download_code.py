#!/usr/bin/env python3
"""Download a 30K code-only slice of the OpenThoughts corpus from HuggingFace.

The paper (§5.1, Table 2) trains the code distillation runs on 30K
**code-domain-only** examples drawn from ``open-thoughts/OpenThoughts3-1.2M``.
No official 30K split is published, so this script filters the 1.2M corpus
to rows whose ``domain == 'code'`` and writes the first ``--max-samples`` of
them (default 30000 to match the paper).  Math / science rows are skipped.

Usage:
    python data/download_code.py --out ./data_cache/openthoughts_code_30k.jsonl
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='Output JSONL path.')
    p.add_argument('--hf-repo', default='open-thoughts/OpenThoughts3-1.2M',
                   help='HuggingFace repository id.')
    p.add_argument('--split', default='train')
    p.add_argument('--max-samples', type=int, default=30000,
                   help='Number of code-only samples to emit (paper default: 30000).')
    p.add_argument('--domain-field', default='domain',
                   help='Column whose value must equal "code" for the row to be kept.')
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.hf_repo, split=args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_non_code = 0
    with out.open('w', encoding='utf-8') as fh:
        for row in ds:
            if str(row.get(args.domain_field, '')).lower() != 'code':
                skipped_non_code += 1
                continue
            messages = _row_to_messages(row)
            if messages is None:
                continue
            fh.write(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')
            written += 1
            if args.max_samples and written >= args.max_samples:
                break
    print(f'Wrote {written} code-only samples -> {out}  '
          f'(skipped {skipped_non_code} non-code rows)')


def _row_to_messages(row):
    conv = row.get('conversations') or []
    role_map = {'user': 'user', 'human': 'user', 'assistant': 'assistant', 'gpt': 'assistant'}
    msgs = [{'role': role_map[t['from']], 'content': t['value']}
            for t in conv if t.get('from') in role_map]
    if msgs and msgs[-1]['role'] == 'assistant':
        return msgs
    return None


if __name__ == '__main__':
    main()

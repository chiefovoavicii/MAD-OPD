#!/usr/bin/env python3
"""Download the ToolACE agentic corpus from HuggingFace.

Consumed by ``data/toolace_dataset.py`` (registered at training time via
``--custom_register_path data/toolace_dataset.py``).

Usage:
    python data/download_toolace.py --out ./data_cache/toolace/data.jsonl
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, help='Output JSONL path.')
    p.add_argument('--hf-repo', default='Team-ACE/ToolACE',
                   help='HuggingFace repository id.')
    p.add_argument('--split', default='train')
    p.add_argument('--max-samples', type=int, default=0,
                   help='Cap number of samples (0 = keep all).')
    args = p.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.hf_repo, split=args.split)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open('w', encoding='utf-8') as fh:
        for row in ds:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
            if args.max_samples and written >= args.max_samples:
                break
    print(f'Wrote {written} samples -> {out}')


if __name__ == '__main__':
    main()

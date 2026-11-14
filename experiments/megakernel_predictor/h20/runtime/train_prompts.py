#!/usr/bin/env python3
"""Recreate the pinned training prompts from the public WikiText rows."""
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROWS = (3, 4, 11, 12, 16, 17, 34, 35, 39, 43, 45, 46, 47, 48, 49, 50)
URL = ('https://datasets-server.huggingface.co/first-rows?dataset=Salesforce%2Fwikitext'
       '&config=wikitext-2-raw-v1&split=test')
DIGEST = '66a8a1cecb8db427ec4de6e31b274c1a9a2d652b2f723a473a7d84ca363b4794'


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('usage: train_prompts.py OUTPUT [WIKITEXT_FIRST_ROWS_JSON]')
    if len(sys.argv) == 3:
        rows = json.loads(Path(sys.argv[2]).read_text())['rows']
    else:
        with urlopen(URL, timeout=30) as response:
            rows = json.load(response)['rows']
    indexed = {row['row_idx']: row['row']['text'] for row in rows}
    prompts = ''.join('Continue the following article: ' + ' '.join(indexed[i].split())[:950] + '\n'
                      for i in ROWS).encode()
    if hashlib.sha256(prompts).hexdigest() != DIGEST:
        raise ValueError('WikiText rows do not reproduce the experiment training prompts')
    Path(sys.argv[1]).write_bytes(prompts)


if __name__ == '__main__':
    main()

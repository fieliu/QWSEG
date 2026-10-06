#!/usr/bin/env python3
"""Extract only trained modality-adaptation weights from a Stage-1 checkpoint.

Stage 1 contains a complete model state for resumability, including frozen and
unused randomly initialized fusion/decoder parameters.  Loading that full file
into Stage 2 would overwrite the corrected Stage-2 initialization.  This tool
keeps only the thermal patch embedding, thermal modality code, and thermal
adapters that Stage 1 actually learned for inference.
"""
import argparse
from pathlib import Path

import torch


TRANSFER_PREFIXES = (
    'backbone.patch_embed.thermal.',
    'backbone.modality_embed.thermal',
    'backbone.adapters.thermal.',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    selected = {
        name: value for name, value in state.items()
        if name.startswith(TRANSFER_PREFIXES)
    }
    if not selected:
        raise ValueError(f'no DINO-TS Stage-1 weights found in {args.input}')

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': selected,
        'meta': {
            'source': str(args.input.resolve()),
            'purpose': 'DINO-TS Stage1 -> Stage2 learned thermal transfer',
            'selected_prefixes': TRANSFER_PREFIXES,
        },
    }, args.output)
    print(f'kept {len(selected)} tensors: {args.output}')


if __name__ == '__main__':
    main()

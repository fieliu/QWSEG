"""Verify a trusted local smoke checkpoint actually contains optimizer updates."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--teacher', type=Path,
                        help='Assert the frozen teacher still matches this checkpoint exactly.')
    parser.add_argument('--require-router-update', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    bad = [name for name, value in state.items()
           if value.is_floating_point() and not torch.isfinite(value).all()]
    assert not bad, f'Non-finite model tensors: {bad[:10]}'
    optimizer = checkpoint['optimizer']
    steps = [int(value['step']) for value in optimizer['state'].values()
             if 'step' in value]
    assert steps and min(steps) == max(steps) == args.steps, (
        f'Expected {args.steps} actual updates, got {sorted(set(steps))}')
    result = dict(checkpoint=str(args.checkpoint), finite=True,
                  updated_parameters=len(steps), optimizer_steps=args.steps)
    if args.teacher:
        source = torch.load(args.teacher, map_location='cpu', weights_only=False)['state_dict']
        changed = [name for name, value in source.items()
                   if 'teacher.' + name not in state
                   or not torch.equal(value, state['teacher.' + name])]
        assert not changed, f'Frozen teacher changed or missing: {changed[:10]}'
        result['frozen_teacher_matches'] = True
        router_changed = [name for name in source if name.startswith('backbone.router.')
                          and not torch.equal(source[name], state[name])]
        result['router_updated_tensors'] = len(router_changed)
        if args.require_router_update:
            assert router_changed, 'Soft router did not update'
    elif args.require_router_update:
        parser.error('--require-router-update requires --teacher')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

"""Regression check: padding must not be corrupted or treated as thermal data."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rgbt_c.protocol import operation, recipe
from rgbt_c.torch_adapter import RGBTDegrader


def main():
    torch.manual_seed(3)
    mean = [123.675, 116.28, 103.53] * 2
    std = [58.395, 57.12, 57.375] * 2
    mean_t = torch.tensor(mean).view(1, 6, 1, 1)
    std_t = torch.tensor(std).view(1, 6, 1, 1)
    raw_rgb = torch.randint(0, 256, (1, 3, 20, 24)).float()
    raw_t = torch.randint(0, 256, (1, 1, 20, 24)).float().repeat(1, 3, 1, 1)
    inputs = (torch.cat([raw_rgb, raw_t], 1) - mean_t) / std_t
    padded = F.pad(inputs, (0, 8, 0, 12), value=0)
    degrader = RGBTDegrader(mean, std)
    for name in ['gaussian_noise', 't_gaussian_noise', 'rgb_missing', 't_missing']:
        spec = recipe([operation(name, 1, 42)])
        expected = degrader.apply(inputs[:, :3], inputs[:, 3:], [spec])
        actual = degrader.apply(padded[:, :3], padded[:, 3:], [spec], [(20, 24)])
        for result, reference in zip(actual, expected):
            assert torch.equal(result[:, :, :20, :24], reference), name
            assert torch.count_nonzero(result[:, :, 20:, :]) == 0, name
            assert torch.count_nonzero(result[:, :, :, 24:]) == 0, name
    clean = degrader.apply(padded[:, :3], padded[:, 3:], [recipe([])], [(20, 24)])
    assert torch.equal(torch.cat(clean[:2], 1), padded)
    try:
        degrader.apply(padded[:, :3], padded[:, 3:], [recipe([])], [(33, 24)])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid image shape must be rejected')
    print('PADDING_OK: crop equivalence, missing modalities, clean identity, bounds')


if __name__ == '__main__':
    main()

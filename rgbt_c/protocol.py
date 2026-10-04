"""Versioned, replayable RGB-T recipes shared by training and benchmarks.

Pixels are uint8 RGB and one-channel thermal. Operators use private random
generators: replay does not change the caller's NumPy/Python random state.
"""
import hashlib
import json

import cv2
import numpy as np

from .corruptions import ALL_CORRUPTIONS, RGB_CORRUPTIONS, get_corruption

PROTOCOL_VERSION = 'rgbtc-2.0'
MISSING = ('rgb_missing', 't_missing')
# An explicit default research split, not a claim about an external benchmark.
HELDOUT_CORRUPTIONS = ('fog', 'stripe_noise', 't_quantization')
TRAIN_CORRUPTIONS = tuple(c for c in ALL_CORRUPTIONS
                          if c not in MISSING + HELDOUT_CORRUPTIONS)
SCOPES = ('global', 'local', 'local_soft')


def stable_seed(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(payload.encode('utf-8')).digest()[:4], 'big')


def resolve_corruptions(names):
    groups = {'all': ALL_CORRUPTIONS, 'seen': TRAIN_CORRUPTIONS,
              'heldout': HELDOUT_CORRUPTIONS, 'missing': MISSING,
              'rgb': RGB_CORRUPTIONS,
              't': [c for c in ALL_CORRUPTIONS if c not in RGB_CORRUPTIONS]}
    result = []
    for name in names:
        for value in groups.get(name, [name]):
            if value not in ALL_CORRUPTIONS:
                raise ValueError(f'Unknown corruption: {value}')
            if value not in result:
                result.append(value)
    return result


def operation(name, severity, seed, scope='global', area_range=(0.1, 0.6)):
    if name not in ALL_CORRUPTIONS or scope not in SCOPES:
        raise ValueError(f'Invalid corruption/scope: {name}/{scope}')
    if not isinstance(severity, (int, np.integer)) or not 1 <= severity <= 5:
        raise ValueError('severity must be an integer in [1, 5]')
    if not 0 < area_range[0] <= area_range[1] <= 1:
        raise ValueError('area_range must be inside (0, 1]')
    spec = dict(name=name, severity=1 if name in MISSING else int(severity),
                seed=int(seed), scope=scope)
    if scope != 'global':
        rng = np.random.default_rng(stable_seed(seed, 'region'))
        area = rng.uniform(*area_range)
        aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        height = min(1., np.sqrt(area * aspect))
        width = min(1., area / height)
        height = min(1., area / width)
        spec['region'] = [float(rng.uniform(0, 1-height)),
                          float(rng.uniform(0, 1-width)), float(height), float(width)]
        spec['feather'] = 0.03 if scope == 'local_soft' else 0.
    return spec


def recipe(operations=()):
    return dict(version=PROTOCOL_VERSION, operations=list(operations))


def spatial_mask(shape, spec):
    h, w = shape[:2]
    if spec['scope'] == 'global':
        return np.ones((h, w, 1), dtype=np.float32)
    y, x, rh, rw = spec['region']
    if not (0 <= y < 1 and 0 <= x < 1 and rh > 0 and rw > 0
            and y + rh <= 1.000001 and x + rw <= 1.000001):
        raise ValueError('Invalid normalized region')
    top, left = min(h-1, int(y*h)), min(w-1, int(x*w))
    bottom, right = min(h, max(top+1, round((y+rh)*h))), min(w, max(left+1, round((x+rw)*w)))
    mask = np.zeros((h, w), dtype=np.float32)
    mask[top:bottom, left:right] = 1.
    if spec['scope'] == 'local_soft':
        sigma = max(0.5, float(spec.get('feather', .03)) * min(h, w))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    return mask[..., None]


def apply_pair(rgb, thermal, spec, return_masks=False):
    """Apply an explicit recipe; return RGB + single-channel T, optionally masks.

    Masks describe applied strength, not semantic reliability or true quality.
    Missing always means zero *raw pixels*, in both training and evaluation.
    """
    if spec.get('version') != PROTOCOL_VERSION:
        raise ValueError('Recipe/library version mismatch')
    if rgb.dtype != np.uint8 or thermal.dtype != np.uint8:
        raise TypeError('RGB and thermal must be uint8 raw pixels')
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected RGB HxWx3')
    if thermal.ndim == 2:
        thermal = thermal[..., None]
    if thermal.shape != rgb.shape[:2] + (1,):
        raise ValueError('Expected aligned, single-channel thermal HxWx1')
    outputs = [rgb.copy(), thermal.copy()]
    masks = [np.zeros(rgb.shape[:2] + (1,), np.float32) for _ in range(2)]
    for op in spec['operations']:
        if op['scope'] not in SCOPES:
            raise ValueError('Unknown spatial scope')
        idx = 0 if op['name'] in RGB_CORRUPTIONS else 1
        corr = get_corruption(op['name'])
        changed = corr(outputs[idx], op['severity'], rng=np.random.default_rng(op['seed']))
        mask = spatial_mask(rgb.shape, op)
        outputs[idx] = np.rint(outputs[idx] * (1-mask) + changed * mask).clip(0, 255).astype(np.uint8)
        level = 5 if op['name'] in MISSING else op['severity']
        masks[idx] = np.maximum(masks[idx], mask * level)
    return (*outputs, *masks) if return_masks else tuple(outputs)


def sample_recipe(rng, corruptions=TRAIN_CORRUPTIONS, degrade_prob=.8,
                  modality_probs=(.45, .45, .10), scope_probs=(.5, .25, .25),
                  severity_range=(1, 5), area_range=(.1, .6), missing_prob=.1):
    """Training-only stochastic policy. Benchmark recipes never call this."""
    if rng.random() >= degrade_prob:
        return recipe()
    which = rng.choice(['rgb', 't', 'both'], p=modality_probs)
    scope = str(rng.choice(SCOPES, p=scope_probs))
    ops = []
    for mod in (['rgb', 't'] if which == 'both' else [which]):
        names = [c for c in corruptions if (c in RGB_CORRUPTIONS) == (mod == 'rgb')]
        if not names:
            raise ValueError(f'No training corruptions configured for {mod}')
        # Never remove both sensors completely: paired cases use corruptions.
        missing = which != 'both' and rng.random() < missing_prob
        name = ('rgb_missing' if mod == 'rgb' else 't_missing') if missing else str(rng.choice(names))
        severity = int(rng.integers(severity_range[0], severity_range[1] + 1))
        ops.append(operation(name, severity, int(rng.integers(0, 2**32)), scope, area_range))
    return recipe(ops)


def benchmark_cases(names, severities=(1, 2, 3, 4, 5), scopes=('global',), paired=()):
    cases = [dict(id='clean', name='clean', severity=0, scope='global', group='clean', names=[])]
    combos = [[name] for name in resolve_corruptions(names)]
    for pair in paired:
        parts = pair.split('+')
        if (len(parts) != 2 or any(p not in ALL_CORRUPTIONS for p in parts)
                or (parts[0] in RGB_CORRUPTIONS) == (parts[1] in RGB_CORRUPTIONS)
                or all(p in MISSING for p in parts)):
            raise ValueError(f'Expected one RGB + one T corruption: {pair}')
        combos.append(parts)
    if not severities or any(s not in range(1, 6) for s in severities):
        raise ValueError('Benchmark severities must be in [1, 5]')
    for scope in dict.fromkeys(scopes):
        if scope not in SCOPES:
            raise ValueError(f'Unknown scope: {scope}')
        for parts in combos:
            name = '+'.join(parts)
            if scope != 'global':
                name = scope + '__' + name
            group = ('missing' if all(c in MISSING for c in parts) else
                     'both' if len(parts) == 2 else
                     'heldout' if parts[0] in HELDOUT_CORRUPTIONS else 'seen')
            for severity in ([1] if group == 'missing' else sorted(set(severities))):
                cases.append(dict(id=f'{name}/{severity}', name=name, severity=severity,
                                  scope=scope, group=group, names=parts))
    # Duplicate aliases or paired entries must not reweight the report.
    return list({case['id']: case for case in cases}.values())


def benchmark_recipe(case, image_id, seed=42):
    # Exclude severity: same spatial mask/noise source across severity levels.
    return recipe(operation(name, case['severity'],
                            stable_seed(PROTOCOL_VERSION, seed, image_id, case['name'], name),
                            case['scope']) for name in case['names'])

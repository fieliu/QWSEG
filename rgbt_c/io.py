"""Explicit file color conventions and Unicode-safe PNG I/O."""
import hashlib
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit image: {path}')
    return image


def unpack_4ch(image):
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError('Expected a four-channel BGR+T PNG')
    return image[:, :, :3][:, :, ::-1].copy(), image[:, :, 3:4].copy()


def pack_4ch(rgb, thermal):
    return np.concatenate([rgb[:, :, ::-1], thermal], axis=2)


def write_png(path, image):
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise OSError(f'PNG encoding failed: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def library_digest():
    folder = Path(__file__).parent
    pairs = [(p.name, sha256_file(p)) for p in sorted(folder.glob('*.py'))]
    return hashlib.sha256(repr(pairs).encode()).hexdigest()

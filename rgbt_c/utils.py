"""RGBT-C 辅助函数 (对齐 ImageNet-C 实现)."""
import numpy as np
import cv2
from scipy.ndimage import zoom as scizoom


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """生成 disk-shaped PSF (对齐 ImageNet-C defocus_blur).

    Args:
        radius: 离焦圆半径
        alias_blur: 抗锯齿高斯 sigma
    Returns:
        (2*radius+1, 2*radius+1) 归一化核
    """
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    # supersample to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def plasma_fractal(mapsize=256, wibbledecay=3):
    """diamond-square 高度图 (对齐 ImageNet-C fog).

    Return square 2d array of floats in range 0-1.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize, stepsize // 2:mapsize_:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_:stepsize, 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    """中心裁剪 + 放大 (对齐 ImageNet-C zoom_blur)."""
    h = img.shape[0]
    ch = int(np.ceil(h / float(zoom_factor)))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], zoom_factor, order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def motion_blur_kernel(length, angle, sigma):
    """构造运动模糊核, 复现 ImageMagick MotionBlur(radius, sigma, angle).

    ImageMagick MotionBlur 语义:
      - radius: 运动轨迹单边延伸长度 (像素), kernel 大小 = 2*radius+1
      - sigma:  沿运动方向的高斯加权 sigma
      - angle:  运动方向角度 (度)

    Args:
        length: 运动轨迹单边长度 (即 ImageMagick 的 radius)
        angle:  运动方向角度 (度)
        sigma:  高斯加权 sigma
    Returns:
        (ksize, ksize) 归一化核, ksize = 2*length+1
    """
    # ImageMagick: kernel 大小 = 2*radius+1
    ksize = 2 * length + 1
    half = length
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    # 沿水平方向画一条高斯加权的线 (中心行)
    for i in range(ksize):
        d = i - half
        kernel[half, i] = np.exp(-(d * d) / (2.0 * sigma * sigma))
    # 旋转到指定角度
    M = cv2.getRotationMatrix2D((half, half), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize), flags=cv2.INTER_LINEAR)
    # 归一化
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return kernel


def apply_blur_per_channel(img, kernel):
    """对多通道图像逐通道卷积.

    Args:
        img: (H, W, C) float32 in [0, 1]
        kernel: 2D float32
    Returns:
        (H, W, C) float32
    """
    if img.ndim == 2:
        return cv2.filter2D(img, -1, kernel)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
    return out


def to_float01(img):
    """uint8 [0,255] -> float32 [0,1]."""
    return np.array(img, dtype=np.float32) / 255.0


def to_uint8(img):
    """float [0,1] -> uint8 [0,255]."""
    return np.clip(img, 0, 1) * 255


def ensure_3ch(img):
    """单通道 -> 3 通道 (用于统一处理)."""
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img


def restore_ch(img, target_ch):
    """恢复原始通道数 (3->1 取平均, 或保持)."""
    if target_ch == 1 and img.shape[2] == 3:
        # T 单通道: 取平均 (因为输入是单通道复制成 3 通道的)
        return img[:, :, :1]
    return img

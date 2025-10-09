from typing import Optional
import numpy as np

try:
    import rasterio
    from rasterio.enums import Resampling
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

try:
    from skimage import io, util
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def read_preview(
    path: str,
    downsample: int = 8,
    max_side: Optional[int] = None,
    resampling: str = "lanczos",
) -> np.ndarray:
    """Return a small uint8 preview (HxW or HxWxC) for plotting.

    - If rasterio is available it reads a resampled preview (memory-safe).
    - Otherwise it falls back to scikit-image and simple slicing.

    Parameters
    - path: image path
    - downsample: integer factor to downsample by when max_side is not set
    - max_side: if set, scale such that the longest side <= max_side
    """
    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            if max_side is not None:
                target = max(1, int(max_side))
                scale = max(1, max(src.width, src.height) // target)
            else:
                scale = max(1, int(downsample))

            out_h = max(1, src.height // scale)
            out_w = max(1, src.width // scale)
            out_shape = (src.count, out_h, out_w)
            # map user string to rasterio Resampling enum if possible
            try:
                resamp_enum = getattr(Resampling, resampling)
            except Exception:
                resamp_enum = Resampling.average
            arr = src.read(out_shape=out_shape, resampling=resamp_enum)
            img = np.transpose(arr, (1, 2, 0)) if arr.shape[0] > 1 else arr[0]

            # normalize to 0..255 and convert to uint8
            img = img.astype('float32')
            ptp = np.ptp(img)
            if ptp > 0:
                img = (img - np.min(img)) / ptp
            else:
                img = img - np.min(img)
            img = (img * 255).clip(0, 255).astype('uint8')
            return img

    if _HAS_SKIMAGE:
        img = io.imread(path)
        if max_side is not None:
            h, w = img.shape[:2]
            factor = max(1, max(h, w) // int(max_side))
        else:
            factor = max(1, int(downsample))

        if factor > 1:
            img = img[::factor, ::factor]

        img = util.img_as_ubyte(img)
        return img

    raise RuntimeError(
        "Neither rasterio nor scikit-image is available in the environment."
    )


def read_full(path: str, to_hwc: bool = True, check_memory: bool = True):
    """Read the entire image into memory.

    - to_hwc: if True, return HxWxC for multi-band images; otherwise return
      (bands, H, W) when rasterio is used.
    - check_memory: if True and psutil is available, estimate required bytes
      and raise MemoryError if available RAM is insufficient.
    """
    # memory guard (optional)
    if check_memory:
        try:
            import psutil
        except Exception:
            psutil = None

        if psutil is not None and _HAS_RASTERIO:
            with rasterio.open(path) as src:
                w, h, b = src.width, src.height, src.count
                item = np.dtype(src.dtypes[0]).itemsize
            need = int(w) * int(h) * int(b) * int(item)
            avail = psutil.virtual_memory().available
            # require a small safety margin
            if need * 1.1 > avail:
                raise MemoryError(
                    f"Not enough RAM to read full image: need {need} bytes, "
                    f"available {avail} bytes"
                )

    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
        if to_hwc and arr.shape[0] > 1:
            return np.transpose(arr, (1, 2, 0))
        return arr

    if _HAS_SKIMAGE:
        img = io.imread(path)
        # skimage returns HxW or HxWxC already
        if to_hwc and img.ndim == 3 and img.shape[2] <= 4:
            return img
        return img

    raise RuntimeError("No backend available to read full image.")

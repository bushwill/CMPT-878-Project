"""Small image helper utilities moved out of the notebook.

All imports are at module level (guarded). Functions do not import.
"""
from typing import Optional
import math
import numpy as np

# Guarded optional imports (module-level)
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    psutil = None
    _HAS_PSUTIL = False

try:
    import rasterio
    from rasterio.enums import Resampling
    _HAS_RASTERIO = True
except Exception:
    rasterio = None
    Resampling = None
    _HAS_RASTERIO = False

try:
    from skimage import io, util
    _HAS_SKIMAGE = True
except Exception:
    io = None
    util = None
    _HAS_SKIMAGE = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    from io import BytesIO
    from IPython.display import display, Image as IPyImage
    _HAS_IPYTHON = True
except Exception:
    BytesIO = None
    display = None
    IPyImage = None
    _HAS_IPYTHON = False


def read_preview(
    path: str,
    downsample: int = 8,
    max_side: Optional[int] = None,
    resampling: str = "lanczos",
) -> np.ndarray:
    """Return a small uint8 preview (H x W or H x W x C) suitable for plots.

    Uses rasterio (resampled read) if available, otherwise falls back to
    scikit-image's ``io.imread`` and simple subsampling.
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
            try:
                resamp_enum = getattr(Resampling, resampling)
            except Exception:
                resamp_enum = Resampling.average
            arr = src.read(out_shape=out_shape, resampling=resamp_enum)
            img = np.transpose(arr, (1, 2, 0)) if arr.shape[0] > 1 else arr[0]

            img = img.astype("float32")
            ptp = np.ptp(img)
            if ptp > 0:
                img = (img - np.min(img)) / ptp
            else:
                img = img - np.min(img)
            img = (img * 255).clip(0, 255).astype("uint8")
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

    raise RuntimeError("Neither rasterio nor scikit-image is available.")


def read_full(path: str, to_hwc: bool = True, check_memory: bool = True):
    """Read a whole image into memory.

    If ``check_memory`` and ``psutil`` are available, estimate memory needed
    and raise MemoryError if insufficient.
    """
    if check_memory and _HAS_PSUTIL and _HAS_RASTERIO:
        with rasterio.open(path) as src:
            w, h, b = src.width, src.height, src.count
            item = np.dtype(src.dtypes[0]).itemsize
        need = int(w) * int(h) * int(b) * int(item)
        avail = psutil.virtual_memory().available
        # Convert bytes to gigabytes for clearer messaging
        need_gb = need / (1024 ** 3)
        avail_gb = avail / (1024 ** 3)
        if need * 1.1 > avail:
            raise MemoryError(
                f"Not enough RAM to read full image: need {need} bytes ({need_gb:.2f} GB), "
                f"available {avail} bytes ({avail_gb:.2f} GB). "
                "Try increasing available memory or use `read_preview()`/rasterio to read a smaller window."
            )

    if _HAS_RASTERIO:
        with rasterio.open(path) as src:
            arr = src.read()  # (bands, H, W)
        if to_hwc and arr.shape[0] > 1:
            return np.transpose(arr, (1, 2, 0))
        return arr

    if _HAS_SKIMAGE:
        img = io.imread(path)
        return img

    raise RuntimeError("No supported backend available to read full image.")


def split_into_n(img: np.ndarray, n: int = 32) -> np.ndarray:
    """Split an image into approximately n tiles and return an object array.

    Returns a 2D numpy.ndarray of dtype object with tiles or None for extras.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    h, w = img.shape[:2]

    r = int(math.floor(math.sqrt(n)))
    if r == 0:
        rows, cols = 1, n
    else:
        cols = int(math.ceil(n / r))
        rows = r

    tile_h = math.ceil(h / rows)
    tile_w = math.ceil(w / cols)

    grid = []
    count = 0
    for ri in range(rows):
        row_tiles = []
        for ci in range(cols):
            if count >= n:
                row_tiles.append(None)
                count += 1
                continue
            y0 = ri * tile_h
            x0 = ci * tile_w
            y1 = min(h, y0 + tile_h)
            x1 = min(w, x0 + tile_w)
            tile = img[y0:y1, x0:x1]
            row_tiles.append(tile if tile.size != 0 else None)
            count += 1
        grid.append(row_tiles)

    return np.array(grid, dtype=object)


def get_tile(img: np.ndarray, row: int, col: int, n: int = 32, base: int = 0) -> Optional[np.ndarray]:
    """Return a single tile (or None) by row/col from an n-split grid."""
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be a numpy array")
    if base not in (0, 1):
        raise ValueError("base must be 0 or 1")
    if base == 1:
        row -= 1
        col -= 1

    r = int(math.floor(math.sqrt(n)))
    if r == 0:
        rows, cols = 1, n
    else:
        cols = int(math.ceil(n / r))
        rows = r

    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise IndexError("row/col out of range for computed grid")

    h, w = img.shape[:2]
    tile_h = math.ceil(h / rows)
    tile_w = math.ceil(w / cols)

    y0 = row * tile_h
    x0 = col * tile_w
    y1 = min(h, y0 + tile_h)
    x1 = min(w, x0 + tile_w)
    tile = img[y0:y1, x0:x1]
    return tile if tile.size != 0 else None


def save_tile(tile: np.ndarray, out_path: str, force_uint8: bool = True):
    """Save tile to disk. Uses skimage.io if available, else numpy.save."""
    if tile is None:
        raise ValueError("tile is None")
    if force_uint8:
        try:
            out = util.img_as_ubyte(tile)
        except Exception:
            t = tile.astype("float32")
            t = (t - t.min()) / max(1e-9, t.ptp())
            out = (t * 255).astype("uint8")
    else:
        out = tile

    if _HAS_SKIMAGE:
        io.imsave(out_path, out)
    else:
        np.save(out_path, out)


def display_image(img: np.ndarray, figsize=(8, 8), title: Optional[str] = None,
                  cmap: Optional[str] = None, interpolation: str = "nearest"):
    """Display an image array via matplotlib when available."""
    if img is None:
        raise ValueError("img is None")

    try:
        disp = util.img_as_ubyte(img)
    except Exception:
        t = img.astype("float32")
        t = (t - t.min()) / max(1e-9, t.ptp())
        disp = (t * 255).astype("uint8")

    if not _HAS_MPL:
        # If matplotlib isn't available, try IPython display if present
        if _HAS_IPYTHON and BytesIO is not None:
            buf = BytesIO()
            if _HAS_SKIMAGE:
                io.imsave(buf, disp, plugin="pil")
            else:
                # fallback to a very small matplotlib-free write using PIL via skimage
                try:
                    from PIL import Image as PILImage
                    PILImage.fromarray(disp).save(buf, format="PNG")
                except Exception:
                    raise RuntimeError("No plotting backend available")
            buf.seek(0)
            display(IPyImage(data=buf.read()))
            return
        raise RuntimeError("matplotlib is required for display_image()")

    plt.figure(figsize=figsize)
    if disp.ndim == 2:
        plt.imshow(disp, cmap=cmap or "gray", interpolation=interpolation)
    else:
        plt.imshow(disp, interpolation=interpolation)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def display_tile_fullres(tile: np.ndarray, title: Optional[str] = None):
    """Attempt to display the tile at native resolution (not resampled)."""
    if tile is None:
        raise ValueError("tile is None")

    try:
        disp = util.img_as_ubyte(tile)
    except Exception:
        t = tile.astype("float32")
        t = (t - t.min()) / max(1e-9, t.ptp())
        disp = (t * 255).astype("uint8")

    if _HAS_IPYTHON and BytesIO is not None:
        buf = BytesIO()
        if _HAS_SKIMAGE:
            io.imsave(buf, disp, plugin="pil")
        elif _HAS_MPL:
            plt.imsave(buf, disp, cmap="gray" if disp.ndim == 2 else None)
        else:
            try:
                from PIL import Image as PILImage
                PILImage.fromarray(disp).save(buf, format="PNG")
            except Exception:
                pass
        buf.seek(0)
        if display is not None:
            display(IPyImage(data=buf.read()))
            if title:
                print(title)
            return

    if not _HAS_MPL:
        raise RuntimeError("No display backend available")

    plt.figure(figsize=(disp.shape[1] / 100, disp.shape[0] / 100))
    if disp.ndim == 2:
        plt.imshow(disp, cmap="gray", interpolation="nearest")
    else:
        plt.imshow(disp, interpolation="nearest")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

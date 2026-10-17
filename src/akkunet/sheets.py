"""Contact sheets: a grid of image tiles with a caption, for eyeballing data."""
import cv2
import numpy as np


def crop_box(image: np.ndarray, cx: float, cy: float, w: float, h: float, pad: float = 0.25) -> np.ndarray:
    """Crop a YOLO box (normalised centre/size) with `pad` x box size of context around it."""
    H, W = image.shape[:2]
    bw, bh = w * W, h * H
    x0 = int(max(0, cx * W - bw / 2 - pad * bw))
    x1 = int(min(W, cx * W + bw / 2 + pad * bw))
    y0 = int(max(0, cy * H - bh / 2 - pad * bh))
    y1 = int(min(H, cy * H + bh / 2 + pad * bh))
    return image[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]


def tile(image: np.ndarray, caption: str, size: int = 220) -> np.ndarray:
    """Fit an image into a size x size square (letterboxed, grey) with a caption strip below."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    out = np.full((size + 22, size, 3), 60, dtype=np.uint8)
    y, x = (size - small.shape[0]) // 2, (size - small.shape[1]) // 2
    out[y:y + small.shape[0], x:x + small.shape[1]] = small
    cv2.putText(out, caption[:34], (4, size + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def grid(tiles: list[np.ndarray], cols: int = 5) -> np.ndarray:
    """Arrange equal-sized tiles in rows of `cols`, padding the last row."""
    if not tiles:
        raise ValueError("No tiles to arrange")
    blank = np.full_like(tiles[0], 30)
    tiles = tiles + [blank] * (-len(tiles) % cols)
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)

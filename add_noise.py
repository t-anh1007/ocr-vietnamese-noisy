"""Them vat can len anh: nen hoa van, watermark cheo, con dau do.
Dung: python add_noise.py   (doc samples/, ghi samples_noisy/)
"""
import cv2, numpy as np, shutil
from pathlib import Path

src, dst = Path("samples"), Path("samples_noisy")
dst.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

for p in src.glob("*.jpg"):
    img = cv2.imread(str(p)); h, w = img.shape[:2]
    # 1) hoa van: luoi song sin mau nhat tron len anh
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = ((np.sin(xx / 6) + np.sin(yy / 9) + np.sin((xx + yy) / 15)) * 40 + 128).astype(np.uint8)
    pattern = cv2.applyColorMap(pattern, cv2.COLORMAP_PINK)
    img = cv2.addWeighted(img, 0.75, pattern, 0.25, 0)
    # 2) watermark cheo
    wm = img.copy()
    for y in range(0, h, max(h // 4, 40)):
        cv2.putText(wm, "BAN SAO - MAU THU", (10, y), cv2.FONT_HERSHEY_SIMPLEX, w / 600, (180, 180, 180), 3)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 25, 1)
    wm = cv2.warpAffine(wm, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    img = cv2.addWeighted(img, 0.8, wm, 0.2, 0)
    # 3) con dau do de len chu
    c = (int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4)))
    r = min(h, w) // 6
    cv2.circle(img, c, r, (30, 30, 220), max(r // 10, 3))
    cv2.circle(img, c, int(r * 0.75), (30, 30, 220), max(r // 20, 2))
    cv2.putText(img, "DA DUYET", (c[0] - r // 2, c[1] + r // 10), cv2.FONT_HERSHEY_SIMPLEX, r / 90, (30, 30, 220), 3)
    cv2.imwrite(str(dst / p.name), img)

shutil.copy(src / "ground_truth.json", dst / "ground_truth.json")
print("Da tao anh nhieu ->", dst)

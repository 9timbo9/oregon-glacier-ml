"""
Train a small U-Net on RGB+NDSI inputs using pseudo-labels generated from NDSI + QA.

Expected .npz per patch (like yours):
  - rgb: (H,W,3) float32 in [0,1]
  - ndsi: (H,W) float32 ~[-1,1]
  - qa_good: (H,W) uint8 0/1 (1 = good pixels)
Optional:
  - qa_pixel: (H,W) uint16 (kept but not required)

This trains a segmentation model:
  input 4 channels: [R,G,B,NDSI]
  output 1 channel: glacier mask probability
"""

import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds

# ---- optional: for mask cleanup (recommended) ----
try:
    from scipy import ndimage as ndi
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ==========================
# Config
# ==========================
SCRIPT_DIR = Path(__file__).resolve().parent

DEM_PATH = SCRIPT_DIR / "data" /"DEM" / "output_hh.tif"  
USE_SLOPE = True  # set False if you only want DEM
USE_ASPECT = True

YEARS = ['1980','2000','2020','Wallowas']  


PATCH_DIR = SCRIPT_DIR / "patches"/ YEARS[3]  # change index to select year
OUT_DIR   = SCRIPT_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
LR = 1e-3
RANDOM_CROP = 256          # training crop size (square); set None to use full image
NDSI_THRESHOLD = 0.25
MIN_OBJECT_PIXELS = 150
EPOCHS = 30# remove tiny specks in pseudo-labels (tune)


def parse_meta_txt(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = float(v.strip())
    return meta

def sample_dem_to_patch(dem_src, meta: dict, H: int, W: int) -> np.ndarray:
    # patch bounds in EPSG:4326
    left, right = meta["lon_min"], meta["lon_max"]
    bottom, top = meta["lat_min"], meta["lat_max"]

    # bounds in DEM CRS
    b = transform_bounds("EPSG:4326", dem_src.crs, left, bottom, right, top, densify_pts=21)

    # destination grid exactly HxW
    dst_transform = from_bounds(b[0], b[1], b[2], b[3], W, H)
    dst = np.empty((H, W), dtype=np.float32)

    reproject(
        source=rasterio.band(dem_src, 1),
        destination=dst,
        src_transform=dem_src.transform,
        src_crs=dem_src.crs,
        dst_transform=dst_transform,
        dst_crs=dem_src.crs,
        resampling=Resampling.bilinear,
    )

    # nodata -> nan
    if dem_src.nodata is not None:
        dst[dst == dem_src.nodata] = np.nan
    return dst.astype(np.float32)

def robust_norm(x: np.ndarray, p1=2, p2=98) -> np.ndarray:
    lo = np.nanpercentile(x, p1)
    hi = np.nanpercentile(x, p2)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0, 1).astype(np.float32)

def terrain_slope_aspect(dem_m: np.ndarray):
    """
    Returns:
      slope_deg: float32
      aspect_deg: float32 in [0,360)
    Note: gradient uses pixel units; still very useful as relative terrain feature.
    """
    dem = dem_m.copy()
    nanmask = np.isnan(dem)
    if nanmask.any():
        dem[nanmask] = np.nanmedian(dem)

    dz_dy, dz_dx = np.gradient(dem)

    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)

    # Aspect (0 = north, 90 = east), common GIS convention-ish
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360
    aspect_deg = aspect_deg.astype(np.float32)

    return slope_deg, aspect_deg

def aspect_to_sin_cos(aspect_deg: np.ndarray):
    ang = np.deg2rad(aspect_deg.astype(np.float32))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)

# ==========================
# Utilities: pseudo-labels
# ==========================

def make_pseudolabel(ndsi: np.ndarray, qa_good: np.ndarray, t: float) -> np.ndarray:
    """
    Returns binary mask (H,W) uint8 where 1 = glacier candidate.
    """
    mask = (ndsi > t) & (qa_good.astype(bool))

    # Clean up speckles / holes if scipy available
    if SCIPY_OK:
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
        # remove small connected components
        labeled, n = ndi.label(mask)
        if n > 0:
            counts = np.bincount(labeled.ravel())
            # counts[0] is background
            keep = np.zeros_like(counts, dtype=bool)
            keep[counts >= MIN_OBJECT_PIXELS] = True
            keep[0] = False
            mask = keep[labeled]
    else:
        # If no scipy, keep it simple (still works; just noisier)
        mask = mask

    return mask.astype(np.uint8)


# ==========================
# Dataset
# ==========================
class GlacierPatchDataset(Dataset):
    def __init__(self, npz_paths, crop_size=256, ndsi_thresh=0.4, augment=True,
                 dem_path=None, use_slope=True, use_aspect=True):
        self.paths = list(npz_paths)
        self.crop_size = crop_size
        self.ndsi_thresh = ndsi_thresh
        self.augment = augment
        self.use_slope = use_slope
        self.use_aspect = use_aspect
        self.dem_src = rasterio.open(str(dem_path)) if dem_path is not None else None

    def __del__(self):
        try:
            if self.dem_src is not None:
                self.dem_src.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.paths)

    def _random_crop(self, x, y, size):
        H, W = y.shape
        if H < size or W < size:
            pad_h = max(0, size - H)
            pad_w = max(0, size - W)
            x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            y = np.pad(y, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            H, W = y.shape

        r0 = np.random.randint(0, H - size + 1)
        c0 = np.random.randint(0, W - size + 1)
        return x[r0:r0+size, c0:c0+size], y[r0:r0+size, c0:c0+size]

    def _augment(self, x, y):
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=0).copy()
            y = np.flip(y, axis=0).copy()
        if np.random.rand() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()
        k = np.random.randint(0, 4)
        if k:
            x = np.rot90(x, k, axes=(0, 1)).copy()
            y = np.rot90(y, k, axes=(0, 1)).copy()
        return x, y

    def __getitem__(self, idx):
        npz_path = self.paths[idx]
        d = np.load(npz_path)

        rgb = d["rgb"].astype(np.float32)      # (H,W,3) in [0,1]
        ndsi = d["ndsi"].astype(np.float32)    # (H,W)
        qa_good = d["qa_good"].astype(np.uint8)

        H, W = ndsi.shape

        # meta for this patch
        meta_path = npz_path.with_name(npz_path.name.replace("_arrays.npz", "_meta.txt"))
        meta = parse_meta_txt(meta_path)

        extras = []

        if self.dem_src is not None:
            dem_patch = sample_dem_to_patch(self.dem_src, meta, H, W)   # meters
            dem_norm = robust_norm(dem_patch)
            extras.append(dem_norm[..., None])

            slope_deg, aspect_deg = terrain_slope_aspect(dem_patch)

            if self.use_slope:
                slope_norm = robust_norm(slope_deg)
                extras.append(slope_norm[..., None])

            if self.use_aspect:
                a_sin, a_cos = aspect_to_sin_cos(aspect_deg)
                # aspect is directional; normalization not needed
                extras.append(a_sin[..., None])
                extras.append(a_cos[..., None])
        else:
            # fallback zeros if no DEM
            n_extra = 1 + (1 if self.use_slope else 0) + (2 if self.use_aspect else 0)
            extras.append(np.zeros((H, W, n_extra), dtype=np.float32))

        extra_stack = np.concatenate(extras, axis=2) if len(extras) > 1 else extras[0]

        # input: RGB + NDSI + extras
        x = np.dstack([rgb, ndsi[..., None], extra_stack])  # (H,W,C)

        # pseudo-label
        y = make_pseudolabel(ndsi, qa_good, self.ndsi_thresh).astype(np.uint8)

        if self.crop_size is not None:
            x, y = self._random_crop(x, y, self.crop_size)

        if self.augment:
            x, y = self._augment(x, y)

        x = torch.from_numpy(np.transpose(x, (2, 0, 1)))              # (C,H,W)
        y = torch.from_numpy(y[None, ...].astype(np.float32))         # (1,H,W)
        return x, y
    
# ==========================
# Tiny U-Net
# ==========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bot = DoubleConv(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bot(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


# ==========================
# Loss: BCE + Dice
# ==========================
def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = 1 - (num / den)
    return dice.mean()

def bce_dice_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d = dice_loss(logits, targets)
    return bce + d


# ==========================
# Train
# ==========================
def main():
    npz_paths = []
    for y in ["2000", "2020","Wallowas"]:  # change index to select different year(s)
        npz_paths += sorted((SCRIPT_DIR / "patches" / y).glob("patch_*_arrays.npz"))

    npz_paths = sorted(npz_paths)
    print("Total patches:", len(npz_paths))    
    
    # if len(npz_paths) < 5:
    #     print(f"WARNING: Only {len(npz_paths)} patches found. Model will overfit. Add more patches for real results.")

    ds = GlacierPatchDataset(
    npz_paths,
    crop_size=RANDOM_CROP,
    ndsi_thresh=NDSI_THRESHOLD,
    augment=True,
    dem_path=DEM_PATH,
    use_slope=USE_SLOPE,
    use_aspect=USE_ASPECT,
    )
    x0, y0 = ds[0]
    print("Input channels:", x0.shape[0])  # should print 8
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    in_ch = 4 + 1 + (1 if USE_SLOPE else 0) + (2 if USE_ASPECT else 0)
    # 4 = RGB+NDSI, +1 DEM, +1 slope, +2 aspect(sin/cos) => 8
    model = UNetSmall(in_ch=in_ch, out_ch=1, base=32).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for x, y in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={np.mean(losses):.4f}")

    out_path = OUT_DIR / "glacier_unet_pseudolabel.pt"
    torch.save({
        "model_state": model.state_dict(),
        "ndsi_threshold": NDSI_THRESHOLD,
        "crop_size": RANDOM_CROP,
    }, out_path)
    print(f"Saved model to: {out_path}")



if __name__ == "__main__":
    main()
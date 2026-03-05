"""
Inference + measurements from predicted glacier mask.

Uses the meta file fields:
  lat_min, lat_max, lon_min, lon_max
to compute pixel size ~ meters/pixel, then area.

Also estimates length/width using oriented bounding box approximations (PCA).
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds
from config import*

# optional: connected components cleanup
try:
    from scipy import ndimage as ndi
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent

# add 'Wallowas' if you have those patches, otherwise just use the first three years

YEAR = CURRENT_YEAR   # change as needed

MODEL_PATH = SCRIPT_DIR / "models" / "glacier_unet_pseudolabel.pt"
NPZ_PATH   = SCRIPT_DIR / "patches" /YEAR/ "patch_001_arrays.npz"
META_PATH  = SCRIPT_DIR / "patches" /YEAR/ "patch_001_meta.txt"

DEM_PATH = SCRIPT_DIR / "data" /"DEM" / "output_hh.tif"
USE_SLOPE = True
USE_ASPECT = True


THRESH = 0.18  # instead of 0.5 to get more area 
# ---- same UNetSmall architecture as training ----
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
    def forward(self, x): return self.net(x)

import torch
import torch.nn.functional as F

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

    @staticmethod
    def match_size(src, ref):
        """
        Center-crop src to have the same H,W as ref (for skip connections).
        """
        _, _, h, w = src.shape
        _, _, rh, rw = ref.shape
        dh = h - rh
        dw = w - rw
        if dh == 0 and dw == 0:
            return src
        top = dh // 2
        left = dw // 2
        return src[:, :, top:top+rh, left:left+rw]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bot(self.pool3(e3))

        u3 = self.up3(b)
        # make u3 match e3 size
        if u3.shape[-2:] != e3.shape[-2:]:
            # crop whichever is bigger to match the smaller
            if u3.shape[-2] >= e3.shape[-2] and u3.shape[-1] >= e3.shape[-1]:
                u3 = self.match_size(u3, e3)
            else:
                e3 = self.match_size(e3, u3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        if u2.shape[-2:] != e2.shape[-2:]:
            if u2.shape[-2] >= e2.shape[-2] and u2.shape[-1] >= e2.shape[-1]:
                u2 = self.match_size(u2, e2)
            else:
                e2 = self.match_size(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        if u1.shape[-2:] != e1.shape[-2:]:
            if u1.shape[-2] >= e1.shape[-2] and u1.shape[-1] >= e1.shape[-1]:
                u1 = self.match_size(u1, e1)
            else:
                e1 = self.match_size(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)
    
def parse_meta(path: Path) -> dict:
    meta = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = float(v.strip())
    return meta

def meters_per_pixel(meta: dict, H: int, W: int):
    """
    Approx meters per pixel using latitude scaling.
    """
    lat_min, lat_max = meta["lat_min"], meta["lat_max"]
    lon_min, lon_max = meta["lon_min"], meta["lon_max"]
    lat_c = 0.5 * (lat_min + lat_max)

    # meters per degree (rough)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_c))

    height_m = (lat_max - lat_min) * m_per_deg_lat
    width_m  = (lon_max - lon_min) * m_per_deg_lon

    return width_m / W, height_m / H

def clean_mask(mask: np.ndarray, min_pixels: int = 50) -> np.ndarray:
    if not SCIPY_OK:
        return mask

    # light cleanup
    mask = ndi.binary_opening(mask, structure=np.ones((3,3)))
    mask = ndi.binary_closing(mask, structure=np.ones((5,5)))

    labeled, n = ndi.label(mask)
    if n == 0:
        return mask

    counts = np.bincount(labeled.ravel())
    keep = (counts >= min_pixels)
    keep[0] = False  # background
    return keep[labeled]

def pca_length_width(mask: np.ndarray, px_w_m: float, px_h_m: float):
    """
    Estimate major/minor axis lengths (meters) using PCA on foreground pixels.
    """
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return 0.0, 0.0

    # convert pixel coords to meters in local patch coordinates
    X = np.stack([xs * px_w_m, ys * px_h_m], axis=1).astype(np.float64)
    X -= X.mean(axis=0, keepdims=True)

    # PCA via covariance eig
    C = (X.T @ X) / max(1, (X.shape[0] - 1))
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]

    proj = X @ vecs
    major = proj[:, 0].max() - proj[:, 0].min()
    minor = proj[:, 1].max() - proj[:, 1].min()
    return float(major), float(minor)

def sample_dem_to_patch(dem_src, meta: dict, H: int, W: int) -> np.ndarray:
    left, right = meta["lon_min"], meta["lon_max"]
    bottom, top = meta["lat_min"], meta["lat_max"]

    b = transform_bounds("EPSG:4326", dem_src.crs, left, bottom, right, top, densify_pts=21)

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

    if dem_src.nodata is not None:
        dst[dst == dem_src.nodata] = np.nan
    return dst.astype(np.float32)

def robust_norm(x: np.ndarray, p1=2, p2=98) -> np.ndarray:
    lo = np.nanpercentile(x, p1)
    hi = np.nanpercentile(x, p2)
    y = (x - lo) / (hi - lo + 1e-6)
    return np.clip(y, 0, 1).astype(np.float32)

def terrain_slope_aspect(dem_m, pixel_width, pixel_height):
    dem = dem_m.copy()
    nanmask = np.isnan(dem)
    if nanmask.any():
        dem[nanmask] = np.nanmedian(dem)

    dz_dy, dz_dx = np.gradient(dem, pixel_height, pixel_width)
    slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360
    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)

def qa_good_from_qapixel(qa: np.ndarray) -> np.ndarray:
    qa = qa.astype(np.uint16)
    fill    = (qa & (1 << 0)) != 0
    dilated = (qa & (1 << 1)) != 0
    cirrus  = (qa & (1 << 2)) != 0
    cloud   = (qa & (1 << 3)) != 0
    shadow  = (qa & (1 << 4)) != 0
    water   = (qa & (1 << 7)) != 0  # optional

    # IMPORTANT: do NOT remove snow (bit 5). We WANT snow/ice pixels.
    good = ~(fill | dilated | cirrus | cloud | shadow)

    # optional: remove water too (often fine for glaciers)
    good = good & ~water
    return good

def aspect_to_sin_cos(aspect_deg: np.ndarray):
    ang = np.deg2rad(aspect_deg.astype(np.float32))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)

def crop_to_match(a, ref):
    """Center-crop a 2D array a to match ref's (H,W)."""
    H, W = a.shape
    rH, rW = ref.shape
    if (H, W) == (rH, rW):
        return a
    top = max(0, (H - rH) // 2)
    left = max(0, (W - rW) // 2)
    return a[top:top+rH, left:left+rW]

def main():
    # Load model (after retraining with 9 channels)
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    in_ch = 5 + 1 + (1 if USE_SLOPE else 0) + (2 if USE_ASPECT else 0)  # =9
    model = UNetSmall(in_ch=in_ch, out_ch=1, base=32).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    patch_files = sorted(SCRIPT_DIR.glob(f"patches/{YEAR}/patch_*_arrays.npz"))
    if not patch_files:
        print("No patch files found.")
        return

    results = []

    with rasterio.open(DEM_PATH) as dem_src:
        for npz_path in patch_files:
            meta_path = npz_path.with_name(npz_path.name.replace("_arrays.npz", "_meta.txt"))
            if not meta_path.exists():
                print(f"Warning: meta file {meta_path} not found, skipping {npz_path}")
                continue

            print(f"\nProcessing {npz_path.stem} ...")

            d = np.load(npz_path)
            green = d["green"].astype(np.float32)
            red   = d["red"].astype(np.float32)
            nir   = d["nir"].astype(np.float32)
            swir1 = d["swir1"].astype(np.float32)
            ndsi  = d["ndsi"].astype(np.float32)

            meta = parse_meta(meta_path)
            H, W = ndsi.shape

            # DEM and derivatives
            dem_patch = sample_dem_to_patch(dem_src, meta, H, W)
            dem_km = dem_patch / 1000.0
            extras = [dem_km[..., None]]
            px_w_m, px_h_m = meters_per_pixel(meta, H, W)

            slope_deg, aspect_deg = terrain_slope_aspect(dem_patch, px_w_m, px_h_m)

            if USE_SLOPE:
                extras.append(slope_deg[..., None])
            if USE_ASPECT:
                a_sin, a_cos = aspect_to_sin_cos(aspect_deg)
                extras.append(a_sin[..., None])
                extras.append(a_cos[..., None])

            extra_stack = np.concatenate(extras, axis=2)

            # Spectral stack
            spec = np.dstack([green, red, nir, swir1, ndsi])

            x = np.dstack([spec, extra_stack])   # (H, W, 9)
            
            x_t = torch.from_numpy(np.transpose(x, (2,0,1))[None, ...]).to(DEVICE)

            with torch.no_grad():
                logits = model(x_t)
                prob_t = torch.sigmoid(logits)
                prob_t = F.interpolate(prob_t, size=(H, W), mode="bilinear", align_corners=False)
                prob = prob_t[0,0].cpu().numpy()
                ndsi_for_prob = crop_to_match(ndsi, prob)
                mask = (prob >= 0.5) & (ndsi_for_prob > 0.25) & (dem_patch > 2000)
            # QA mask (optional, but keep for now)
            if "qa_pixel" in d:
                qa_good = qa_good_from_qapixel(d["qa_pixel"])
                prob = np.where(qa_good, prob, 0.0)

            # Thresholding – adjust after retraining!
            # mask = prob >= 0.5
            # print("prob percentiles:", np.percentile(prob, [0,1,5,25,50,75,95,99,100]))
            mask = clean_mask(mask, min_pixels=500)   # ~0.05 km² at 30 m

            # Measurements (unchanged)
            px_w_m, px_h_m = meters_per_pixel(meta, H, W)
            area_m2 = mask.sum() * (px_w_m * px_h_m)
            area_km2 = area_m2 / 1e6
            length_m, width_m = pca_length_width(mask, px_w_m, px_h_m)


            # Save outputs
            out_dir = SCRIPT_DIR / "outputs" / YEAR   # change YEAR to the year you're processing
            out_dir.mkdir(exist_ok=True)

            base_name = npz_path.stem.replace("_arrays", "")
            np.save(out_dir / f"{base_name}_prob.npy", prob)
            np.save(out_dir / f"{base_name}_mask.npy", mask.astype(np.uint8))

            results.append({
                "patch": base_name,
                "area_km2": area_km2,
                "length_m": length_m,
                "width_m": width_m,
                "pixel_count": int(mask.sum()),
                "px_width_m": px_w_m,
                "px_height_m": px_h_m,
                "lat_min": meta["lat_min"],
                "lat_max": meta["lat_max"],
                "lon_min": meta["lon_min"],
                "lon_max": meta["lon_max"],
            })

            print(f"  Area: {area_km2:.4f} km², Length: {length_m:.1f} m, Width: {width_m:.1f} m")
    # Save summary CSV
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / "glacier_measurements.csv", index=False)
        print(f"\nSaved summary to {out_dir / 'glacier_measurements.csv'}")
                
if __name__ == "__main__":
    main()
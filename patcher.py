import os
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.warp import transform as crs_transform
from rasterio.crs import CRS
from config import*



# Paths
  
OUT_DIR  = BASE_DIR / "patches"/ CURRENT_YEAR 
OUT_DIR.mkdir(exist_ok=True)


files = {}

def find_files(year, sensor):
    if sensor == "L7" or sensor == "L5":
       for file in os.listdir(DATA_DIR / year):
        if file.endswith(".TIF"):
            if "SR_B1" in file:
                files["B1"] = DATA_DIR /year /file
            elif "SR_B2" in file:
                files["B2"] = DATA_DIR /year /file
            elif "SR_B3" in file:
                files["B3"] = DATA_DIR /year /file
            elif "SR_B4" in file:
                files["B4"] = DATA_DIR /year /file
            elif "SR_B5" in file:
                files["B5"] = DATA_DIR /year /file
            elif "SR_B7" in file:
                files["B7"] = DATA_DIR /year /file
            elif "QA_PIXEL" in file:
                files["QA"] = DATA_DIR /year /file
# Landsat 8 has different band map, so we need to map them to the same keys (B1, B2, etc.) for consistency.
    elif sensor == "L8" or sensor == "L9":
        for file in os.listdir(DATA_DIR / year):
            if file.endswith(".TIF"):
                if "SR_B2" in file:
                    files["B1"] = DATA_DIR /year /file
                elif "SR_B3" in file:
                    files["B2"] = DATA_DIR /year /file
                elif "SR_B4" in file:
                    files["B3"] = DATA_DIR /year /file
                elif "SR_B5" in file:
                    files["B4"] = DATA_DIR /year /file
                elif "SR_B6" in file:
                    files["B5"] = DATA_DIR /year /file
                elif "SR_B7" in file:
                    files["B7"] = DATA_DIR /year /file
                elif "QA_PIXEL" in file:
                    files["QA"] = DATA_DIR /year /file


                
find_files(CURRENT_YEAR, LANDSAT)  # Change index to select different year
        

# Read TIF
def read_tif(path):
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file:\n{path}")
    with rasterio.open(path) as src:
        arr = src.read(1)
        prof = src.profile
    return arr, prof

# Pixel -> LatLon
def pixel_to_latlon(prof, col, row):
    x, y = prof["transform"] * (col, row)
    lon, lat = crs_transform(prof["crs"], "EPSG:4326", [x], [y])
    return lat[0], lon[0]

# Zoom into lat/lon region
def zoom_into_region(arr, prof, lat_min, lat_max, lon_min, lon_max):
    img_crs = prof["crs"]
    src_crs = CRS.from_epsg(4326)

    xs, ys = crs_transform(src_crs, img_crs,
                           [lon_min, lon_max],
                           [lat_max, lat_min])

    x_min, x_max = xs
    y_max, y_min = ys

    transform_obj = prof["transform"]

    c_left,  r_top    = ~transform_obj * (x_min, y_max)
    c_right, r_bottom = ~transform_obj * (x_max, y_min)

    row_min = int(np.floor(min(r_top, r_bottom)))
    row_max = int(np.ceil (max(r_top, r_bottom)))
    col_min = int(np.floor(min(c_left, c_right)))
    col_max = int(np.ceil (max(c_left, c_right)))

    row_min = max(0, row_min)
    row_max = min(arr.shape[0], row_max)
    col_min = max(0, col_min)
    col_max = min(arr.shape[1], col_max)

    if row_min >= row_max or col_min >= col_max:
        return None  # empty crop

    return arr[row_min:row_max, col_min:col_max]

def meters_per_pixel_from_transform(prof):
    # Pixel size in projected units
    # For Landsat Collection 2 L2, CRS is usually UTM meters, so this is meters/pixel.
    a = prof["transform"].a      # x pixel size
    e = prof["transform"].e      # y pixel size (negative)
    return float(abs(a)), float(abs(e))

# NDSI FUNCTION 
def compute_ndsi(green, swir1):
    green = green.astype(np.float32)
    swir1 = swir1.astype(np.float32)
    eps = 1e-6
    return (green - swir1) / (green + swir1 + eps)

# QA_PIXEL decode helpers (Landsat C2 QA_PIXEL)
def bit_is_set(arr_uint, bit):
    """Return boolean mask where a given bit is 1."""
    return ((arr_uint >> bit) & 1).astype(bool)

def make_good_pixel_mask(qa_pixel):
    """
    A conservative 'good pixel' mask:
      - exclude fill, dilated cloud, cloud, cloud shadow
    Notes:
      - You can also exclude snow (bit 5) if seasonal snow is hurting you,
        but for glaciers you often keep it and rely on NDSI/season filtering.
    """
    qa = qa_pixel.astype(np.uint16)

    fill          = bit_is_set(qa, 0)
    dilated_cloud = bit_is_set(qa, 1)
    cloud         = bit_is_set(qa, 3)
    cloud_shadow  = bit_is_set(qa, 4)

    bad = fill | dilated_cloud | cloud | cloud_shadow
    return ~bad

# Load Landsat 7 bands + QA
B1, prof = read_tif(files["B1"])  # Blue
B2, _    = read_tif(files["B2"])  # Green
B3, _    = read_tif(files["B3"])  # Red
B4, _    = read_tif(files["B4"])  # NIR
B5, _    = read_tif(files["B5"])  # SWIR1
B7, _    = read_tif(files["B7"])  # SWIR2
QA, _    = read_tif(files["QA"])  # QA_PIXEL (uint16 bitmask)

scale = 0.0000275
offset = -0.2

BLUE  = B1.astype(np.float32) * scale + offset
GREEN = B2.astype(np.float32) * scale + offset
RED   = B3.astype(np.float32) * scale + offset
NIR   = B4.astype(np.float32) * scale + offset
SWIR1 = B5.astype(np.float32) * scale + offset

BLUE  = np.clip(BLUE,  0, 1)
GREEN = np.clip(GREEN, 0, 1)
RED   = np.clip(RED,   0, 1)
NIR   = np.clip(NIR,   0, 1)
SWIR1 = np.clip(SWIR1, 0, 1)

# RGB for viewing / saving thumbnails (optional)
rgb = np.dstack([RED, GREEN, BLUE]).astype(np.float32)
rgb = np.clip(rgb, 0, 1)

# NDSI from reflectance (recommended)
ndsi = compute_ndsi(GREEN, SWIR1)
# Interactive patch picker loop
print("\nInteractive patch picker:")
print(" - Click a point on the RGB map to define a patch center")
print(" - After preview, type 'y' to save, anything else to skip")
print(" - To exit: close the figure window OR press Enter at the prompt\n")

# Patch size in degrees (tune these)
HALF_LAT = 0.08   # ~9 km
HALF_LON = 0.10   # ~8-10 km depending on latitude

patch_id = 1


while True:
    fig = plt.figure(figsize=(9, 9))
    plt.imshow(rgb)
    plt.title("Click a patch center (close window to exit)")
    plt.axis("off")

    pts = plt.ginput(1, timeout=0)
    plt.close(fig)

    if len(pts) == 0:
        print("Exit: no click detected (window closed).")
        break

    x, y = pts[0]
    lat0, lon0 = pixel_to_latlon(prof, int(x), int(y))

    lat_min, lat_max = lat0 - HALF_LAT, lat0 + HALF_LAT
    lon_min, lon_max = lon0 - HALF_LON, lon0 + HALF_LON

    crop_rgb  = zoom_into_region(rgb,  prof, lat_min, lat_max, lon_min, lon_max)
    crop_ndsi = zoom_into_region(ndsi, prof, lat_min, lat_max, lon_min, lon_max)
    crop_qa   = zoom_into_region(QA,   prof, lat_min, lat_max, lon_min, lon_max)
    crop_red   = zoom_into_region(RED,   prof, lat_min, lat_max, lon_min, lon_max)
    crop_green = zoom_into_region(GREEN, prof, lat_min, lat_max, lon_min, lon_max)
    crop_nir   = zoom_into_region(NIR,   prof, lat_min, lat_max, lon_min, lon_max)
    crop_swir1 = zoom_into_region(SWIR1, prof, lat_min, lat_max, lon_min, lon_max)

    if crop_rgb is None or crop_ndsi is None or crop_qa is None:
        print("WARNING: crop region empty/outside image. Try clicking inside the valid swath.")
        continue

    # Build a 'good pixel' mask from cropped QA
    crop_good = make_good_pixel_mask(crop_qa)

    # Optional: a quick glacier-candidate view (for sanity)
    # (This is NOT a final glacier mask; just helps you see if QA masking is working.)
    glacier_candidate = (crop_ndsi > 0.4) & crop_good

    # Preview crops
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(crop_rgb)
    plt.title(f"RGB patch #{patch_id}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(crop_ndsi, cmap="coolwarm")
    plt.title(f"NDSI patch #{patch_id}")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    plt.imshow(glacier_candidate, cmap="gray")
    plt.title(f"NDSI>0.4 & QA good (preview) #{patch_id}")
    plt.axis("off")

    plt.show()

    ans = input(f"Save patch #{patch_id}? (y to save, Enter to exit, anything else to skip): ").strip().lower()

    if ans == "":
        print("Exit: user pressed Enter.")
        break

    if ans == "y":
        rgb_path  = OUT_DIR / f"patch_{patch_id:03d}_rgb.png"
        ndsi_path = OUT_DIR / f"patch_{patch_id:03d}_ndsi.png"
        meta_path = OUT_DIR / f"patch_{patch_id:03d}_meta.txt"
        npz_path  = OUT_DIR / f"patch_{patch_id:03d}_arrays.npz"

        plt.imsave(rgb_path, crop_rgb)
        plt.imsave(ndsi_path, crop_ndsi, cmap="coolwarm")

        # Save arrays for ML later:
        # - rgb: float32 (H,W,3)
        # - ndsi: float32 (H,W)
        # - qa_pixel: uint16 (H,W) raw QA bits
        # - qa_good: bool (H,W) "good pixels" mask
        np.savez_compressed(
            npz_path,
            rgb=crop_rgb.astype(np.float32),   # if you still want it
            red=crop_red.astype(np.float32),
            green=crop_green.astype(np.float32),
            nir=crop_nir.astype(np.float32),
            swir1=crop_swir1.astype(np.float32),
            ndsi=crop_ndsi.astype(np.float32),
            qa_pixel=crop_qa.astype(np.uint16),
            qa_good=crop_good.astype(np.uint8),
        )

        # meta = (
        #     f"center_lat={lat0}\n"
        #     f"center_lon={lon0}\n"
        #     f"lat_min={lat_min}\nlat_max={lat_max}\n"
        #     f"lon_min={lon_min}\nlon_max={lon_max}\n"
        #     f"half_lat={HALF_LAT}\nhalf_lon={HALF_LON}\n"
        # )
        px_w_m, px_h_m = meters_per_pixel_from_transform(prof)

        meta = (
            f"center_lat={lat0}\n"
            f"center_lon={lon0}\n"
            f"lat_min={lat_min}\nlat_max={lat_max}\n"
            f"lon_min={lon_min}\nlon_max={lon_max}\n"
            f"half_lat={HALF_LAT}\nhalf_lon={HALF_LON}\n"
            f"px_w_m={px_w_m}\n"
            f"px_h_m={px_h_m}\n"
        )
        
        meta_path.write_text(meta)

        print(f"Saved:\n - {rgb_path}\n - {ndsi_path}\n - {npz_path}\n - {meta_path}\n")
        patch_id += 1
    else:
        print("Skipped (not saved).")

print("Done.")
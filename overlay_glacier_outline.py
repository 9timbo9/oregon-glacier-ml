from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, transform

try:
    from tqdm import tqdm
    TQDM_OK = True
except:
    TQDM_OK = False

SCRIPT_DIR = Path(__file__).resolve().parent
YEARS = ['1980','2000','2020','Wallowas']
YEAR = YEARS[2]   # change as needed

MASK_DIR = SCRIPT_DIR / "outputs" / YEAR
OUT_DIR = MASK_DIR / "outline_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rgb(npz_path):
    d = np.load(npz_path)
    rgb = d["rgb"].astype(np.float32)
    return np.clip(rgb, 0, 1)

def get_contours(mask, level=0.5, min_length=50):
    contours = measure.find_contours(mask.astype(float), level)
    return [c for c in contours if len(c) >= min_length]

def save_overlay(rgb, contours, out_path, linewidth=2, color='red'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=linewidth, solid_capstyle='round')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def main():
    mask_files = sorted(MASK_DIR.glob("patch_*_mask.npy"))
    if not mask_files:
        print(f"No mask files in {MASK_DIR}")
        return

    iterator = tqdm(mask_files) if TQDM_OK else mask_files
    for mask_path in iterator:
        patch_id = mask_path.stem.replace("_mask", "")
        npz_path = SCRIPT_DIR / "patches" / YEAR / f"{patch_id}_arrays.npz"
        if not npz_path.exists():
            print(f"Warning: {npz_path} not found, skipping {mask_path}")
            continue

        rgb = load_rgb(npz_path)
        mask = np.load(mask_path).astype(bool)

        # Ensure same size
        if mask.shape != rgb.shape[:2]:
            mask = transform.resize(mask.astype(float), rgb.shape[:2],
                                    order=0, preserve_range=True).astype(bool)

        contours = get_contours(mask, min_length=50)
        out_path = OUT_DIR / f"overlay_{patch_id}.png"
        save_overlay(rgb, contours, out_path, linewidth=2, color='yellow')
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
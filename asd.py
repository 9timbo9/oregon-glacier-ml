import matplotlib.pyplot as plt
import numpy as np

d = np.load("patches/2020/patch_001_arrays.npz")
rgb = d["rgb"]

print(rgb.min(), rgb.mean(), rgb.max())
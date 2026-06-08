"""Validate the brain-map render path with a fake (timesteps x 20484) array,
so we don't need the 12GB model just to confirm the nilearn plotting works."""
import os
import numpy as np

arr = (np.random.RandomState(0).randn(12, 20484).astype("float32") * 0.1)
vec = arr.mean(axis=0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets, plotting

fs = datasets.fetch_surf_fsaverage("fsaverage5")
half = vec.shape[0] // 2
lh, rh = vec[:half], vec[half:]
vmax = float(np.abs(vec).max()) or 1.0
fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(11, 5))
plotting.plot_surf_stat_map(fs.infl_left, lh, hemi="left", bg_map=fs.sulc_left,
                            vmax=vmax, axes=axes[0], colorbar=False, title="left")
plotting.plot_surf_stat_map(fs.infl_right, rh, hemi="right", bg_map=fs.sulc_right,
                            vmax=vmax, axes=axes[1], colorbar=True, title="right")
fig.suptitle("Predicted cortical response (mean over time)")
out = os.path.join(os.path.dirname(__file__), "test_brainmap.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
plt.close(fig)
print("OK", out, os.path.getsize(out), "bytes")

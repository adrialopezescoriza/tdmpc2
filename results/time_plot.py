import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from results import *


set_style()

# LaNE: 1d 16h 50m -> hours
#  1 day = 24 hours; total = 24 + 16 + 50/60 = 40.8333 hours
lane_total_steps = 200_000
lane_total_time_hours = 24 + 16 + 50/60
lane_time_per_step = lane_total_time_hours / lane_total_steps

# Ours: 6h 1m -> hours
#  total = 6 + 1/60 = 6.0167 hours
ours_total_steps = 116_000
ours_total_time_hours = 6 + 1/60
ours_time_per_step = ours_total_time_hours / ours_total_steps

# Modem: 20h 55m -> hours
#  total = 20 + 55/60 = 20.9167 hours
modem_total_steps = 250_000
modem_total_time_hours = 20 + 55/60
modem_time_per_step = modem_total_time_hours / modem_total_steps

# ------------------------------------------------
# 2) Compute time to reach from 0 to 100k steps
# ------------------------------------------------
max_steps = 100_000
steps = np.linspace(0, max_steps, 200)  # create a grid of points from 0 to 100k

# Calculate time in hours for each algorithm
lane_time = lane_time_per_step * steps
ours_time = ours_time_per_step * steps
modem_time = modem_time_per_step * steps

# ------------------------------------------------
# 3) Plot time vs. steps
# ------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(steps / 1e3, lane_time, label="LaNE", color="tab:blue")
plt.plot(steps / 1e3, ours_time, label="Ours", color="tab:orange")
plt.plot(steps / 1e3, modem_time, label="Modem", color="tab:green")

# Labeling
plt.xlabel("Steps (1e3)")
plt.ylabel("Time (hours)")
plt.xlim([0, max_steps / 1e3])
plt.legend()
plt.grid(True)
plt.tight_layout()

save_fig('time_comparison')

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
from src.mission.mission_output import display_path_on_map
from unittest.mock import patch

image = np.zeros((40, 70, 3), dtype=np.uint8)
path = [(5, 20), (20, 20), (30, 20), (45, 20), (60, 20)]
with patch("src.mission.mission_output.plt.show"):
    rendered = display_path_on_map(
        image, [], None, path, [1, 3], home=(5, 5),
        reload_indices=[2])

# Show a range of pixels along row 20
for x in range(0, 70, 2):
    print(f"  pixel (20, {x}) = {tuple(rendered[20, x])}")

# Show along row 18 and row 22 too
for row in [18, 22]:
    for x in [10, 12, 14, 48, 50, 52, 54, 56]:
        print(f"  pixel ({row}, {x}) = {tuple(rendered[row, x])}")

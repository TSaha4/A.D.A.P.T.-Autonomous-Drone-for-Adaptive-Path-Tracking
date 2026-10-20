"""E9 - How plausible is the configured geography? (Plausibility estimate, NOT a georeference.)

The repository maps carry no georeferencing. To judge whether METERS_PER_PIXEL = 2.0 and the default reference
(25.3176 N, 82.9739 E) could describe them, a least-squares similarity transform (scale, rotation, translation)
is fitted between map LABEL positions and APPROXIMATE town coordinates.
Caveats (stated in the paper): label pixel positions were read off the images by eye (about +/-15 px);
town coordinates are approximate public values (about +/-0.02 deg); labels sit beside, not on, the towns.
The result is an order-of-magnitude plausibility check only.
Output: results/geography/map_scale_estimate.json
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from src.mission.coordinates import pixel_to_latlon  # noqa: E402
from tests import geodesy_reference as ref  # noqa: E402

# (label, x_px, y_px in the ORIGINAL image, approx. latitude, approx. longitude)
LANDMARKS = {
    "varanasi": [("Varanasi", 355, 280, 25.3176, 82.9739), ("Babatpur", 215, 128, 25.4500, 82.8590),
                 ("Chunar", 250, 535, 25.1265, 82.8813), ("Chandauli", 675, 352, 25.2600, 83.2700),
                 ("Chakia", 600, 600, 25.0461, 83.2244)],
    "kanpur": [("Kanpur", 712, 465, 26.4499, 80.3319), ("Unnao", 832, 395, 26.5393, 80.4878),
               ("Bithoor", 662, 340, 26.6114, 80.2730), ("Bilhaur", 500, 133, 26.8433, 80.0698),
               ("Ghatampur", 572, 738, 26.1527, 80.1680), ("Hamirpur", 572, 868, 25.9563, 80.1497),
               ("Kalpi", 255, 763, 26.1167, 79.7333), ("Auraiya", 72, 466, 26.4667, 79.5167),
               ("Akbarpur", 412, 537, 26.4290, 79.9600), ("Pukhrayan", 330, 670, 26.2300, 79.8400)],
    "assam": [("Kokrajhar", 120, 355, 26.40, 90.27), ("Goalpara", 130, 405, 26.17, 90.62),
              ("Barpeta", 225, 370, 26.32, 91.00), ("Nalbari", 305, 340, 26.44, 91.44),
              ("Nagaon", 500, 333, 26.35, 92.68), ("Golaghat", 632, 270, 26.52, 93.96),
              ("Jorhat", 740, 255, 26.75, 94.20), ("North Lakhimpur", 710, 180, 27.24, 94.10),
              ("Dibrugarh", 820, 135, 27.47, 94.91), ("Tinsukia", 895, 93, 27.49, 95.36),
              ("Silchar", 540, 577, 24.83, 92.78), ("Hojai", 535, 390, 26.00, 92.86)],
}
DEFAULT_REF, MPP = (25.3176, 82.9739), 2.0


def main():
    out = {}
    for name, lms in LANDMARKS.items():
        H0, W0 = cv2.imread(os.path.join(common.REPO, "data", "input", f"{name}.png")).shape[:2]
        lat_r, lon_r = np.mean([l[3] for l in lms]), np.mean([l[4] for l in lms])
        k_n = ref.meridional_radius(lat_r) * math.pi / 180
        k_e = ref.prime_vertical_radius(lat_r) * math.cos(math.radians(lat_r)) * math.pi / 180
        zp = np.array([complex(l[1], -l[2]) for l in lms])
        zg = np.array([complex((l[4] - lon_r) * k_e, (l[3] - lat_r) * k_n) for l in lms])
        A = np.vstack([zp, np.ones_like(zp)]).T
        (a, b), *_ = np.linalg.lstsq(A, zg, rcond=None)
        resid = np.abs(A @ np.array([a, b]) - zg)
        scale = abs(a)
        zc = a * complex((W0 - 1) / 2, -(H0 - 1) / 2) + b
        c_lat, c_lon = lat_r + zc.imag / k_n, lon_r + zc.real / k_e
        tl = pixel_to_latlon(0, 0, ((W0 - 1) / 2, (H0 - 1) / 2), DEFAULT_REF, MPP)
        br = pixel_to_latlon(W0 - 1, H0 - 1, ((W0 - 1) / 2, (H0 - 1) / 2), DEFAULT_REF, MPP)
        out[name] = {
            "image_px": [W0, H0], "landmarks": len(lms),
            "configured": {"meters_per_pixel": MPP, "reference": DEFAULT_REF,
                           "coverage_km": [W0 * MPP / 1000, H0 * MPP / 1000],
                           "bbox": {"north": tl[0], "south": br[0], "west": tl[1], "east": br[1]}},
            "estimated": {"meters_per_pixel": scale, "rotation_deg": math.degrees(math.atan2(a.imag, a.real)),
                          "residual_rms_km": float(math.sqrt(np.mean(resid ** 2)) / 1000),
                          "residual_rms_px": float(math.sqrt(np.mean(resid ** 2)) / scale),
                          "coverage_km": [W0 * scale / 1000, H0 * scale / 1000], "image_centre": [c_lat, c_lon]},
            "scale_ratio_estimated_over_configured": scale / MPP,
            "default_reference_to_estimated_centre_km": ref.vincenty_inverse(*DEFAULT_REF, c_lat, c_lon)[0] / 1000,
        }
        print(name, {k: v for k, v in out[name].items() if k != "configured"})
    common.save_json(out, "geography", "map_scale_estimate.json")


if __name__ == "__main__":
    main()

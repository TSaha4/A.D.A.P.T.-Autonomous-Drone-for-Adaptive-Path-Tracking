import cv2
import numpy as np
from matplotlib import pyplot as plt

def generate_mission_file(safe_points, home, filename="mission.waypoints", base_altitude=100, drop_altitude=10):
    # Mission:
    # Takeoff at home → visit drop points → drop packages → return home → land.
    with open(filename, "w") as f:
        f.write("QGC WPL 110\n")
        seq = 0

        # 0: Home location
        f.write(f"{seq}\t1\t3\t16\t0\t0\t0\t0\t{home[1]}\t{home[0]}\t0\t1\n")
        seq += 1

        # 1: Takeoff at home
        f.write(f"{seq}\t0\t3\t22\t0\t0\t0\t0\t{home[1]}\t{home[0]}\t{base_altitude}\t1\n")
        seq += 1

        for pt in safe_points:
            lat = 12.0 + pt[1] / 10000.0
            lon = 77.0 + pt[0] / 10000.0

            # Fly to drop location at cruise altitude
            f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
            seq += 1
            # Loiter 5 sec
            f.write(f"{seq}\t0\t3\t19\t5\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
            seq += 1
            # Descend to drop altitude
            f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{drop_altitude}\t1\n")
            seq += 1
            # Drop package (servo trigger)
            f.write(f"{seq}\t0\t3\t183\t0\t0\t2000\t0\t{lat}\t{lon}\t{drop_altitude}\t1\n")
            seq += 1
            # Ascend back to cruise altitude
            f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
            seq += 1

        # Return to home
        f.write(f"{seq}\t0\t3\t20\t0\t0\t0\t0\t{home[1]}\t{home[0]}\t{base_altitude}\t1\n")
        seq += 1
        # Land at home
        f.write(f"{seq}\t0\t3\t21\t0\t0\t0\t0\t{home[1]}\t{home[0]}\t0\t1\n")

def display_path_on_map(image, contours, safe_points, path_order, home=None):
    import cv2
    from matplotlib import pyplot as plt

    vis = image.copy()

    # Draw contours
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)

    # Draw safe points (small blue dots)
    for pt in safe_points:
        pt_int = tuple(map(int, pt))
        cv2.circle(vis, pt_int, 5, (255, 0, 0), -1)

    # Draw HOME (green)
    if home:
        cv2.circle(vis, tuple(map(int, home)), 10, (0, 0, 0), -1)

    # Draw path lines (thin red)
    for i in range(len(path_order) - 1):
        pt1 = tuple(map(int, safe_points[path_order[i]]))
        pt2 = tuple(map(int, safe_points[path_order[i + 1]]))
        cv2.line(vis, pt1, pt2, (0, 0, 255), 2) 

    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Drone Path with Safe Drop Zones & Home")
    plt.axis("off")
    plt.show()


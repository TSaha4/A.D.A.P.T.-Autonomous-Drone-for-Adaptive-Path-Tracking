import cv2
import numpy as np
from matplotlib import pyplot as plt
from src.mission.coordinates import pixel_to_latlon

def generate_mission_file(full_path, drop_indices, home, image_center_px, geo_center, meters_per_pixel, filename="mission.waypoints", base_altitude=100, drop_altitude=10, servo_channel=9, servo_pwm=2000):
    # Mission:
    # Takeoff at home → visit waypoints → drop packages at drop indices → return home → land.
    # Convert every point before opening the file, so an invalid coordinate cannot leave a partial mission
    home_lat, home_lon = pixel_to_latlon(home[0], home[1], image_center_px, geo_center, meters_per_pixel)
    path_latlon = [pixel_to_latlon(pt[0], pt[1], image_center_px, geo_center, meters_per_pixel) for pt in full_path]

    with open(filename, "w") as f:
        f.write("QGC WPL 110\n")
        seq = 0

        # 0: Home location
        f.write(f"{seq}\t1\t3\t16\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t0\t1\n")
        seq += 1

        # 1: Takeoff at home
        f.write(f"{seq}\t0\t3\t22\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t{base_altitude}\t1\n")
        seq += 1

        for i, pt in enumerate(full_path):
            lat, lon = path_latlon[i]

            # If it's the home point (start or end of mission), skip unless it's a drop (usually not)
            if i == 0 and pt == home:
                continue

            # Fly to location at cruise altitude
            f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
            seq += 1
            
            # First/last entries are the HOME departure/return; any other occurrence of HOME is
            # the drop for the cluster whose safe point was chosen as HOME.
            if i in drop_indices and 0 < i < len(full_path) - 1:
                # Loiter 5 sec
                f.write(f"{seq}\t0\t3\t19\t5\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
                seq += 1
                # Descend to drop altitude
                f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{drop_altitude}\t1\n")
                seq += 1
                # Drop package (MAV_CMD_DO_SET_SERVO: param1 = servo channel, param2 = PWM)
                f.write(f"{seq}\t0\t3\t183\t{servo_channel}\t{servo_pwm}\t0\t0\t{lat}\t{lon}\t{drop_altitude}\t1\n")
                seq += 1
                # Ascend back to cruise altitude
                f.write(f"{seq}\t0\t3\t16\t0\t0\t0\t0\t{lat}\t{lon}\t{base_altitude}\t1\n")
                seq += 1

        # Return to home
        f.write(f"{seq}\t0\t3\t20\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t{base_altitude}\t1\n")
        seq += 1
        # Land at home
        f.write(f"{seq}\t0\t3\t21\t0\t0\t0\t0\t{home_lat}\t{home_lon}\t0\t1\n")

def display_path_on_map(image, contours, pred_mask, full_path, drop_indices, home=None, tsp_path=None):
    vis = image.copy()

    # Draw current contours
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    
    # Draw predicted spread
    if pred_mask is not None:
        pred_contours, _ = cv2.findContours(pred_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, pred_contours, -1, (0, 165, 255), 2) # Orange for predicted spread

    # Draw safe points (small blue dots)
    for idx in drop_indices:
        if idx < len(full_path):
            pt_int = tuple(map(int, full_path[idx]))
            cv2.circle(vis, pt_int, 5, (255, 0, 0), -1)

    # Draw HOME (green)
    if home:
        cv2.circle(vis, tuple(map(int, home)), 10, (0, 0, 0), -1)

    # Draw D* Lite path lines (blue) via cv2
    for i in range(len(full_path) - 1):
        pt1 = tuple(map(int, full_path[i]))
        pt2 = tuple(map(int, full_path[i + 1]))
        cv2.line(vis, pt1, pt2, (255, 0, 0), 2) 

    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    
    # Draw TSP direct path lines (red dashed) using matplotlib so it's clearly distinguishable
    if tsp_path:
        tsp_x = [pt[0] for pt in tsp_path]
        tsp_y = [pt[1] for pt in tsp_path]
        plt.plot(tsp_x, tsp_y, 'r--', linewidth=2, label="TSP Direct Path")
        plt.plot([], [], 'b-', linewidth=2, label="D* Lite Avoidance Path")
        plt.legend(loc="upper right")
        
    plt.title("Drone Path with Safe Drop Zones, Obstacles & Home")
    plt.axis("off")
    plt.show()


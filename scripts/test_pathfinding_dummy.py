import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Add the project root to sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routing.pathfinding import nearest_neighbor_tsp, compute_full_path

def main():
    print("Setting up dummy data...")
    
    # 1. Create a dummy obstacle mask (100x100 grid)
    # 0 = free space, 255 = obstacle
    h, w = 100, 100
    obstacle_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Add a large obstacle wall in the middle
    cv2.rectangle(obstacle_mask, (40, 20), (60, 80), 255, -1)
    # Add another small obstacle block
    cv2.rectangle(obstacle_mask, (70, 70), (90, 90), 255, -1)

    # 2. Define dummy points
    home = (10, 50)
    safe_points = [
        (30, 20),  # Point A (Before wall)
        (80, 50),  # Point B (Behind wall)
        (20, 80),  # Point C (Bottom left)
    ]
    
    print("Running Nearest-Neighbor TSP...")
    # 3. Get TSP ordering (straight lines, ignoring obstacles)
    order, ordered_points = nearest_neighbor_tsp(safe_points, home=home)
    print(f"TSP Order: {order}")
    print(f"Ordered Points: {ordered_points}")

    print("Running D* Lite obstacle avoidance...")
    # 4. Compute full path with D* Lite (avoiding obstacles)
    # Note: downsample_factor=1 for our small 100x100 dummy grid
    full_path, drop_indices = compute_full_path(ordered_points, obstacle_mask, downsample_factor=1)
    
    # 5. Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot obstacles as background (inverted so obstacles are dark)
    plt.imshow(255 - obstacle_mask, cmap='gray', origin='upper')
    
    # Plot TSP Path (Direct lines, red dashed)
    tsp_x = [p[0] for p in ordered_points]
    tsp_y = [p[1] for p in ordered_points]
    plt.plot(tsp_x, tsp_y, 'r--', label='TSP Direct Path', alpha=0.5, linewidth=2)
    
    # Plot D* Lite Path (Blue solid line)
    dstar_x = [p[0] for p in full_path]
    dstar_y = [p[1] for p in full_path]
    plt.plot(dstar_x, dstar_y, 'b-', label='D* Lite Actual Path', linewidth=2)
    
    # Plot Drops and Home
    plt.plot(home[0], home[1], 'g*', markersize=15, label='Home')
    drop_x = [p[0] for p in safe_points]
    drop_y = [p[1] for p in safe_points]
    plt.plot(drop_x, drop_y, 'co', markersize=8, label='Drop Points')

    plt.title("Dummy Pathfinding Test (TSP vs D* Lite)")
    plt.legend()
    plt.xlim(0, w)
    plt.ylim(h, 0) # Invert y-axis to match image coords
    
    # Save the plot and show it
    plt.savefig("dummy_path_test.png")
    print("Saved plot to 'dummy_path_test.png'. Showing plot...")
    plt.show()

if __name__ == '__main__':
    main()

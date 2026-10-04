import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.sample_points = []       
        self.hsv_lower = None
        self.hsv_upper = None
        self.is_red_wrap = False
        self.image = None

    def select_sample_points(self, image):
        #Let user select multiple flood points by clicking
        self.image = image.copy()
        cv2.namedWindow("Select flood points (click) and press 'c'")
        cv2.setMouseCallback("Select flood points (click) and press 'c'", self.mouse_click)
        
        print("Click on flood areas (red). Press 'c' when done.") #press 'c' to finish.
        while True:
            cv2.imshow("Select flood points (click) and press 'c'", self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                break

        cv2.destroyAllWindows()

        if not self.sample_points:
            raise ValueError("No points selected for HSV sampling.")

        self.compute_dynamic_hsv()
        return self.hsv_lower, self.hsv_upper

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sample_points.append((x, y))
            cv2.circle(self.image, (x, y), 5, (0, 255, 0), 2)
            print(f"Sample point selected at: ({x}, {y})")

    def auto_sample_points(self, image, count=3, min_saturation=150, min_value=100,
                           hue_tolerance=6, window=11, target_density=0.5, min_separation=40):
        """Pick flood sample points the way an operator would click them.

        The flood overlay is the dominant hue among strongly saturated pixels
        (circular, so red at 0/179 counts as one hue). Points sit on overlay
        pixels whose ``window`` x ``window`` neighbourhood (the patch
        ``compute_dynamic_hsv`` samples) is about ``target_density`` overlay,
        i.e. a click on a flood dot inside a flood cluster, at least
        ``min_separation`` px apart.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.int32)
        strong = (hsv[:, :, 1] >= min_saturation) & (hsv[:, :, 2] >= min_value)
        if not strong.any():
            raise ValueError("No strongly saturated overlay pixels found for automatic flood sampling.")
        hist = np.bincount(hue[strong].ravel(), minlength=180).astype(float)
        smooth = sum(np.roll(hist, k) for k in range(-hue_tolerance, hue_tolerance + 1))
        mode = int(np.argmax(smooth))
        distance = np.abs(hue - mode)
        distance = np.minimum(distance, 180 - distance)
        overlay = strong & (distance <= hue_tolerance)
        density = cv2.boxFilter(overlay.astype(np.float32), -1, (window, window))
        score = np.where(overlay, -np.abs(density - target_density), -np.inf).astype(np.float32)
        points = []
        for _ in range(count):
            y, x = np.unravel_index(int(np.argmax(score)), score.shape)
            if not np.isfinite(score[y, x]):
                break
            points.append((int(x), int(y)))
            cv2.circle(score, (int(x), int(y)), min_separation, -np.inf, -1)
        print(f"Auto flood sampling: dominant saturated hue={mode}, sample points={points}")
        return points

    def compute_dynamic_hsv(self, sample_radius=5, hue_margin=20, sv_margin=30):
        hsv_samples = []

        for x, y in self.sample_points:
            region = self.image[max(y - sample_radius, 0): y + sample_radius + 1,
                                max(x - sample_radius, 0): x + sample_radius + 1]
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hsv_samples.append(hsv_region.reshape(-1, 3))

        hsv_samples = np.vstack(hsv_samples)
        h_vals, s_vals, v_vals = hsv_samples[:,0], hsv_samples[:,1], hsv_samples[:,2]

        h_mean, s_mean, v_mean = np.mean(h_vals), np.mean(s_vals), np.mean(v_vals)
        h_min, h_max = np.min(h_vals), np.max(h_vals)
        s_min, s_max = np.min(s_vals), np.max(s_vals)
        v_min, v_max = np.min(v_vals), np.max(v_vals)

        # Expand margins
        h_min, s_min, v_min = int(h_min), int(s_min), int(v_min)
        h_max, s_max, v_max = int(h_max), int(s_max), int(v_max)

        lower_h = max(h_min - hue_margin, 0)
        upper_h = min(h_max + hue_margin, 179)  # H channel caps at 179 in OpenCV, not 255
        lower_s = max(s_min - sv_margin, 0)
        upper_s = min(s_max + sv_margin, 255)
        lower_v = max(v_min - sv_margin, 0)
        upper_v = min(v_max + sv_margin, 255)

        self.hsv_lower = np.array([int(lower_h), int(lower_s), int(lower_v)], dtype=np.uint8)
        self.hsv_upper = np.array([int(upper_h), int(upper_s), int(upper_v)], dtype=np.uint8)

        # Check for red wrap-around
        self.is_red_wrap = lower_h <= 10 or upper_h >= 170

        print(f"Dynamic HSV Threshold set:\n Lower = {self.hsv_lower}\n Upper = {self.hsv_upper}")
        print(f"Red wrap-around: {self.is_red_wrap}")

    def mask_flood_areas(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if self.is_red_wrap:
            lower_mask = cv2.inRange(hsv,
                                     np.array([0, self.hsv_lower[1], self.hsv_lower[2]]),
                                     np.array([10, self.hsv_upper[1], self.hsv_upper[2]]))
            upper_mask = cv2.inRange(hsv,
                                     np.array([170, self.hsv_lower[1], self.hsv_lower[2]]),
                                     np.array([179, self.hsv_upper[1], self.hsv_upper[2]]))
            mask = cv2.bitwise_or(lower_mask, upper_mask)
        else:
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological clean-up
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        print(f"Mask created. Non-zero pixels: {np.count_nonzero(mask)}")
        return mask

    def find_filtered_contours(self, mask, min_area=50):
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
        print(f"Filtered {len(filtered)} contours with area >= {min_area}")
        return filtered

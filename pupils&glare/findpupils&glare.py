import cv2
import numpy as np
import os

DISPLAY_SCALE = 0.4
BASE_DIR = r"c:\Users\Anastasia\Desktop\Computer Technologies Projects\pupils&glare"

def show(title, img):
    if img is None: return
    h, w = img.shape[:2]
    preview = cv2.resize(img, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
    cv2.imshow(title, preview)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def calculate_eccentricity_from_contour(contour):
        ellipse = cv2.fitEllipse(contour)
        (center, axes, angle) = ellipse
        major = max(axes) / 2
        minor = min(axes) / 2
        eccentricity = np.sqrt(1 - (minor**2 / major**2))
        return eccentricity

def find_circles(mask, min_area=10, top_n=2):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            results.append(((int(x), int(y)), int(radius), area, cnt))
    
    results = sorted(results, key=lambda x: x[2], reverse=True)[:top_n]
    return results

def process_triplet(paths, debug=True):
    imgs = []
    for p in paths:
        im = cv2.imread(p, 0) 
        if im is None:
            print(f"Error: Could not load {p}")
            return None
        imgs.append(im)

    if len(imgs) < 3: return None
    img_w, img_g1, img_g2 = imgs

    pupil_diff = cv2.subtract(img_w, img_g1)
    pupil_diff = cv2.subtract(pupil_diff, img_g2)
    glint_diff = cv2.absdiff(img_g1, img_g2)

    _, max_p, _, _ = cv2.minMaxLoc(pupil_diff)
    ret_p, pupil_thresh = cv2.threshold(pupil_diff, max_p - 80, 255, cv2.THRESH_BINARY)
    
    _, max_g, _, _ = cv2.minMaxLoc(glint_diff)
    ret_g, glint_thresh = cv2.threshold(glint_diff, max_g - 25, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    processed_pupils = np.zeros_like(pupil_thresh)
    contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        single_cnt_mask = np.zeros_like(pupil_thresh)
        cv2.drawContours(single_cnt_mask, [cnt], -1, 255, -1)
        
        if area > 25:
            single_cnt_mask = cv2.erode(single_cnt_mask, kernel, iterations=1)
        
        processed_pupils = cv2.bitwise_or(processed_pupils, single_cnt_mask)

    pupil_dilated = cv2.dilate(processed_pupils, kernel, iterations=2)
    glint_dilated = cv2.dilate(glint_thresh, kernel, iterations=1)

    pupils = find_circles(pupil_dilated, min_area=50, top_n=2)
    glints = find_circles(glint_dilated, min_area=5, top_n=2)

    return pupils, glints, img_w

def draw_final_results(base_img, pupils, glints, idx):
    output = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    
    for i, (center, radius, area, contour) in enumerate(pupils):

        cv2.circle(output, center, radius, (0, 255, 0), 2)
        cv2.drawMarker(output, center, (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
        
        ecc = calculate_eccentricity_from_contour(contour)
        text = f"e={ecc:.3f}"
        cv2.putText(output, text, (center[0] - 40, center[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"Pupil {i+1}: center={center}, radius={radius}, eccentricity={ecc:.3f}")
    
    for center, radius, area, contour in glints:
        cv2.circle(output, center, radius + 1, (0, 0, 255), 2)
    
    show(f"Final Result - Triplet {idx}", output)

if __name__ == "__main__":
    triplets = []
    for i in range(1, 5):
        p = [os.path.join(BASE_DIR, f"eye{i}_{n}.png") for n in ["white", "g1", "g2"]]
        if all(os.path.exists(path) for path in p):
            triplets.append(p)

    if not triplets:
        print(f"Error: No images found in folder {BASE_DIR}.")
    else:
        for idx, paths in enumerate(triplets):
            print(f"\nProcessing Triplet {idx + 1}")
            res = process_triplet(paths, debug=True)
            if res:
                pupils, glints, base = res
                print(f"Found: {len(pupils)} pupils, {len(glints)} glints")
                draw_final_results(base, pupils, glints, idx + 1)

    cv2.destroyAllWindows()
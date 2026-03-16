import cv2
import numpy as np
import os

# --- CONFIGURATION ---
DISPLAY_SCALE = 0.4
BASE_DIR = r"c:\Users\Anastasia\Desktop\Computer Technologies Projects\pupils&glare"

def show(title, img):
    """Displays the image at a scale for convenient viewing."""
    if img is None: return
    h, w = img.shape[:2]
    preview = cv2.resize(img, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
    cv2.imshow(title, preview)
    print(f"Displaying: {title}. Press any key...")
    cv2.waitKey(0)
    cv2.destroyWindow(title)

def find_circles(mask, min_area=10, top_n=2):
    """Finds contours on the mask, performs approximation and returns centers and radii."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            results.append(((int(x), int(y)), int(radius), area))
    
    results = sorted(results, key=lambda x: x[2], reverse=True)[:top_n]
    return [(res[0], res[1]) for res in results]

def process_triplet(paths, debug=True):
    """Full processing pipeline for one triplet with conditional erosion."""
    imgs = []
    for p in paths:
        im = cv2.imread(p, 0) 
        if im is None:
            print(f"Error: Could not load {p}")
            return None
        imgs.append(im)

    if len(imgs) < 3: return None
    img_w, img_g1, img_g2 = imgs

    # === STAGE 1: DIFFERENCE ===
    pupil_diff = cv2.subtract(img_w, img_g1)
    pupil_diff = cv2.subtract(pupil_diff, img_g2)
    glint_diff = cv2.absdiff(img_g1, img_g2)

    # === STAGE 2: BINARIZATION ===
    _, max_p, _, _ = cv2.minMaxLoc(pupil_diff)
    ret_p, pupil_thresh = cv2.threshold(pupil_diff, max_p - 80, 255, cv2.THRESH_BINARY)
    
    _, max_g, _, _ = cv2.minMaxLoc(glint_diff)
    ret_g, glint_thresh = cv2.threshold(glint_diff, max_g - 25, 255, cv2.THRESH_BINARY)

    # === STAGE 3: CONDITIONAL MORPHOLOGY ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Создаем пустую маску для очищенных зрачков
    processed_pupils = np.zeros_like(pupil_thresh)
    contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Создаем временную маску для текущего контура
        single_cnt_mask = np.zeros_like(pupil_thresh)
        cv2.drawContours(single_cnt_mask, [cnt], -1, 255, -1)
        
        # Если площадь boundingRect больше 25 (ядро 5х5), применяем эрозию
        if area > 25:
            single_cnt_mask = cv2.erode(single_cnt_mask, kernel, iterations=1)
        
        # Добавляем результат на общую маску
        processed_pupils = cv2.bitwise_or(processed_pupils, single_cnt_mask)

    # Финальная дилатация для объединения
    pupil_dilated = cv2.dilate(processed_pupils, kernel, iterations=2)
    glint_dilated = cv2.dilate(glint_thresh, kernel, iterations=1)

    if debug:
        show("3. Processed Pupil Mask", pupil_dilated)
        show("3. Glint Mask", glint_dilated)

    # === STAGE 4: CIRCLE APPROXIMATION ===
    pupils = find_circles(pupil_dilated, min_area=50, top_n=2)
    glints = find_circles(glint_dilated, min_area=5, top_n=2)

    return pupils, glints, img_w

def draw_final_results(base_img, pupils, glints, idx):
    """Draws final circles on the source image."""
    output = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    for center, radius in pupils:
        cv2.circle(output, center, radius, (0, 255, 0), 2)
        cv2.drawMarker(output, center, (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
    for center, radius in glints:
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
            print(f"\n--- Processing Triplet #{idx + 1} ---")
            res = process_triplet(paths, debug=True)
            if res:
                pupils, glints, base = res
                print(f"Found: {len(pupils)} pupils, {len(glints)} glints")
                draw_final_results(base, pupils, glints, idx + 1)

    cv2.destroyAllWindows()
    print("\nProcessing complete.")
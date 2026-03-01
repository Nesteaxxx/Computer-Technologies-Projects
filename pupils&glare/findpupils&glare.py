import cv2
import numpy as np
import math

# Конфигурация
TRIPLETS = [
    ["pupils&glare/eye1_white.png", "pupils&glare/eye1_g1.png", "pupils&glare/eye1_g2.png"],
    ["pupils&glare/eye2_white.png", "pupils&glare/eye2_g1.png", "pupils&glare/eye2_g2.png"],
    ["pupils&glare/eye3_white.png", "pupils&glare/eye3_g1.png", "pupils&glare/eye3_g2.png"],
    ["pupils&glare/eye4_white.png", "pupils&glare/eye4_g1.png", "pupils&glare/eye4_g2.png"]
]
DISPLAY_SCALE = 0.5
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Утилиты
def get_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return 0
    return (4 * math.pi * area) / (perimeter * perimeter)

def sight_check(image, title):
    h, w = image.shape[:2]
    preview = cv2.resize(image, (int(w * DISPLAY_SCALE), int(h * DISPLAY_SCALE)))
    cv2.imshow(title, preview)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

# Обработка одного глаза
def process_single_eye(roi_white, roi_g1, offset_x, offset_y):
    pupil_data = None
    glint_contour_global = None
    # 1. Поиск Зрачка
    blurred_p = cv2.GaussianBlur(roi_white, (5, 5), 0)
    _, max_val_p, _, _ = cv2.minMaxLoc(blurred_p)
    thresh_val_p = max(100, max_val_p - 40)
    _, thresh_p = cv2.threshold(blurred_p, thresh_val_p, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_p = cv2.morphologyEx(thresh_p, cv2.MORPH_OPEN, kernel)
    contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_pupil_area = 0
    best_pupil_cnt = None
    for cnt in contours_p:
        area = cv2.contourArea(cnt)
        if 10 < area < 2000:
            circ = get_circularity(cnt)
            if circ > 0.4 and area > best_pupil_area:
                best_pupil_area = area
                best_pupil_cnt = cnt
    if best_pupil_cnt is not None:
        (x, y), radius = cv2.minEnclosingCircle(best_pupil_cnt)
        px = int(x) + offset_x
        py = int(y) + offset_y
        pupil_data = ((px, py), int(radius))
    # 2. Поиск Блика
    _, max_val_g, _, _ = cv2.minMaxLoc(roi_g1)
    thresh_val_g = max(150, max_val_g - 20)
    _, thresh_g = cv2.threshold(roi_g1, thresh_val_g, 255, cv2.THRESH_BINARY)
    contours_g, _ = cv2.findContours(thresh_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_dist_to_pupil = float('inf')
    ref_x = pupil_data[0][0] - offset_x if pupil_data else roi_g1.shape[1] // 2
    ref_y = pupil_data[0][1] - offset_y if pupil_data else roi_g1.shape[0] // 2
    for cnt in contours_g:
        area = cv2.contourArea(cnt)
        if area < 150:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                gx = int(M["m10"] / M["m00"])
                gy = int(M["m01"] / M["m00"])
                dist = math.sqrt((gx - ref_x)**2 + (gy - ref_y)**2)
                if dist < min_dist_to_pupil:
                    min_dist_to_pupil = dist
                    glint_contour_global = cnt + [offset_x, offset_y]
    return pupil_data, glint_contour_global

# Основной пайплайн
def find_eye_features(paths):
    img_white = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
    img_g1 = cv2.imread(paths[1], cv2.IMREAD_GRAYSCALE)
    if img_white is None or img_g1 is None:
        raise FileNotFoundError("Check image paths")
    final_pupils = []
    final_glints = []
    # 1. Поиск глаз
    eyes = eye_cascade.detectMultiScale(img_white, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
    # 2. Фильтр 2-х глаз
    if len(eyes) > 2:
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    # 3. Обработка
    for (x, y, w, h) in eyes:
        roi_w = img_white[y:y+h, x:x+w]
        roi_1 = img_g1[y:y+h, x:x+w]
        pupil, glint_cnt = process_single_eye(roi_w, roi_1, offset_x=x, offset_y=y)
        if pupil: final_pupils.append(pupil)
        if glint_cnt is not None: final_glints.append(glint_cnt)
    return final_pupils, final_glints, img_g1

# Рендеринг
def render_results(base_img, pupils, glint_contours, idx):
    output = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    # Зрачки
    for (center, radius) in pupils:
        cv2.circle(output, center, radius + 2, (0, 255, 0), 2)
    # Блики
    for cnt in glint_contours:
        cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
    sight_check(output, f"FINAL - Triplet {idx}")

# Main
if __name__ == "__main__":
    for i, triplet in enumerate(TRIPLETS):
        print(f"--- Processing Triplet {i+1} ---")
        pupils, glints, base = find_eye_features(triplet)
        render_results(base, pupils, glints, i+1)
    cv2.destroyAllWindows()
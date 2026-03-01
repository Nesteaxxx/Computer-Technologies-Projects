import cv2
import numpy as np

img = cv2.imread("apples\\redapple.png")
cv2.imshow('Original',img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 50, 30])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([165, 50, 30])
upper_red2 = np.array([180, 255, 255])
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

lower_dark = np.array([0, 0, 0])
upper_dark = np.array([180, 255, 50])
mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

mask = cv2.bitwise_and(mask_red, cv2.bitwise_not(mask_green))
mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_dark))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_dil = cv2.dilate(mask, kernel, iterations=1)
contours, _ = cv2.findContours(mask_dil.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
	main = max(contours, key=cv2.contourArea)
	mask = np.zeros_like(mask)
	cv2.drawContours(mask, [main], -1, 255, -1)

def recolor_red_to_green_hsv(img_bgr, mask, green_ref_mask=None):

	hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
	h = hsv_img[:, :, 0]
	s = hsv_img[:, :, 1]
	v = hsv_img[:, :, 2]

	if green_ref_mask is None:
		green_ref = (mask_green == 255) & (mask == 0)
	else:
		green_ref = (green_ref_mask == 255) & (mask == 0)

	if np.count_nonzero(green_ref) == 0:
		green_ref = (mask == 0)

	g_h_vals = h[green_ref]
	if g_h_vals.size == 0:
		target_h = 60
	else:
		target_h = int(np.median(g_h_vals))

	g_s_vals = s[green_ref]
	g_v_vals = v[green_ref]
	g_s_min, g_s_max = int(g_s_vals.min()), int(g_s_vals.max())
	g_v_min, g_v_max = int(g_v_vals.min()), int(g_v_vals.max())

	r_s_vals = s[mask == 255]
	r_v_vals = v[mask == 255]
	if r_v_vals.size == 0:
		return img_bgr.copy()
	r_s_min, r_s_max = int(r_s_vals.min()), int(r_s_vals.max())
	r_v_min, r_v_max = int(r_v_vals.min()), int(r_v_vals.max())

	v_denom = float(r_v_max - r_v_min) if (r_v_max - r_v_min) != 0 else 1.0
	s_denom = float(r_s_max - r_s_min) if (r_s_max - r_s_min) != 0 else 1.0

	hsv_out = hsv_img.copy()
	mask_idx = (mask == 255)

	norm_v = (v.astype(np.float32) - r_v_min) / v_denom
	norm_s = (s.astype(np.float32) - r_s_min) / s_denom
	norm_v = np.clip(norm_v, 0.0, 1.0)
	norm_s = np.clip(norm_s, 0.0, 1.0)

	new_v = (g_v_min + norm_v * (g_v_max - g_v_min)).astype(np.float32)
	new_s = (g_s_min + norm_s * (g_s_max - g_s_min)).astype(np.float32)

	hsv_out[:, :, 0][mask_idx] = target_h
	hsv_out[:, :, 1][mask_idx] = np.clip(new_s[mask_idx], 0, 255)
	hsv_out[:, :, 2][mask_idx] = np.clip(new_v[mask_idx], 0, 255)

	hsv_out = hsv_out.astype(np.uint8)
	bgr_out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)
	return bgr_out

img_recolored = recolor_red_to_green_hsv(img, mask, green_ref_mask=mask_green)

cv2.imwrite('greenapple.png', img_recolored)
cv2.imshow('Mask', mask)
cv2.imshow('Recolored', img_recolored)
cv2.waitKey(0)
cv2.destroyAllWindows()
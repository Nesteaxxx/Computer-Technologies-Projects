import cv2
import numpy as np

def hog_descriptor(img_gray, win_size=(128, 128), cell_size=(8, 8), block_size=(2, 2), nbins=9):
	img = cv2.resize(img_gray, win_size, interpolation=cv2.INTER_AREA)
	img = np.float32(img) / 255.0

	gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
	gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
	mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
	ang = np.mod(ang, 180.0)

	cell_h, cell_w = cell_size[1], cell_size[0]
	n_cells_x = win_size[0] // cell_size[0]
	n_cells_y = win_size[1] // cell_size[1]

	bins = np.zeros((n_cells_y, n_cells_x, nbins), dtype=np.float32)
	bin_width = 180.0 / nbins

	for i in range(n_cells_y):
		for j in range(n_cells_x):
			y0 = i * cell_size[1]
			y1 = y0 + cell_size[1]
			x0 = j * cell_size[0]
			x1 = x0 + cell_size[0]
			cell_mag = mag[y0:y1, x0:x1]
			cell_ang = ang[y0:y1, x0:x1]
			mags = cell_mag.ravel()
			angs = cell_ang.ravel()
			hist = np.zeros(nbins, dtype=np.float32)
			for k in range(angs.size):
				a = angs[k]
				m = mags[k]
				bin_idx = int(a // bin_width) % nbins
				hist[bin_idx] += m
			bins[i, j, :] = hist

	bx = block_size[0]
	by = block_size[1]
	descriptors = []
	eps = 1e-6
	for i in range(n_cells_y - by + 1):
		for j in range(n_cells_x - bx + 1):
			block = bins[i:i+by, j:j+bx, :].ravel()
			norm = np.linalg.norm(block) + eps
			block = block / norm
			descriptors.append(block)

	if len(descriptors) == 0:
		return np.zeros(0, dtype=np.float32)
	return np.hstack(descriptors)


def classify_coin_side(image_path, show=False, out_path='result.jpg'):
	img = cv2.imread(image_path)
	if img is None:
		raise FileNotFoundError(f"Image not found: {image_path}")

	orig = img.copy()
	h, w = img.shape[:2]
	if max(h, w) > 1200:
		scale = 1200.0 / max(h, w)
		img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
								   cv2.THRESH_BINARY, 11, 2)
	binary_inv = cv2.bitwise_not(binary)

	kernel = np.ones((3, 3), np.uint8)
	clean = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

	contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	raw_edged = cv2.Canny(gray, 50, 150)
	blurred2 = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred2, 50, 150)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

	all_contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	raw_contours, _ = cv2.findContours(raw_edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	min_area = 2000
	result = img.copy()
	classifications = []
	coins_info = []

	def contours_centroids(contours_list):
		pts = []
		areas = []
		for c in contours_list:
			a = cv2.contourArea(c)
			if a <= 0:
				continue
			m = cv2.moments(c)
			if m.get("m00", 0) == 0:
				continue
			cx = m["m10"] / m["m00"]
			cy = m["m01"] / m["m00"]
			pts.append((cx, cy))
			areas.append(a)
		if len(pts) == 0:
			return np.empty((0, 2)), np.array(areas)
		return np.array(pts, dtype=np.float32), np.array(areas)

	all_pts, all_areas = contours_centroids(all_contours)

	for i, cnt in enumerate(contours):
		area = cv2.contourArea(cnt)
		if area < min_area:
			continue

		(cx_circle, cy_circle), r = cv2.minEnclosingCircle(cnt)
		r2 = r * r

		if all_pts.shape[0] == 0:
			inner_count = 0
		else:
			dx = all_pts[:, 0] - cx_circle
			dy = all_pts[:, 1] - cy_circle
			dist2 = dx * dx + dy * dy

			inner_mask = (dist2 <= r2) & (all_areas > 50)
			inner_count = int(np.count_nonzero(inner_mask))

		x, y, w, h = cv2.boundingRect(cnt)
		coin_mask = np.zeros_like(gray)
		cv2.drawContours(coin_mask, [cnt], -1, 255, -1)
		coin_gray = gray[y:y+h, x:x+w]
		coin_mask_crop = coin_mask[y:y+h, x:x+w]
		edges = cv2.Canny(coin_gray, 100, 200)
		edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=coin_mask_crop))
		coin_area = max(1, cv2.countNonZero(coin_mask_crop))
		edge_density = edge_pixels / coin_area

		hog_c = hog_descriptor(coin_gray)
		coins_info.append({
			'cnt': cnt,
			'bbox': (x, y, w, h),
			'inner_count': inner_count,
			'edge_density': edge_density,
			'hog': hog_c
		})

	labels_assigned = []
	n_coins = len(coins_info)
	if n_coins == 0:
		return []

	hogs = [info['hog'] for info in coins_info]
	hogs_valid = [h for h in hogs if h.size > 0]
	if n_coins >= 2 and len(hogs_valid) >= 2:
		L = None
		for h in hogs:
			if h.size > 0:
				L = h.size
				break
		if L is None:
			for info in coins_info:
				if info['inner_count'] >= 4 or info['edge_density'] > 0.02:
					labels_assigned.append('eagle')
				else:
					labels_assigned.append('rider')
		else:
			data = np.zeros((n_coins, L), dtype=np.float32)
			for i_h, h in enumerate(hogs):
				if h.size > 0:
					data[i_h, :min(L, h.size)] = h.ravel()[:min(L, h.size)]
			try:
				criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
				ret, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
				labels = labels.flatten()
				cluster_scores = []
				for k in range(2):
					idx = np.where(labels == k)[0]
					if idx.size == 0:
						cluster_scores.append(-1.0)
						continue
					mean_inner = np.mean([coins_info[i]['inner_count'] for i in idx])
					mean_edge = np.mean([coins_info[i]['edge_density'] for i in idx])
					cluster_scores.append(mean_inner + mean_edge * 100.0)
				eagle_cluster = int(np.argmax(cluster_scores))
				for i in range(n_coins):
					labels_assigned.append('eagle' if int(labels[i]) == eagle_cluster else 'rider')
			except Exception:
				for info in coins_info:
					if info['inner_count'] >= 4 or info['edge_density'] > 0.02:
						labels_assigned.append('eagle')
					else:
						labels_assigned.append('rider')
	else:
		for info in coins_info:
			if info['inner_count'] >= 4 or info['edge_density'] > 0.02:
				labels_assigned.append('eagle')
			else:
				labels_assigned.append('rider')

	for info, label in zip(coins_info, labels_assigned):
		x, y, w, h = info['bbox']
		inner_count = info['inner_count']
		edge_density = info['edge_density']
		classifications.append((x, y, w, h, label, inner_count, edge_density))
		color = (0, 255, 0) if label == 'eagle' else (0, 128, 255)
		cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
		cv2.putText(result, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
		cv2.putText(result, f"c:{inner_count} d:{edge_density:.3f}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

	cv2.imwrite(out_path, result)
	if show:
		cv2.imshow('Classification', result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return classifications

def main():

	img_path = r"coins\coins.png"
	out_path = r"coins\result.png"

	print('Processing', img_path)
	try:
		res = classify_coin_side(img_path, show=False, out_path=out_path)
		for i, item in enumerate(res, 1):
			x, y, w, h, label, inner_count, edge_density = item
			print(f"Coin{i}: {label}, inner_count={inner_count}, edge_density={edge_density:.3f}")
		print('Result saved to', out_path)
	except FileNotFoundError as e:
		print(e)

if __name__ == "__main__":
    main()
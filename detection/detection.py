import cv2
import numpy as np

left_img = cv2.imread('left.jpg')
center_img = cv2.imread('center.jpg')
right_img = cv2.imread('right.jpg')

frames = [left_img, center_img, right_img]
frame_names = ['left', 'center', 'right']

if left_img is None or center_img is None or right_img is None:
    print('Image loading error')
    exit()

cv2.namedWindow('Select object on first frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Select object on first frame', 1200, 800)

x, y, w, h = cv2.selectROI('Select object on first frame', left_img, False, False)
cv2.destroyAllWindows()

if w == 0 or h == 0:
    print('Object was not selected')
    exit()

left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

mask = np.zeros(left_gray.shape, dtype=np.uint8)
mask[y:y+h, x:x+w] = 255

orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)

kp1, des1 = orb.detectAndCompute(left_gray, mask)

if des1 is None or len(kp1) < 8:
    print('Too few keypoints inside the selected region')
    exit()

object_corners = np.float32([
    [x, y],
    [x + w, y],
    [x + w, y + h],
    [x, y + h]
]).reshape(-1, 1, 2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

for i in range(len(frames)):
    img = frames[i].copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if i == 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for p in kp1:
            px, py = p.pt
            cv2.circle(img, (int(px), int(py)), 2, (0, 0, 255), -1)

        cv2.namedWindow(frame_names[i], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(frame_names[i], 1200, 800)
        cv2.imshow(frame_names[i], img)
        cv2.imwrite(f'{frame_names[i]}_tracked.jpg', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue

    kp2, des2 = orb.detectAndCompute(gray, None)

    if des2 is None or len(kp2) < 8:
        print(f'Too few keypoints on frame {frame_names[i]}')
        continue

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.78 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        print(f'Not enough good matches on frame {frame_names[i]}')
        continue

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    transformed_corners = None
    inlier_points = []

    if H is not None and mask_h is not None:
        transformed_corners = cv2.perspectiveTransform(object_corners, H)

        for j in range(len(dst_pts)):
            if mask_h[j, 0]:
                px, py = dst_pts[j, 0]
                inlier_points.append((int(px), int(py)))
    else:
        A, mask_a = cv2.estimateAffinePartial2D(
            src_pts.reshape(-1, 2),
            dst_pts.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )

        if A is not None:
            transformed_corners = cv2.transform(object_corners.reshape(1, -1, 2), A).reshape(-1, 1, 2)

            if mask_a is not None:
                for j in range(len(dst_pts)):
                    if mask_a[j, 0]:
                        px, py = dst_pts[j, 0]
                        inlier_points.append((int(px), int(py)))

    if transformed_corners is None:
        print(f'Failed to detect the object on frame {frame_names[i]}')
        continue

    transformed_corners[:, 0, 0] = np.clip(transformed_corners[:, 0, 0], 0, img.shape[1] - 1)
    transformed_corners[:, 0, 1] = np.clip(transformed_corners[:, 0, 1], 0, img.shape[0] - 1)

    pts = np.array(inlier_points, dtype=np.float32)

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    new_x = int(cx - w / 2)
    new_y = int(cy - h / 2)

    new_x = max(0, min(new_x, img.shape[1] - w))
    new_y = max(0, min(new_y, img.shape[0] - h))

    cv2.rectangle(img, (new_x, new_y), (new_x + w, new_y + h), (0, 255, 0), 2)

    for px, py in inlier_points:
        cv2.circle(img, (px, py), 2, (0, 0, 255), -1)

    cv2.namedWindow(frame_names[i], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(frame_names[i], 1200, 800)
    cv2.imshow(frame_names[i], img)
    cv2.imwrite(f'{frame_names[i]}_tracked.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
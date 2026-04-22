import cv2
import numpy as np

# three frames are in the same folder
image_files = ['left.jpg', 'center.jpg', 'right.jpg']

# use left.jpg as init frame
init_index = 0

# choose ROI interactively from left.jpg
use_roi_select = True
# if prefer fixed ROI, set use_roi_select=False and define bbox below
bbox = (100, 100, 100, 100)

if len(image_files) == 0:
    raise SystemExit('No image files provided')

# Load all images
images = []
for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        raise SystemExit(f'Failed to load {fname}')
    images.append(img)

first_img = images[init_index]

if use_roi_select:
    # show full image scaled to screen and allow ROI selection over it
    draw = first_img.copy()
    h0, w0 = draw.shape[:2]
    max_dim = 1600
    if max(h0, w0) > max_dim:
        scale = max_dim / max(h0, w0)
        draw = cv2.resize(draw, (int(w0 * scale), int(h0 * scale)))
    else:
        scale = 1.0

    cv2.imshow('Select ROI on left.jpg', draw)
    roi_scaled = cv2.selectROI('Select ROI on left.jpg', draw, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi_scaled)
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    if w == 0 or h == 0:
        raise SystemExit('ROI not selected')
else:
    x, y, w, h = bbox

if w <= 0 or h <= 0:
    raise SystemExit('Invalid bbox')

# Extract template from left image
template = first_img[y:y+h, x:x+w]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Initialize SIFT
sift = cv2.SIFT_create()
kp_template, des_template = sift.detectAndCompute(template_gray, None)

if des_template is None or len(kp_template) < 4:
    raise SystemExit('Not enough features found in template. Please select a better ROI')

# Initialize BFMatcher
bf = cv2.BFMatcher()

# Initial rectangle coordinates
init_rect = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]]).reshape(-1, 1, 2)

# Process all images
results = []

for idx, (fname, frame) in enumerate(zip(image_files, images)):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if idx == init_index:
        # For left image, use the selected ROI directly
        tracked_rect = init_rect
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        print(f'{fname} coordinates: top-left ({x_min}, {y_min}), bottom-right ({x_max}, {y_max})')
        results.append((frame, tracked_rect, fname))
        continue
    
    # Find keypoints and descriptors in current frame
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    
    if des_frame is None or len(kp_frame) < 4:
        print(f'WARNING: Not enough features found in {fname}')
        results.append((frame, None, fname))
        continue
    
    # Match descriptors
    matches = bf.knnMatch(des_template, des_frame, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    print(f'{fname}: Found {len(good_matches)} good matches out of {len(matches)}')
    
    if len(good_matches) < 4:
        print(f'WARNING: Not enough good matches for {fname} (need at least 4, got {len(good_matches)})')
        results.append((frame, None, fname))
        continue
    
    # Get matched keypoints
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f'WARNING: Homography computation failed for {fname}')
        results.append((frame, None, fname))
        continue
    
    # Apply homography to initial rectangle
    tracked_rect = cv2.perspectiveTransform(init_rect, H)
    
    # Get bounding rectangle coordinates
    pts = np.int32(tracked_rect)
    x_min, y_min = tuple(pts[:, 0, :].min(axis=0))
    x_max, y_max = tuple(pts[:, 0, :].max(axis=0))
    
    print(f'{fname} coordinates: top-left ({x_min}, {y_min}), bottom-right ({x_max}, {y_max})')
    results.append((frame, tracked_rect, fname))

# Display all results
for frame, rect, fname in results:
    out = frame.copy()
    
    if rect is not None:
        # Draw the rectangle
        cv2.polylines(out, [np.int32(rect)], isClosed=True, color=(0, 0, 255), thickness=3)
        
        # Get coordinates for text display
        pts = np.int32(rect)
        x_min, y_min = tuple(pts[:, 0, :].min(axis=0))
        x_max, y_max = tuple(pts[:, 0, :].max(axis=0))
        
        # Add text with coordinates
        cv2.putText(out, f'Object detected', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out, f'Top-left: ({x_min}, {y_min})', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(out, f'Bottom-right: ({x_max}, {y_max})', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(out, 'Object not found', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(out, f'Frame: {fname}', (10, out.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Resize for display if too large
    h, w = out.shape[:2]
    if max(h, w) > 800:
        scale = 800 / max(h, w)
        out = cv2.resize(out, (int(w * scale), int(h * scale)))
    
    cv2.imshow(f'Result - {fname}', out)

print("\nPress any key to exit...")
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
import cv2

img = cv2.imread("coins\\coins.png")
cv2.imshow('Original', img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
# cv2.imshow('Blurred',gray_blurred)
_, img_binary = cv2.threshold(gray_blurred, 240, 255, cv2.THRESH_BINARY)
# cv2.imshow('Binary',img_binary)

kernel = np.ones((3,3), np.uint8)
img_dilated = cv2.dilate(img_binary, kernel, iterations=1)
# cv2.imshow('Dilated',img_dilated)

sobel_x = cv2.Sobel(img_dilated, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_dilated, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.uint8(np.clip(magnitude, 0, 255))
# cv2.imshow('Sobel Magnitude', magnitude)

contours, _ = cv2.findContours(magnitude, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
# cv2.imshow('Contours',contour_img)

img_area = img.copy()
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    cv2.drawContours(img_area, [contour], -1, (0, 255, 0), 1)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img_area, f"{area:.0f}", (cx-15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
cv2.imshow('Areas on image', img_area)

cv2.waitKey(0)
cv2.destroyAllWindows()
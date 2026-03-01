import cv2
import numpy as np

def kmeans_clustering_grayscale(image_path, n_clusters=3):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixels = gray.reshape(-1, 1)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    clustered_data = centers[labels.flatten()]
    clustered_img = clustered_data.reshape(gray.shape)
    
    labels_reshaped = labels.reshape(gray.shape)
    percentages = [(np.sum(labels_reshaped == i) / labels_reshaped.size * 100) for i in range(n_clusters)]
    
    vis = np.zeros((gray.shape[0], gray.shape[1] * 2, 3), dtype=np.uint8)
    vis.fill(255)
    
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    clustered_3ch = cv2.cvtColor(clustered_img, cv2.COLOR_GRAY2BGR)
    
    vis[0:gray.shape[0], 0:gray.shape[1]] = gray_3ch
    vis[0:gray.shape[0], gray.shape[1]:2*gray.shape[1]] = clustered_3ch
    
    return vis, centers.flatten(), percentages

def kmeans_clustering_color(image_path, n_clusters):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    centers = np.uint8(centers)
    clustered_data = centers[labels.flatten()]
    clustered_img = clustered_data.reshape(img_rgb.shape)
    clustered_bgr = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR)
    
    vis = np.zeros((img.shape[0], img.shape[1] * 2, 3), dtype=np.uint8)
    vis.fill(255)
    
    vis[0:img.shape[0], 0:img.shape[1]] = img
    vis[0:img.shape[0], img.shape[1]:2*img.shape[1]] = clustered_bgr
    
    labels_reshaped = labels.reshape(img_rgb.shape[:2])
    percentages = [(np.sum(labels_reshaped == i) / labels_reshaped.size * 100) for i in range(n_clusters)]
    
    return vis, centers, percentages

def main():
    image_path = r"claster\\redapple.png"
    
    print("3 clusters (grayscale):")
    gray_result, gray_centers, gray_percents = kmeans_clustering_grayscale(image_path, 3)
    cv2.imshow("Grayscale 3 clusters", gray_result)
    cv2.waitKey(0)
    for i, (c, p) in enumerate(zip(gray_centers, gray_percents)):
        print(f"Cluster {i}: brightness {c} {p:.1f}%")
    
    print("\n5 clusters (color):")
    vis5, centers5, percents5 = kmeans_clustering_color(image_path, 5)
    cv2.imshow("Color 5 clusters", vis5)
    cv2.waitKey(0)
    for i, (c, p) in enumerate(zip(centers5, percents5)):
        print(f"Cluster {i}: RGB{c} {p:.1f}%")
    
    print("\n7 clusters (color):")
    vis7, centers7, percents7 = kmeans_clustering_color(image_path, 7)
    cv2.imshow("Color 7 clusters", vis7)
    cv2.waitKey(0)
    for i, (c, p) in enumerate(zip(centers7, percents7)):
        print(f"Cluster {i}: RGB{c} {p:.1f}%")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
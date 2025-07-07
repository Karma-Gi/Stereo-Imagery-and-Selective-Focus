import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Stereo calibration values from the coursework
focal_length_px = 5806.559
doffs = 114.291
baseline_mm = 174.019

# Target image resolution
original_width = 3088
resized_width = 740

scale_factor = resized_width / original_width
focal_length_px *= scale_factor
doffs *= scale_factor


# ================================================
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
    stereo.setPreFilterSize(5)  # Preprocessing filter size
    stereo.setPreFilterCap(31)  # Cutoffs for preprocessing filters
    stereo.setTextureThreshold(10)  # Filter low texture areas
    stereo.setUniquenessRatio(15)  # Uniqueness ratio to reduce false matches
    stereo.setSpeckleWindowSize(100)  # Filtering small speckled noise
    stereo.setSpeckleRange(32)  # Maximum difference between spots
    stereo.setDisp12MaxDiff(1)  # Maximum difference between left and right consistency checks

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Avoid zero depth
    disparity = disparity.astype(np.float32) / 16.0

    return disparity


# ================================================


# ================================================
def plot_3d(disparity, focal_length, baseline, doffs):
    h, w = disparity.shape

    x_coords = []
    y_coords = []
    z_coords = []

    for y in range(0, h, 4):
        for x in range(0, w, 4):
            d = disparity[y, x]
            if d > 0:
                Z = (baseline * focal_length) / (d + doffs)
                if 1000 < Z < 8000:
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(Z)

    if not x_coords:
        print("No valid points for plotting.")
        return

    fig = plt.figure(figsize=(15, 5))

    # ----------- (a) 3D View -------------
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x_coords, y_coords, z_coords, c='royalblue', s=1)
    ax1.set_title("3D View")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.view_init(elev=30, azim=-60)
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0)
    ax1.set_zlim(min(z_coords), max(z_coords))

    # ----------- (b) Top View (X-Y) -------------
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(x_coords, y_coords, z_coords, c='royalblue', s=1)
    ax2.set_title("Top View (X-Y)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.view_init(elev=90, azim=-90)
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_zlim(min(z_coords), max(z_coords))

    # ----------- (c) Side View (X-Z) -------------
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(x_coords, y_coords, z_coords, c='royalblue', s=1)
    ax3.set_title("Side View (X-Z)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.view_init(elev=0, azim=-90)
    ax3.set_xlim(0, w)
    ax3.set_ylim(h, 0)
    ax3.set_zlim(min(z_coords), max(z_coords))

    plt.tight_layout()
    plt.savefig('3d_views_cleaned.png', dpi=300, bbox_inches='tight')
    plt.show()


# ================================================


# ================================================
def update(val):
    num_disp = cv2.getTrackbarPos('NumDisparities', 'Disparity')
    block_size = cv2.getTrackbarPos('BlockSize', 'Disparity')

    num_disp = max(16, num_disp * 16)
    block_size = max(5, block_size | 1)

    # Use Sobel edge detection in the update function
    imgL_sobel = cv2.Sobel(imgL, cv2.CV_64F, 1, 0, ksize=3)
    imgR_sobel = cv2.Sobel(imgR, cv2.CV_64F, 1, 0, ksize=3)
    imgL_sobel = cv2.normalize(imgL_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    imgR_sobel = cv2.normalize(imgR_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp = getDisparityMap(imgL_sobel, imgR_sobel, num_disp, block_size)

    # Add post-processing: median filter denoising, replace median filter in update function with more robust filtering
    disp_filtered = cv2.medianBlur(disp, 5)
    disp_filtered = cv2.morphologyEx(disp_filtered, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Normalization is used for display, adjust the normalization method in the update function
    disp_norm = cv2.normalize(disp_filtered, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = cv2.convertScaleAbs(disp_norm, alpha=1.5, beta=0)  # Enhanced contrast

    cv2.imshow('Disparity', disp_norm)

    global current_disparity
    current_disparity = disp_filtered


# ================================================


if __name__ == '__main__':
    # Load images
    imgL = cv2.imread('./images/umbrellaL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('./images/umbrellaR.png', cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        print("Error loading images.")
        sys.exit()

    # Resize for speed (optional if not already resized)
    imgL = cv2.resize(imgL, (740, 505))
    imgR = cv2.resize(imgR, (740, 505))

    # Create window and trackbars
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('NumDisparities', 'Disparity', 6, 20, update)  # steps of 16
    cv2.createTrackbar('BlockSize', 'Disparity', 15, 30, update)  # odd values only

    current_disparity = getDisparityMap(imgL, imgR, 64, 5)
    update(0)

    print("Press '3' to show 3D plot. Press SPACE or ESC to quit.")

    while True:
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
        elif key == ord('3'):
            plot_3d(current_disparity, focal_length_px, baseline_mm, doffs)

    cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import time


# ====================
def getDisparityMap(imL, imR, numDisparities, blockSize):
    blockSize = max(5, blockSize | 1)
    blockSize = min(blockSize, min(imL.shape))  # Prevent exceeding image size

    numDisparities = max(16, numDisparities)
    numDisparities = numDisparities + (16 - numDisparities % 16)  # Guaranteed to be a multiple of 16

    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    stereo.setPreFilterCap(31)
    stereo.setPreFilterSize(9)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)

    disparity = stereo.compute(imL, imR).astype(np.float32) / 16.0
    disparity[disparity <= 0] = 0.1
    return disparity


# ====================
def applySelectiveFocus(color_img, disparity, k):
    depth = 1.0 / (disparity + k)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)

    # Adaptive thresholding + Swell corrosion is more robust
    # _, mask = cv2.threshold(depth, 110, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.inRange(depth, 20, 150)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill in small holes
    mask = cv2.GaussianBlur(mask, (9, 9), 0)  # Smooth edges
    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.medianBlur(mask, 5)  # Remove the small noise

    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    foreground = cv2.bitwise_and(color_img, color_img, mask=mask)
    background = cv2.bitwise_and(gray, gray, mask=cv2.bitwise_not(mask))
    result = cv2.add(foreground, background)

    return result, depth, mask


# ====================
def update(val):
    num_disp = cv2.getTrackbarPos("NumDisparities", "Selective Focus") * 16
    block_size = cv2.getTrackbarPos("BlockSize", "Selective Focus")
    k = cv2.getTrackbarPos("k", "Selective Focus") / 10.0

    # Gaussian blur preprocessing to reduce noise
    blurL = cv2.GaussianBlur(imgL_gray, (5, 5), 0)
    blurR = cv2.GaussianBlur(imgR_gray, (5, 5), 0)

    disparity = getDisparityMap(blurL, blurR, num_disp, block_size)
    result, depth_img, mask = applySelectiveFocus(imgL_color, disparity, k)

    cv2.imshow("Selective Focus", result)
    cv2.imshow("Depth Map", depth_img)
    cv2.imshow("Mask", mask)

    global current_result, current_depth, current_mask
    current_result = result
    current_depth = depth_img
    current_mask = mask


# ====================
if __name__ == '__main__':
    # Load color & grayscale maps
    imgL_color = cv2.imread('./images/girlL.png')
    imgR_color = cv2.imread('./images/girlR.png')
    imgL_gray = cv2.cvtColor(imgL_color, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR_color, cv2.COLOR_BGR2GRAY)

    # Create windows & trackbar
    cv2.namedWindow("Selective Focus", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("NumDisparities", "Selective Focus", 6, 16, update)
    cv2.createTrackbar("BlockSize", "Selective Focus", 15, 100, update)
    cv2.createTrackbar("k", "Selective Focus", 10, 50, update)

    update(0)

    print("Press SPACE or ESC to exit")
    while True:
        key = cv2.waitKey(1)
        if key == 27 or key == ord(' '):
            break
        elif key == ord('s') or key == ord('S'):
            os.makedirs("output", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"output/result_{timestamp}.png", current_result)
            cv2.imwrite(f"output/depth_{timestamp}.png", current_depth)
            cv2.imwrite(f"output/mask_{timestamp}.png", current_mask)
            print(f"Images saved to output/ at {timestamp}")

    cv2.destroyAllWindows()

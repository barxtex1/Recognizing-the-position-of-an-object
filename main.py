import cv2
import numpy as np
from math import sqrt, pi, atan
import time


def calculate_side(image):
    mask = cv2.inRange(image, np.array([79, 99, 83]), np.array([180, 256, 256]))
    kernel_erosion = np.ones((2, 2), np.uint8)
    kernel_opening = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel_erosion, iterations=1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel_opening)
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for count, contour in enumerate(contours):
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Calculate the position of the corners of the contour
        if len(np.squeeze(approx)) == 4:  # Take only with 4 corners
            # Positions of the corners
            x1, y1 = np.squeeze(approx)[0]
            x2, y2 = np.squeeze(approx)[1]
            x3, y3 = np.squeeze(approx)[2]
            x4, y4 = np.squeeze(approx)[3]
            # Length of each side
            side1 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            side2 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
            side3 = sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
            side4 = sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)
            if abs(side1-side2) < 2 and abs(side1-side3) < 2 and abs(side1-side4) < 2 and side1 > 10:
                print("side", side1)
                # Draw each contour only for visualisation purposes
                #cv2.drawContours(image, contours, count, (0, 0, 255), 2)
                #cv2.imshow("image", image)
                #cv2.waitKey()
                return side1

def rotate_table(gray):
    height, width = gray.shape
#     window_of_mask = 150  # then window shape = (300,300)
#     mask = gray[height // 2 - window_of_mask:height // 2 + window_of_mask,
#            width // 2 - window_of_mask:width // 2 + window_of_mask]
#     height, width = mask.shape
    cX, cY = (height // 2, width // 2)  # center point of frame

    harris_corners = cv2.cornerHarris(gray, 4, 3, 0.23)  # detect corners
    corners = np.zeros_like(harris_corners)
    corners[harris_corners > 0.025 * harris_corners.max()] = 1
    
    result = np.where(corners == 1)  
    corners_points = list(zip(result[0], result[1]))  # list of coord corners
    cv2.imshow("corners", corners)  # DEBUG
    
#    x, y = corners_points[len(corners_points) // 2]  # could be any point (I take middle one)

#     correct_length = sqrt((166 - 57) ** 2 + (142 - 184) ** 2)  # calculated to test == 116.8
#     for pts in corners_points:
#         x0, y0 = pts
#         length_of_pts_to_corner = sqrt((x - x0) ** 2 + (y - y0) ** 2)
#         error = abs(correct_length - length_of_pts_to_corner)
#         if error < 0.4:
# 
#             a = (y - y0) / (x - x0)  # slope of a straight line
#             theta = atan(abs(x - x0) / abs(y - y0))  # angle of rotate (in radians)
#             if a < 0:
#                 M = cv2.getRotationMatrix2D((cX, cY), -theta * 180 / pi, 1.0)
#             else:
#                 M = cv2.getRotationMatrix2D((cX, cY), theta * 180 / pi, 1.0)
#             rotated = cv2.warpAffine(mask, M, (width, height))
#             return rotated


def main():
    cap = cv2.VideoCapture(0)  # open the default camera

    key = ord('a')
    while key != ord('q'):
        start_time = time.time()
        
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        side = calculate_side(frame)
#        rotate_table(gray_img)

        
        # Display the result of our processing
#         cv2.imshow('gray_img', gray_img)
        print((time.time() - start_time)*1000, "miliseconds")  # use for process debugging

        # Wait a little (5 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(5)



    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


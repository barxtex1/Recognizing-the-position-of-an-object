import cv2
import numpy as np
from math import sqrt, pi, atan
import time
import threading
import click


def display_window(frame):
    cv2.imshow("Original frame", frame)


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
            if abs(side1 - side2) < 2 and abs(side1 - side3) < 2 and abs(side1 - side4) < 2 and side1 > 10:
                print("side", side1)
                # Draw each contour only for visualisation purposes
                # cv2.drawContours(image, contours, count, (0, 0, 255), 2)
                # cv2.imshow("image", image)
                # cv2.waitKey()
                return side1


def rotate_table(gray, side):
    width, height = gray.shape
    cX, cY = (height // 2, width // 2)  # center point of frame

    harris_corners = cv2.cornerHarris(gray, 4, 1, 0.11)  # detect corners
    corners = np.zeros_like(harris_corners)
    corners[harris_corners > 0.025 * harris_corners.max()] = 1

    result = np.where(corners == 1)
    corners_points = list(zip(result[0], result[1]))  # list of coord corners
    # cv2.imshow("corners", corners)  # DEBUG

    x, y = corners_points[len(corners_points) // 2]  # could be any point (I take middle one)  MAYBE NOT

    # correct_length = sqrt((166 - 57) ** 2 + (142 - 184) ** 2)  # calculated to test == 116.8
    for pts in corners_points:
        x0, y0 = pts
        length_of_pts_to_corner = sqrt((x - x0) ** 2 + (y - y0) ** 2)
        error = abs(side - length_of_pts_to_corner)
        if error < 0.4:
            try:
                a = (y - y0) / (x - x0)  # slope of a straight line
                theta = atan(abs(x - x0) / abs(y - y0))  # angle of rotate (in radians)
                if a < 0:
                    M = cv2.getRotationMatrix2D((cX, cY), -theta * 180 / pi, 1.0)
                else:
                    M = cv2.getRotationMatrix2D((cX, cY), theta * 180 / pi, 1.0)
                rotated = cv2.warpAffine(gray, M, (width, height))
                cv2.imshow("rotated", rotated)  # DEBUG
                return rotated
            except:
                pass  # For while


def cut_out_square(img, side, kernel_size):
    height, width = img.shape
    harris_corners = cv2.cornerHarris(img, 7, 1, 0.21)  # detect corners
    corners = np.zeros_like(harris_corners)
    corners[harris_corners > 0.025 * harris_corners.max()] = 1

    result = np.where(corners == 1)
    corners_points = list(zip(result[0], result[1]))  # list of coord corners
    cv2.imshow("corners", corners)  # DEBUG
    for x, y in corners_points:
        if 0.15 * height < x < 0.85 * height and 0.15 * width < y < 0.85 * width:
            print("height, width:", height, width)
            print("coord of pick corner:", x, y)
            return img[x:x + int(kernel_size * side), y:y + int(kernel_size * side)]

    # diagonal_length = sqrt((33 - 215) ** 2 + (55 - 236) ** 2)  # calculated to test
    # diagonal_length = 3 * side * sqrt(2)
    # for pts1 in corners_points:
    #     x, y = pts1
    #     for pts2 in corners_points:
    #         x0, y0 = pts2
    #         length_of_pts1_to_pts2 = sqrt((x - x0) ** 2 + (y - y0) ** 2)
    #         error = abs(diagonal_length - length_of_pts1_to_pts2)
    #         if error < 0.4:
    #             cv2.line(img, (y, x), (y0, x0), (0, 0, 255), 2)  # ADD to test
    #             if y > 150:
    #                 # plot_cut_out_square(img, x, y) # ADD to test
    #                 return img[x:x + int(3 * side), y - int(3 * side):y]
    #             else:
    #                 # plot_cut_out_square(img, x, y) # ADD to test
    #                 return img[x:x + int(3 * side), y:y + int(3 * side)]


@click.command(no_args_is_help=True)
@click.option('-k', '--kernel', type=int, help='Size of kernel')
def main(kernel):
    cap = cv2.VideoCapture(0)  # open the default camera
    # frame = cv2.imread("resources/image-night.jpg")

    key = ord('a')
    while key != ord('q'):
        start_time = time.time()
        print(kernel)
        # Capture frame-by-frame
        ret, frame = cap.read()
        threading.Thread(target=display_window(frame), args=(1,)).start()  # DISPLAY

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        side = calculate_side(frame)
        if side is not None:  # Check if program calculate side of pixel
            rotated_frame = rotate_table(gray_frame, side)
            if rotated_frame is not None:
                square = cut_out_square(rotated_frame, side, kernel)
                if square is not None:
                    cv2.imshow("Cut out square", square)
                    print("CUT OUT SIDE:", side)
                    print("---------------------------------------------------------------------------------------")
                else:
                    pass
                    print("[-] Warning: square missing")
            else:
                pass
                print("[-] Warning: rotated frame = None (probably error > 0.4 for each point)")
        else:
            pass
            print("[-] Warning: Side = None")
        # Display the result of our processing
        #         cv2.imshow('gray_img', gray_img)
        print((time.time() - start_time) * 1000, "miliseconds")  # use for process debugging

        # Wait a little (5 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(5)

    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

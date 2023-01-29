import json
import math
import pickle
import socket
import time
from math import dist
from binary_orientation import binary_orientation
import cv2
import numpy as np
from scipy.spatial.distance import cdist


def display_position(table, kernel, position):
    height, width = position
    board_len = 30
    table_size = table.shape[0]
    side_square_size = table_size // board_len
    cv2.rectangle(table, (width * side_square_size, height * side_square_size),
                  (width * side_square_size + kernel * side_square_size,
                   height * side_square_size + kernel * side_square_size),
                  (0, 0, 255), 3)
    cv2.imshow("Position visualization", table)
    # cv2.imwrite("visualization.png", table)


def calculate_side(image, kernel):
    # HSV
    # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(gray_frame, np.array(thresh[0]), np.array(thresh[1]))

    # SIMPLE THRESHOLD VISUALIZATION
    # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(gray_frame, thresh, 255, cv2.THRESH_BINARY)

    # ADAPTIVE
    # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # mask = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thresh, 0)

    # OTSU
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, otsu_mask = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blur_gray, 10, 17)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 60  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(gray_frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

        binary_result = otsu_mask - line_image
        binary_result = cv2.erode(binary_result, (7, 7), iterations=1)
        ret, line_mask = cv2.threshold(line_image, 127, 255, cv2.THRESH_BINARY_INV)
        edge_length = kernel * 3  # DEBUG

        contours, hierarchy = cv2.findContours(binary_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for count, contour in enumerate(contours):
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)  # Calculate the position of the corners of the contour
            if len(np.squeeze(approx)) == 4:  # Take only with 4 corners
                corners = np.squeeze(approx)
                # Length of each side
                lengths = []
                for i in range(len(corners)):
                    if i != len(corners) - 1:
                        lengths.append(math.dist(corners[i], corners[i + 1]))
                    else:
                        lengths.append(math.dist(corners[0], corners[i]))
                if all(abs(lengths[0] - length) < 2 for length in lengths[1:]) and lengths[0] > 20:
                    # Draw each contour only for visualisation purposes
                    # cv2.drawContours(image, contours, count, (0, 0, 255), 2)
                    return lengths[0] + edge_length, contour, binary_result, line_mask
        return None, None, None, None
    else:
        return None, None, None, None


def rotate_table(image, contour_field, binary_image, binary_line):
    # FIRST METHOD
    width, height, _ = image.shape
    cX, cY = (height // 2, width // 2)  # center point of frame
    if contour_field is None:
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, otsu_mask = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blur_gray, 10, 17)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 60  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(gray_frame) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

            binary_result = otsu_mask - line_image
            binary_result = cv2.erode(binary_result, (7, 7), iterations=1)
            ret, line_mask = cv2.threshold(line_image, 127, 255, cv2.THRESH_BINARY_INV)

            contours, hierarchy = cv2.findContours(binary_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for count, contour in enumerate(contours):
                epsilon = 0.1 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon,
                                          True)  # Calculate the position of the corners of the contour
                if len(np.squeeze(approx)) == 4:  # Take only with 4 corners
                    corners = np.squeeze(approx)
                    # Length of each side
                    lengths = []
                    for i in range(len(corners)):
                        if i != len(corners) - 1:
                            lengths.append(math.dist(corners[i], corners[i + 1]))
                        else:
                            lengths.append(math.dist(corners[0], corners[i]))
                    if all(abs(lengths[0] - length) < 2 for length in lengths[1:]) and lengths[0] > 20:
                        rect = cv2.minAreaRect(contour)
                        angle = rect[2]
                        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                        rotated_binary = cv2.warpAffine(binary_image, M, (height, width))
                        rotated_frame = cv2.warpAffine(image, M, (height, width))
                        rotated_binary_line = cv2.warpAffine(line_mask, M, (height, width))
                        return rotated_binary, rotated_frame, rotated_binary_line
    else:
        rect = cv2.minAreaRect(contour_field)
        angle = rect[2]
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated_binary = cv2.warpAffine(binary_image, M, (height, width))
        rotated_frame = cv2.warpAffine(image, M, (height, width))
        rotated_binary_line = cv2.warpAffine(binary_line, M, (height, width))
        return rotated_binary, rotated_frame, rotated_binary_line

# def find_cutout(position):
#     if position == 0:


def cut_out_square(binary_image, side, kernel_size, frame, rotated_binary_line):
    # FIRST METHOD
    # height, width = img.shape
    # harris_corners = cv2.cornerHarris(img, 7, 1, 0.21)  # detect corners
    # corners = np.zeros_like(harris_corners)
    # corners[harris_corners > 0.025 * harris_corners.max()] = 1
    #
    # result = np.where(corners == 1)
    # corners_points = list(zip(result[0], result[1]))  # list of coord corners
    # # cv2.imshow("corners", corners)  # DEBUG
    # for x, y in corners_points:
    #     if 0.15 * height < x < 0.85 * height and 0.15 * width < y < 0.85 * width:
    #         return img[x:x + int(kernel_size * side), y:y + int(kernel_size * side)]

    # SECOND METHOD
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # height_binary, width_binary = binary_image.shape
    # center_square_binary = binary_image[height_binary // 2 - int(0.6 * kernel_size * side):height_binary // 2 + int(
    #     0.6 * kernel_size * side),
    #                        width_binary // 2 - int(0.6 * kernel_size * side):width_binary // 2 + int(
    #                            0.6 * kernel_size * side)]
    #
    # height_frame, width_frame, _ = frame.shape
    # center_square_frame = frame[
    #                       height_frame // 2 - int(0.6 * kernel_size * side):height_frame // 2 + int(
    #                           0.6 * kernel_size * side),
    #                       width_frame // 2 - int(0.6 * kernel_size * side):width_frame // 2 + int(
    #                           0.6 * kernel_size * side)]
    #
    # return center_square_binary, center_square_frame

    # --------------------------------------------------------------------------------------------------

    # h_center_square, w_center_square = center_square.shape
    #
    # center_square = cv2.GaussianBlur(center_square, (21, 21), 0)

    # harris_corners = cv2.cornerHarris(center_square, 9, 1, 0.2)  # detect corners
    # # --------------------------------------------------------
    #
    # # dst = cv2.dilate(harris_corners, None)
    # ret, dst = cv2.threshold(harris_corners, 0.025 * harris_corners.max(), 255, cv2.THRESH_BINARY)
    #
    # dst = np.uint8(dst)
    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners_points = cv2.cornerSubPix(center_square, np.float32(centroids), (20, 20), (-1, -1), criteria)
    # corners_points = np.uint8(corners_points)
    # --------------------------------------------------------
    # WITHOUT SUBPIX
    # corners = np.zeros_like(harris_corners)
    # corners[harris_corners > 0.025 * harris_corners.max()] = 1

    # cv2.imshow("cornes_square", corners)
    # cv2.waitKey()

    # result = np.where(corners == 1)
    # corners_points = list(zip(result[0], result[1]))  # list of coord corners
    # print(corners_points)
    # corners_points = np.array(corners_points)  # convert list to numpy array

    # TRACK
    # corners_points = cv2.goodFeaturesToTrack(center_square, 10, 0.2, 40)
    # corners_points = np.int0(corners_points)
    # corners_points = np.squeeze(corners_points)
    # # track = cv2.cvtColor(center_square, cv2.COLOR_GRAY2BGR)
    # # for corner_coord in corners_points:
    # #     x, y = corner_coord
    # #     cv2.circle(track, (x, y), 3, (0, 0, 255), -1)
    # #
    # # return track
    #
    # distance = cdist(np.array([[0, 0]]), corners_points)
    # idx = np.argmin(distance)
    # min_distance = np.min(distance)
    #
    # for x in np.array([[[0, w_center_square]], [[h_center_square, 0]], [[h_center_square, w_center_square]]]):
    #     distance = cdist(x, corners_points)
    #     if min_distance > np.min(distance):
    #         min_distance = np.min(distance)
    #         idx = np.argmin(distance)
    #
    # h_min_d_square, w_min_d_square = corners_points[idx]  # Coordinate of the min. corner distance between one of the 4 corners of the window
    # x_img, y_img = height // 2 - int(0.5 * kernel_size * side), width // 2 - int(0.5 * kernel_size * side)
    # h_min_d, w_min_d = x_img + h_min_d_square, y_img + w_min_d_square
    # if h_min_d_square < h_center_square // 2 and w_min_d_square < w_center_square // 2:
    #     square_ = binary_image[h_min_d: h_min_d + int(kernel_size * side), w_min_d: w_min_d + int(kernel_size * side)]
    # elif h_min_d_square < h_center_square // 2 and w_min_d_square > w_center_square // 2:
    #     square_ = binary_image[h_min_d: h_min_d + int(kernel_size * side), w_min_d - int(kernel_size * side): w_min_d]
    # elif h_min_d_square > h_center_square // 2 and w_min_d_square < w_center_square // 2:
    #     square_ = binary_image[h_min_d - int(kernel_size * side): h_min_d, w_min_d: w_min_d + int(kernel_size * side)]
    # else:
    #     square_ = binary_image[h_min_d - int(kernel_size * side): h_min_d, w_min_d - int(kernel_size * side): w_min_d]
    #
    # # cv2.imshow("square_2", square_)  # DEBUG
    # # cv2.imshow("corners", corners)
    # return square_
    height_binary, width_binary = rotated_binary_line.shape
    center_square_binary = binary_image[height_binary // 2 - int(0.6 * kernel_size * side):height_binary // 2 + int(
        0.6 * kernel_size * side),
                           width_binary // 2 - int(0.6 * kernel_size * side):width_binary // 2 + int(
                               0.6 * kernel_size * side)]
    center_square_frame = frame[height_binary // 2 - int(0.6 * kernel_size * side):height_binary // 2 + int(
        0.6 * kernel_size * side),
                           width_binary // 2 - int(0.6 * kernel_size * side):width_binary // 2 + int(
                               0.6 * kernel_size * side)]
    center_square = rotated_binary_line[height_binary // 2 - int(0.6 * kernel_size * side):height_binary // 2 + int(
        0.6 * kernel_size * side),
                           width_binary // 2 - int(0.6 * kernel_size * side):width_binary // 2 + int(
                               0.6 * kernel_size * side)]
    side_of_window = center_square.shape[0] // kernel_size
    kernel_erosion = np.ones((13, 13), np.uint8)
    center_square = cv2.erode(center_square, kernel_erosion, iterations=1)
    contours, _ = cv2.findContours(center_square, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # pomoc = center_square_frame.copy()
    for count, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < (side**2)*0.55:
            continue

        # cv2.drawContours(pomoc, contours, count, (0, 0, 255), 2)
        # cv2.imshow("center", pomoc)
        # cv2.waitKey()
        x, y, w, h = cv2.boundingRect(contour)
        center_point_w, center_point_h = (x + w // 2, y + h // 2)
        for i in range(kernel_size):
            for j in range(kernel_size):
                if j * side_of_window < center_point_w < (j + 1) * side_of_window and i * side_of_window < center_point_h < (i + 1) * side_of_window:
                    h_binary, w_binary = int(side*i), int(side*j)
                    difference_h = abs(y - h_binary)
                    difference_w = abs(x - w_binary)
                    binary_square = center_square_binary[difference_h: difference_h + int(kernel_size * side), difference_w: difference_w + int(kernel_size * side)]
                    frame_square = center_square_frame[difference_h: difference_h + int(kernel_size * side), difference_w: difference_w + int(kernel_size * side)]
                    return binary_square, frame_square
    return None, None


def find_mask_of_triangle(hsv_square, thresh):
    mask = cv2.inRange(hsv_square, np.array(thresh[0][0]), np.array(thresh[0][1]))
    kernelOpen = np.ones((5, 5))
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    return maskOpen


def find_orientation(triangle, side):
    contours, _ = cv2.findContours(triangle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < side/8:
            continue
        epsilon = 0.15 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Calculate the position of the corners of the contour
        # print(len(np.squeeze(approx)))
        if len(np.squeeze(approx)) == 3:  # Take only with 3 corners (triangle)
            # find orientation
            coord_of_corners = np.squeeze(approx)
            # print(coord_of_corners)
            w_1, h_1 = coord_of_corners[0]
            w_2, h_2 = coord_of_corners[1]
            error_w = abs(w_2 - w_1)
            error_h = abs(h_2 - h_1)
            if w_2 > w_1 and error_w > 3:
                return "up"
            elif w_2 < w_1 and error_h > 3:
                return "right"
            elif w_2 < w_1 and error_h < 3:
                return "left"
            else:
                return "down"


# @click.command(no_args_is_help=True)
# @click.option('-k', '--kernel', required=True, type=str, help='Size of kernel')
def main(kernel):
    # Upload the required files
    positions = json.loads(open("resources/positions_30_30.json").read())
    table = cv2.imread("resources/table_30_30.png")
    h, w, _ = table.shape
    table = cv2.resize(table, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    thresholds = json.loads(open("threshold_arrays.json").read())

    # STREAM CONFIGURATION
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 100000000)
    serverip = "192.168.125.58"
    serverport = 8000

    cap = cv2.VideoCapture(0)  # open the default camera
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    fps_return = cap.set(cv2.CAP_PROP_FPS, 60)  # Set frame rate to 30
    print("Camera fps status", fps_return)

    if_side = False
    pos = False
    position = None
    side = None
    key = ord('a')
    avg_time = []
    process_time = 0
    cos = 0
    while key != ord('q'):
        # Capture frame-by-frame
        cos += 1
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if not if_side:
            side, contour, binary_image, binary_line = calculate_side(frame, kernel)
        else:
            contour = None
            binary_image = None
            binary_line = None
        if side is not None:  # Check if program calculate side of binary field
            rotated_binary_frame, rotated_frame, rotated_binary_line = rotate_table(frame, contour, binary_image, binary_line)

            if rotated_binary_frame is not None:
                binary_square, frame_square = cut_out_square(rotated_binary_frame, side, kernel, rotated_frame, rotated_binary_line)
                if binary_square is not None:
                    ret, buffer = cv2.imencode(".jpg", binary_square, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    x_as_bytes = pickle.dumps(buffer)
                    s.sendto(x_as_bytes, (serverip, serverport))

                    hsv_square = cv2.cvtColor(frame_square, cv2.COLOR_BGR2HSV)
                    triangle = find_mask_of_triangle(hsv_square, thresholds["triangle"])
                    orientation = find_orientation(triangle, side)
                    avg_time.append((time.time() - start_time) * 1000)  # DEBUG
                    process_time += (time.time() - start_time) * 1000  # DEBUG
                    if orientation is not None:
                        position_value = binary_orientation(binary_square, orientation, kernel, side)
                    # if position_value is not None:
                    #     cv2.imwrite("data/pomoc_" + str(cos) + ".png", position_value)
                    #     cv2.imwrite("data/frame_" + str(cos) + ".png", frame_square)

        #         # print("position value:", position_value)
        #         try:
        #             if pos:
        #                 dis = cdist(np.array([position]), np.array([eval(positions[str(position_value)])]))
        #                 print("position value", position_value)
        #                 if dis[0][0] <= 2:
        #                     position = eval(positions[str(position_value)])
        #             else:
        #                 position = eval(positions[str(position_value)])
        #                 pos = True
        #                 if_side = True
        #                 print(position, position_value)
        #             threading.Thread(target=display_position(table, kernel, position), args=(1,)).start()  # DISPLAY
        #         except:
        #             print("Wrong position")
        #         else:
        #             pass
        #             # print("[-] Warning: square missing")
        #     else:
        #         pass
        #         # print("[-] Warning: rotated frame = None (probably error > 0.4 for each point)")
        # else:
        #     pass
        #     # print("[-] Warning: Side = None")

        # print((time.time() - start_time) * 1000, "miliseconds")  # use for process debugging
        # avg_time.append((time.time() - start_time) * 1000)
        # Wait a little (5 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(5)

        if process_time > 10000:
            break

    print("average time:", sum(avg_time) / len(avg_time), "miliseconds")
    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(3)

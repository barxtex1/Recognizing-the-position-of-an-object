import cv2
import numpy as np
import json
from math import sqrt, pi, atan
from scipy.spatial.distance import cdist
import time
from binary_orientation import binary_orientation
import threading
import socket
import pickle
import click


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


def calculate_side(image, thresh, kernel, help):
    # hsv_square = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_square = cv2.medianBlur(hsv_square, 5)
    # mask = cv2.inRange(hsv_square, np.array(thresh[0]), np.array(thresh[1]))
    # kernel_opening = np.ones((5, 5), np.uint8)
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)
    # cv2.imshow("opening", opening)

    # SIMPLE THRESHOLD
    # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(gray_frame, 5)
    # ret, mask = cv2.threshold(img, 152, 255, cv2.THRESH_BINARY)
    # cv2.imshow("opening", mask)
    # cv2.waitKey()

    # ADAPTIVE
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 73, 0)
    kernel_erosion = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(mask, kernel_erosion, iterations=1)

    edge_length = kernel * 3  # DEBUG

    # FIRST METHOD
    # contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for count, contour in enumerate(contours):
    #     epsilon = 0.1 * cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, epsilon, True)  # Calculate the position of the corners of the contour
    #     if len(np.squeeze(approx)) == 4:  # Take only with 4 corners
    #         # Positions of the corners
    #         x1, y1 = np.squeeze(approx)[0]
    #         x2, y2 = np.squeeze(approx)[1]
    #         x3, y3 = np.squeeze(approx)[2]
    #         x4, y4 = np.squeeze(approx)[3]
    #         # Length of each side
    #         side1 = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    #         side2 = sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
    #         side3 = sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)
    #         side4 = sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2)
    #         if abs(side1 - side2) < 2 and abs(side1 - side3) < 2 and abs(side1 - side4) < 2 and side1 > 20:
    #             print("side", side1)
    #             # Draw each contour only for visualisation purposes
    #             cv2.drawContours(image, contours, count, (0, 0, 255), 2)
    #             cv2.imshow("side", image)
    #             # cv2.waitKey()
    #             return side1 + edge_length

    # SECOND METHOD - wymagało użycia approxPolyDp bo nie było tylko 4 punktów
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
            if abs(side1 - side2) < 2 and abs(side1 - side3) < 2 and abs(side1 - side4) < 2 and side1 > 20:
                # print("side", side1)
                # Draw each contour only for visualisation purposes
                # cv2.drawContours(help, contours, count, (0, 0, 255), 2)
                # cv2.imshow("side", help)
                # cv2.waitKey()
                return side1 + edge_length


def rotate_table(image, cos, thresh):
    # FIRST METHOD
    width, height, _ = image.shape
    cX, cY = (height // 2, width // 2)  # center point of frame

    hsv_square = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_square = cv2.medianBlur(hsv_square, 5)
    mask = cv2.inRange(hsv_square, np.array(thresh[0]), np.array(thresh[1]))
    kernel_opening = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening)

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(gray_frame, 5)
    # ret, mask = cv2.threshold(img, 152, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("check/thresh_" + str(cos) + ".png", mask)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
            if abs(side1 - side2) < 2 and abs(side1 - side3) < 2 and abs(side1 - side4) < 2 and side1 > 20:
                rect = cv2.minAreaRect(contour)
                angle = rect[2]
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                rotated = cv2.warpAffine(image, M, (height, width))
                return rotated, opening
    return None, opening

    # SECOND METHOD
    # width, height, _ = frame.shape
    # cX, cY = (height // 2, width // 2)  # center point of frame
    # rect = cv2.fitEllipse(contour)
    # angle = rect[2]
    # M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # rotated = cv2.warpAffine(frame, M, (height, width))
    # return rotated

    # WCZEŚNIEJSZE ROZWIĄZANIE
    # width, height = gray.shape
    # cX, cY = (height // 2, width // 2)  # center point of frame
    #
    # harris_corners = cv2.cornerHarris(gray, 10, 1, 0.15)  # detect corners
    # corners = np.zeros_like(harris_corners)
    # corners[harris_corners > 0.025 * harris_corners.max()] = 1
    #
    # result = np.where(corners == 1)
    # corners_points = list(zip(result[0], result[1]))  # list of coord corners
    # cv2.imshow("corners", corners)  # DEBUG
    # cv2.waitKey()
    #
    # x, y = corners_points[len(corners_points) // 2]  # could be any point (I take middle one)  MAYBE NOT
    #
    # # correct_length = sqrt((166 - 57) ** 2 + (142 - 184) ** 2)  # calculated to test == 116.8
    # for pts in corners_points:
    #     x0, y0 = pts
    #     length_of_pts_to_corner = sqrt((x - x0) ** 2 + (y - y0) ** 2)
    #     error = abs(side - length_of_pts_to_corner)
    #     if error < 0.4:
    #         try:
    #             a = (y - y0) / (x - x0)  # slope of a straight line
    #             theta = atan(abs(x - x0) / abs(y - y0))  # angle of rotate (in radians)
    #             if a < 0:
    #                 M = cv2.getRotationMatrix2D((cX, cY), -theta * 180 / pi, 1.0)
    #             else:
    #                 M = cv2.getRotationMatrix2D((cX, cY), theta * 180 / pi, 1.0)
    #             rotated = cv2.warpAffine(gray, M, (width, height))
    #             # cv2.imshow("rotated", rotated)  # DEBUG
    #             return rotated
    #         except:
    #             return None  # For while


def cut_out_square(img, side, kernel_size):
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
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = np.float32(gray_frame)
    height, width = gray_frame.shape
    center_square = gray_frame[height // 2 - int(0.5 * kernel_size * side):height // 2 + int(0.5 * kernel_size * side),
                    width // 2 - int(0.5 * kernel_size * side):width // 2 + int(0.5 * kernel_size * side)]

    h_center_square, w_center_square = center_square.shape

    center_square = cv2.medianBlur(center_square, 5)
    harris_corners = cv2.cornerHarris(center_square, 5, 1, 0.06)  # detect corners
    # --------------------------------------------------------

    dst = cv2.dilate(harris_corners, None)
    ret, dst = cv2.threshold(dst, 0.025 * dst.max(), 255, cv2.THRESH_BINARY)

    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners2 = cv2.cornerSubPix(center_square, np.float32(centroids), (20, 20), (-1, -1), criteria)
    # Now draw them
    corners_sub = np.zeros_like(center_square)
    for w, h in corners2:
        corners_sub[int(h), int(w)] = 1
    # cv2.imshow("cornes_square_sub", corners_sub)

    # --------------------------------------------------------
    # WITHOUT SUBPIX
    # corners = np.zeros_like(harris_corners)
    # corners[harris_corners > 0.025 * harris_corners.max()] = 1

    # cv2.imshow("cornes_square", corners)
    # cv2.waitKey()

    result = np.where(corners_sub == 1)
    corners_points = list(zip(result[0], result[1]))  # list of coord corners
    corners_points = np.array(corners_points)  # convert list to numpy array
    distance = cdist(np.array([[0, 0]]), corners_points)
    idx = np.argmin(distance)
    min_distance = np.min(distance)

    for x in np.array([[[0, w_center_square]], [[h_center_square, 0]], [[h_center_square, w_center_square]]]):
        distance = cdist(x, corners_points)
        if min_distance > np.min(distance):
            min_distance = np.min(distance)
            idx = np.argmin(distance)

    h_min_d_square, w_min_d_square = corners_points[
        idx]  # Coordinate of the min. corner distance between one of the 4 corners of the window
    x_img, y_img = height // 2 - int(0.5 * kernel_size * side), width // 2 - int(0.5 * kernel_size * side)
    h_min_d, w_min_d = x_img + h_min_d_square, y_img + w_min_d_square
    if h_min_d_square < h_center_square // 2 and w_min_d_square < w_center_square // 2:
        square_ = img[h_min_d: h_min_d + int(kernel_size * side), w_min_d: w_min_d + int(kernel_size * side)]
    elif h_min_d_square < h_center_square // 2 and w_min_d_square > w_center_square // 2:
        square_ = img[h_min_d: h_min_d + int(kernel_size * side), w_min_d - int(kernel_size * side): w_min_d]
    elif h_min_d_square > h_center_square // 2 and w_min_d_square < w_center_square // 2:
        square_ = img[h_min_d - int(kernel_size * side): h_min_d, w_min_d: w_min_d + int(kernel_size * side)]
    else:
        square_ = img[h_min_d - int(kernel_size * side): h_min_d, w_min_d - int(kernel_size * side): w_min_d]

    # cv2.imshow("square_2", square_)  # DEBUG
    # cv2.imshow("corners", corners)
    return square_


def find_mask_of_triangle(hsv_square, thresh):
    mask = cv2.inRange(hsv_square, np.array(thresh[0]), np.array(thresh[1]))
    kernelOpen = np.ones((5, 5))
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    return maskOpen


def find_orientation(triangle):
    contours, _ = cv2.findContours(triangle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
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
# @click.option('-k', '--kernel', type=int, help='Size of kernel')
def main(kernel):
    # Upload the required files
    positions = json.loads(open("resources/positions_30_30.json").read())
    table = cv2.imread("resources/table_30_30.png")
    h, w, _ = table.shape
    table = cv2.resize(table, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    thresholds = json.loads(open("threshold_arrays.json").read())

    # STREAM CONFIGURATION
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10000000)
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
    cos = 0
    while key != ord('q'):
        cos += 1  # DEBUG
        start_time = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame = cv2.imread("frame.png")
        # cv2.imshow("frame", frame)
        # cv2.imwrite("check/frame_" + str(cos) + ".png", frame)
        # cv2.imwrite("frame_git.png", frame)
        if not if_side:
            side = calculate_side(frame, thresholds["side"], kernel, frame)

        if side is not None:  # Check if program calculate side of pixel
            rotated_frame, photo = rotate_table(frame, cos, thresholds["side"])
            ret, buffer = cv2.imencode(".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            x_as_bytes = pickle.dumps(buffer)
            s.sendto(x_as_bytes, (serverip, serverport))
            # if rotated_frame is not None:
            #     # cv2.imshow("rotated", rotated_frame)
            #     square = cut_out_square(rotated_frame, side, kernel)
            #     if square is not None:
            #         # square = cv2.imread("square.png", cv2.IMREAD_GRAYSCALE)  # DEBUG
            #         # cv2.imshow("Cut out square", square)
            #
            #         hsv_square = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
            #
            #         # cv2.waitKey()
            #         # cv2.imwrite("square.png", square)
            #         #             bgr_square = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)
            #         #             hsv_square = cv2.cvtColor(bgr_square, cv2.COLOR_BGR2HSV)
            #         #             cv2.imwrite("triangle.png", hsv_square)
            #         triangle = find_mask_of_triangle(hsv_square, thresholds["triangle"])
            #         # cv2.imshow("triangle", triangle)
            #         # cv2.waitKey()
            #         #
            #         orientation = find_orientation(triangle)
            #         # print("Current Orientation: ", orientation)
            #         position_value = binary_orientation(square, orientation, kernel, thresholds["side"])
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

        print((time.time() - start_time) * 1000, "miliseconds")  # use for process debugging
        avg_time.append((time.time() - start_time) * 1000)
        # Wait a little (5 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(5)

    print("average time:", sum(avg_time) / len(avg_time), "miliseconds")
    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(3)

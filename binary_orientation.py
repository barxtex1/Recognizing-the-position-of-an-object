import numpy as np
import cv2


def kernel_up(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                index = j+kernel*i
                number = number[:index] + str(1) + number[index+1:]
    return number


def kernel_down(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                index = 8 - (j + kernel*i)
                number = number[:index] + str(1) + number[index+1:]
    return number


def kernel_left(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                index = j*3 + (2 - i)
                number = number[:index] + str(1) + number[index+1:]
    return number


def kernel_right(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                index = (6 + i) - j*3
                number = number[:index] + str(1) + number[index+1:]
    return number


def find_where_is_one(coord, number, orientation, shape, kernel):
    side = shape[0] / kernel
    if orientation == "up":
        for x, y in coord:
            number = kernel_up(side, number, x, y, kernel)
        return number
    elif orientation == "down":
        for x, y in coord:
            number = kernel_down(side, number, x, y, kernel)
        return number
    elif orientation == "right":
        for x, y in coord:
            number = kernel_right(side, number, x, y, kernel)
        return number
    else:
        for x, y in coord:
            number = kernel_left(side, number, x, y, kernel)
        return number


def binary_orientation(binary_frame, orientation, kernel, side):
    kernel_erosion = np.ones((13, 13), np.uint8)
    erosion = cv2.erode(binary_frame, kernel_erosion, iterations=1)
    binary_value = "".zfill(kernel ** 2)
    coord_numbers = []
    contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # pomoc = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
    for count, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < (side**2)*0.4:
            continue
        # Draw each contour only for visualisation purposes
        # cv2.drawContours(pomoc, contours, count, (0, 0, 255), 2)

        x, y, w, h = cv2.boundingRect(contour)
        coord_numbers.append((x + w // 2, y + h // 2))
    binary_value = find_where_is_one(coord_numbers, binary_value, orientation, erosion.shape, kernel)
    return int(binary_value, 2)  # decimal value of position

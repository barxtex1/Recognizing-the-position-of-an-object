import numpy as np
import cv2


def kernel_up(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                number[j+kernel*i] = 1
    return number


def kernel_down(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                number[8 - (j + kernel*i)] = 1
    return number


def kernel_left(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                number[j*3 + (2 - i)] = 1
    return number


def kernel_right(side, number, x, y, kernel):
    for i in range(kernel):
        for j in range(kernel):
            if j*side < x < (j+1)*side and i*side < y < (i+1)*side:
                number[(6 + i) - j*3] = 1
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


def binary_orientation(hsv, orientation, kernel, image):
    mask = cv2.inRange(hsv, np.array([0, 0, 142]), np.array([180, 256, 256]))
    kernel_erosion = np.ones((10, 10))
    erosion = cv2.erode(mask, kernel_erosion, iterations=1)
    cv2.imshow("dilation", erosion)
    binary_value = np.zeros(kernel ** 2)
    coord_numbers = []
    contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for count, contour in enumerate(contours):
        # Draw each contour only for visualisation purposes
        # cv2.drawContours(image, contours, count, (0, 0, 255), 2)
        # cv2.imshow("image", image)
        # cv2.waitKey()

        x, y, w, h = cv2.boundingRect(contour)
        coord_numbers.append((x + w // 2, y + h // 2))
    binary_value = find_where_is_one(coord_numbers, binary_value, orientation, mask.shape, kernel)
    return int(sum([j * (2 ** i) for i, j in list(enumerate(reversed(binary_value)))]))  # decimal value of position

import numpy as np
import cv2


def triangle(square: np.ndarray, side: int):
    side_of_triangle = (side - 2) // 2
    for i in range(side_of_triangle):
        square[i, side_of_triangle + i:] = np.array([0, 0, 255])
    return square


matrix = np.load('resources/matrix_150_150.npy')
board_len = 30
matrix = matrix[:board_len-1, :board_len-1]
window_size = 3000
height, width = matrix.shape
side_of_square = window_size//board_len
table = np.zeros((window_size, window_size, 3))
print(table.shape)
for h, line in enumerate(matrix):
    for w, value in enumerate(line):
        if value:
            table_of_ones = np.ones((side_of_square, side_of_square, 3))
            for count, line in enumerate(table_of_ones):
                if count == 0:
                    table_of_ones[count, :] = np.array([0, 0, 0])
                elif count == side_of_square - 1:
                    table_of_ones[count, :] = np.array([0, 0, 0])
                else:
                    table_of_ones[count, 0] = np.array([0, 0, 0])
                    table_of_ones[count, -1] = np.array([0, 0, 0])
            if h % 3 == 0 and w % 3 == 0:
                table_of_ones[1:-1, 1:-1] = triangle(table_of_ones[1:-1, 1:-1], side_of_square)
            table[h * side_of_square:h * side_of_square + side_of_square,
            w * side_of_square:w * side_of_square + side_of_square] = table_of_ones
        else:
            table_of_zeros = np.zeros((side_of_square, side_of_square, 3))
            for count, line in enumerate(table_of_zeros):
                if count == 0:
                    table_of_zeros[count, :] = np.array([255, 255, 255])
                elif count == side_of_square - 1:
                    table_of_zeros[count, :] = np.array([255, 255, 255])
                else:
                    table_of_zeros[count, 0] = np.array([255, 255, 255])
                    table_of_zeros[count, -1] = np.array([255, 255, 255])
            if h % 3 == 0 and w % 3 == 0:
                table_of_zeros[1:-1, 1:-1] = triangle(table_of_zeros[1:-1, 1:-1], side_of_square)
            table[h * side_of_square:h * side_of_square + side_of_square,
            w * side_of_square:w * side_of_square + side_of_square] = table_of_zeros

print(table.shape)
table = cv2.convertScaleAbs(table, alpha=255.0)
cv2.imwrite("resources/table_"+str(board_len)+"_"+str(board_len)+".png", table)

import numpy as np
import json

table = np.load('resources/matrix_150_150.npy')
board_len = 30
table = table[:board_len-1, :board_len-1]
height, width = table.shape
kernel = 3
possible_position = {}
for h in range(height - kernel + 1):
    for w in range(width - kernel + 1):
        mask = table[h:h + kernel, w:w + kernel]
        binary_number = ""
        for line in mask:
            for i in line:
                binary_number += str(i)
        possible_position[int(binary_number, 2)] = str(h)+","+str(w)
        # print("binary:", binary_number, "decimal: ", int(binary_number, base=2))

print("Count of possible position: ", len(possible_position))
# print(possible_position)
if len(possible_position) != len(set(possible_position)):
    print("duplicates found in the list")
    duplicates = {x for x in possible_position if possible_position.count(x) > 1}
    print("count of duplicates:", len(duplicates))
    print("duplicates:", duplicates)
else:
    print("No duplicates found in the list")
    # create json object from dictionary
    json = json.dumps(possible_position)
    # open file for writing, "w"
    f = open("resources/positions_30_30.json", "w")
    # write json object to file
    f.write(json)
    # close file
    f.close()

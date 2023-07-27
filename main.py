import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time

board_size = 8

board_color = np.zeros([board_size, board_size])

for row in range(board_size):
    for col in range(board_size):
        board_color[row, col] = np.mod(row + col, 2)

plt.imshow(board_color,cmap='Greys')
dot = plt.plot([3], [4], 'ro')

plt.pause(1)

for rep in range(5):
    new_coords = input("Enter coordinates:")
    new_coords = np.fromstring(new_coords, dtype=int, sep=',')
    dot[0].set_xdata([new_coords[0]])
    dot[0].set_ydata([new_coords[1]])
    plt.pause(1)

new_coords = input("Enter coordinates:")


while False:

    dot[0].set_xdata(new_coords[0])
    dot[0].set_ydata(new_coords[1])


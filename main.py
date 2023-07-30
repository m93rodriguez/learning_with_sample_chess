import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
from Piece import Pawn, Rook, Knight, Bishop, Queen, King, Team

board_size = 8

board_color = np.zeros([board_size, board_size])

for row in range(board_size):
    for col in range(board_size):
        board_color[row, col] = np.mod(row + col, 2)

fig, ax = plt.subplots()

plt.imshow(board_color, cmap='Greys')

player = [Team(0, 1, 1), Team(7, 6, -1)]

plt.ioff()


def onclick(event):
    position = [np.round(event.xdata), np.round(event.ydata)]
    player[0].move_piece(0, position, player[1])
    plt.draw()
    print(player[1].piece_positions.size)


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()










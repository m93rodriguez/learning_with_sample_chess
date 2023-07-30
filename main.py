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

plt.imshow(board_color, cmap='Greys')


player = [Team(0, 1, 1), Team(7, 6, -1)]

player[0].Pieces['Knight'][0].move([3, 5])

test = player[0].Pieces['Knight'][0].movement_range(player[0].list_pieces()[0], player[1].list_pieces()[0])
print(test)

plt.plot(test[:,0], test[:,1],'xr')
plt.show()







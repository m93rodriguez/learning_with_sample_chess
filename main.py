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

allowed_moves_visual = plt.plot([], [], 'yo')

player = [Team(0, 1, 1), Team(7, 6, -1)]

plt.ioff()

player[1].move_piece(0, [3, 2], player[0])

turn = 0
turn_starts = True
piece_id = None
possible_moves = None


def onclick(event):
    global turn, turn_starts, piece_id, possible_moves

    def select_own_piece():

        _piece_id = (player[turn].piece_positions == position).all(1)
        if not any(_piece_id):
            return None, None
        _piece_id = np.argwhere(_piece_id)[0, 0]

        current_piece = player[turn].Pieces[player[turn].piece_list[_piece_id][0]]
        current_piece = current_piece[player[turn].piece_list[_piece_id][1]]
        _possible_moves = current_piece.movement_range(player[turn].piece_positions,
                                                      player[(turn+1) % 2].piece_positions)

        if _possible_moves.size < 2:
            allowed_moves_visual[0].set_xdata([])
            allowed_moves_visual[0].set_ydata([])
            return _piece_id, None
        else:
            allowed_moves_visual[0].set_xdata(_possible_moves[:, 0])
            allowed_moves_visual[0].set_ydata(_possible_moves[:, 1])
            
        return _piece_id, _possible_moves

    def select_destination():
        if any((position == player[turn].piece_positions).all(1)):
            return select_own_piece()
        if any((position == possible_moves).all(1)):
            player[turn].move_piece(piece_id, position, player[(turn+1) % 2])
            return None, None
        else:
            return piece_id, possible_moves

    position = [np.round(event.xdata), np.round(event.ydata)]

    if turn_starts:
        piece_id, possible_moves = select_own_piece()
        if possible_moves is not None:
            turn_starts = False
    else:
        piece_id, possible_moves = select_destination()
        if piece_id is None:
            turn = (turn + 1) % 2
            turn_starts = True
            allowed_moves_visual[0].set_xdata([])
            allowed_moves_visual[0].set_ydata([])

    # player[turn].move_piece(0, position, player[1])
    plt.draw()
    # print(player[1].piece_positions.size)


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()










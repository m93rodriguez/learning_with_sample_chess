import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from Piece import Pawn, Rook, Knight, Bishop, Queen, King, Team

board_size = 8

board_color = np.zeros([board_size, board_size])

for row in range(board_size):
    for col in range(board_size):
        board_color[row, col] = np.mod(row + col, 2)

fig, ax = plt.subplots()

plt.imshow(board_color, cmap='Greys', extent=(-.5, 7.5, -.5, 7.5), alpha=0.5)

player = [Team(0), Team(1)]

#player[1].move_piece(0, [3, 2], player[0])

turn = 0
turn_starts = True
piece_id = None
possible_moves = None

allowed_moves_visual = plt.plot([], [], 'yo')

ac = player[0].attacked_cases(player[1])


ax = plt.gca()
ax.set_xlim([-0.5, 7.5])
ax.set_ylim([-0.5, 7.5])


def onclick(event):
    global turn, turn_starts, piece_id, possible_moves

    def select_own_piece():

        _piece_id = (player[turn].piece_positions == position).all(1)
        if not any(_piece_id):
            return None, None  # Returns both None if enemy piece or empty case is selected
        _piece_id = np.argwhere(_piece_id)[0, 0]

        current_piece = player[turn].Pieces[player[turn].piece_list[_piece_id][0]]
        current_piece = current_piece[player[turn].piece_list[_piece_id][1]]
        _possible_moves = current_piece.movement_range(player[turn], player[(turn+1) % 2])

        if _possible_moves.size < 2:
            allowed_moves_visual[0].set_xdata([])
            allowed_moves_visual[0].set_ydata([])
            return _piece_id, None  # Returns one number and one None if no movable places are possible
        else:
            allowed_moves_visual[0].set_xdata(_possible_moves[:, 0])
            allowed_moves_visual[0].set_ydata(_possible_moves[:, 1])
            
        return _piece_id, _possible_moves  # Returns chosen piece and possible moves if both exist

    def select_destination():
        if any((position == player[turn].piece_positions).all(1)):
            return select_own_piece()  # If an allied piece is selected, behave as if turn is starting
        if any((position == possible_moves).all(1)):
            player[turn].move_piece(piece_id, position, player[(turn+1) % 2])
            return None, None  # If movable space is selected, move the piece and "clean" the output variables
        else:
            return piece_id, possible_moves  # If no suitable move is chosen, nothing changes and wait for next call

    position = [np.round(event.xdata), np.round(event.ydata)]

    if turn_starts:
        piece_id, possible_moves = select_own_piece()
        if possible_moves is not None:
            turn_starts = False
    else:
        piece_id, possible_moves = select_destination()
        if piece_id is None:  # This happens if select_destination() succeed in finding movable case
            turn = (turn + 1) % 2
            turn_starts = True
            allowed_moves_visual[0].set_xdata([])
            allowed_moves_visual[0].set_ydata([])

    # player[turn].move_piece(0, position, player[1])
    plt.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

print(ac)

allowed_moves_visual[0].set_xdata(ac[:, 0])
allowed_moves_visual[0].set_ydata(ac[:, 1])

plt.show()










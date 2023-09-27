import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Piece:
    type = None

    def __init__(self, coords, team):
        self.position = coords
        self.team = team
        self.has_moved = False
        self.can_en_passant = False
        self.visual = self.plot()

    def plot(self):
        if self.type is None:
            return
        color_list = ['White', 'Black']
        icon_file = 'icons/' + self.type + '_' + color_list[self.team] + '.png'
        icon_img = mpimg.imread(icon_file)
        img_pos = (-0.5+self.position[0], 0.5+self.position[0], -0.5+self.position[1], 0.5+self.position[1])
        return plt.imshow(icon_img, extent=img_pos, alpha=1)

    def move(self, new_position):
        self.position = new_position
        self.has_moved = True
        img_pos = (-0.5 + new_position[0], 0.5 + new_position[0], -0.5 + new_position[1], 0.5 + new_position[1])
        self.visual.set_extent(img_pos)

    def gets_eaten(self):
        self.visual.remove()
        del self

    @staticmethod
    def moves_inside_board(moves):
        board_limits = [0, 7]
        ind = np.logical_and(moves <= board_limits[1], moves >= board_limits[0]).all(1)
        return moves[ind, :]

    @staticmethod
    def is_inside_board(position):
        board_limits = [0, 7]
        position = np.reshape(position, (1, position.size))
        return np.logical_and(position <= board_limits[1], position >= board_limits[0]).all(1)

    @staticmethod
    def moves_are_attacked(moves, attacked_cases):
        if moves.size < 2:
            return None
        is_attacked = []
        for move in moves:
            is_attacked.append(all(np.all(move == attacked_cases, axis=1)))
        return np.array(is_attacked)


class Pawn(Piece):
    def __init__(self, coords, team, y_step):
        self.type = 'Pawn'
        self.y_step = y_step
        Piece.__init__(self, coords, team)

    def attack_cases(self, allies, foes, **kwargs):
        reachable = np.empty((0, 2))
        for direction in [-1, 1]:
            test_pos = self.position + np.array([direction, self.y_step])
            reachable = np.append(reachable, [test_pos], axis=0)
        return self.moves_inside_board(reachable)

    def attack_en_passant(self, ally, foe):
        reachable = np.empty((0, 2))
        self.can_en_passant = False
        if foe.prev_turn_jump is None:
            return reachable
        attack_row = 7 - ally.back_row - 3*self.y_step
        if not self.position[1] == attack_row:
            return reachable
        if np.abs(foe.prev_turn_jump[0] - self.position[0]) < 2:
            reachable = np.copy(foe.prev_turn_jump)
            reachable[1] = reachable[1] + self.y_step
            self.can_en_passant = True
        return np.array([reachable])

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        reachable = np.empty((0, 2))

        for mag in [1, 2]:
            forward_move = self.position + np.array([0, mag*self.y_step])

            if ((forward_move == ally_pos).all(1).any() or (forward_move == foe_pos).all(1).any() or
                    not self.is_inside_board(forward_move)):
                break
            reachable = np.append(reachable, [forward_move], axis=0)
            if self.has_moved:
                break
        side_move = self.attack_cases(ally_pos, foe_pos)
        side_move = side_move[is_member_rows(side_move, foe_pos), :]
        side_move = np.append(side_move, self.attack_en_passant(ally, foe), axis=0)

        if reachable.size == 0:
            return side_move

        if side_move.size == 0:
            return reachable
        
        return np.append(reachable, side_move, axis=0)

    def move(self, new_position):
        old_position = self.position
        super().move(new_position)
        if np.abs(old_position[1] - new_position[1]) > 1:
            return 'Pawn_Jump'
        if np.abs(old_position[0] - new_position[0]) == 1 and self.can_en_passant:
            return 'Passant_Possible'


class Rook(Piece):
    def __init__(self, coords, team):
        self.type = 'Rook'
        Piece.__init__(self, coords, team)

    @classmethod
    def attack_cases(cls, position, allies, foes, max_movement=7, **kwargs):
        reachable = np.empty((0, 2))
        for direction in [[0, 1], [1, 0]]:
            for sense in [1, -1]:
                for mag in range(1, max_movement+1):
                    test_pos = position + np.array([direction])*sense*mag
                    if not cls.is_inside_board(test_pos):
                        break
                    reachable = np.append(reachable, test_pos, axis=0)
                    if (test_pos == allies).all(1).any():
                        break
                    if (test_pos == foes).all(1).any():
                        break

        return np.array(reachable)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = ally.piece_positions
        reachable = self.attack_cases(self.position, ally_pos, foe_pos)
        return reachable[~is_member_rows(reachable, ally_pos), :]


class Knight(Piece):
    def __init__(self, coords, team):
        self.type = 'Knight'
        Piece.__init__(self, coords, team)

    def attack_cases(self, allies, foes, **kwargs):
        reachable = []
        for x_mag in [1, 2]:
            y_mag = 3 - x_mag
            for x_dir in [-1, 1]:
                for y_dir in [-1, 1]:
                    test_pos = self.position + np.array([x_mag*x_dir, y_mag*y_dir])
                    if not self.is_inside_board(test_pos):
                        continue
                    reachable.append(test_pos)

        return np.array(reachable)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        reachable = self.attack_cases(ally_pos, foe_pos)
        return reachable[~is_member_rows(reachable, ally_pos), :]


class Bishop(Piece):
    def __init__(self, coords, team):
        self.type = 'Bishop'
        Piece.__init__(self, coords, team)

    @classmethod
    def attack_cases(cls, position, allies, foes, max_movement=7, **kwargs):
        reachable = np.empty((0, 2))
        for x_dir in [-1, 1]:
            for y_dir in [-1, 1]:
                for mag in range(1, max_movement+1):
                    test_pos = position + np.array([[x_dir, y_dir]]) * mag
                    if not cls.is_inside_board(test_pos):
                        break

                    reachable = np.append(reachable, test_pos, axis=0)

                    if (test_pos == allies).all(1).any():
                        break
                    if (test_pos == foes).all(1).any():
                        break

        return np.array(reachable)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        reachable = self.attack_cases(self.position, ally_pos, foe_pos)
        return reachable[~is_member_rows(reachable, ally_pos), :]


class Queen(Piece):
    def __init__(self, coords, team):
        self.type = 'Queen'
        Piece.__init__(self, coords, team)

    def attack_cases(self, allies, foes, **kwargs):
        reachable_rook = Rook.attack_cases(self.position, allies, foes)
        reachable_bishop = Bishop.attack_cases(self.position, allies, foes)
        return np.append(reachable_rook, reachable_bishop, axis=0)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        reachable = self.attack_cases(ally_pos, foe_pos)
        return reachable[~is_member_rows(reachable, ally_pos), :]


class King(Piece):
    def __init__(self, coords, team):
        self.type = 'King'
        self.in_check = False
        Piece.__init__(self, coords, team)

    def is_in_check(self, ally, foe):
        if any(np.all(self.position == foe.attacked_cases(ally), axis=1)):
            return True
        return False

    def castling_moves(self, ally, foe):
        if self.has_moved:
            return np.empty((0, 2))

        if self.is_in_check(ally, foe):
            return np.empty((0, 2))

        attacked = foe.attacked_cases(ally)
        reachable = np.empty((0, 2))
        for rook in ally.Pieces['Rook']:
            if rook.has_moved:
                continue
            if not self.position[1] == rook.position[1]:
                continue
            possible = True
            start = np.minimum(self.position[0], rook.position[0]) + 1
            stop = np.maximum(self.position[0], rook.position[0])
            for x in range(start, stop):
                case = np.array([[x, self.position[1]]])
                if (any(np.all(case == attacked, axis=1)) or any(np.all(case == ally.piece_positions, axis=1)) or
                        any(np.all(case == foe.piece_positions, axis=1))):
                    possible = False
                    break
            if possible:
                direction = np.sign(rook.position[0] - self.position[0])
                reachable = np.append(reachable, np.array([[self.position[0] + 2*direction, self.position[1]]]), axis=0)

        return reachable

    def attack_cases(self, allies, foes, **kwargs):
        reachable_rook = Rook.attack_cases(self.position, allies, foes, max_movement=1)
        reachable_bishop = Bishop.attack_cases(self.position, allies, foes, max_movement=1)
        return np.append(reachable_rook, reachable_bishop, axis=0)

    def movement_range(self, ally, foe, **kwargs):
        """ Provides the move the current Piece can make in a turn.

            Inputs:
                Ally: (Team)
                Foe: (Team)
            kwargs:
                castling (default: True): if True, calculates the possible moves due to castling with rooks"""

        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions

        reachable = self.attack_cases(ally_pos, foe_pos)

        if reachable.size > 1:
            reachable = reachable[~is_member_rows(reachable, ally_pos), :]
            reachable = reachable[~is_member_rows(reachable, foe.attacked_cases(ally)), :]

        return np.append(reachable, self.castling_moves(ally, foe), axis=0)

    def move(self, new_position):
        old_position = self.position
        super().move(new_position)
        if np.abs(old_position[0] - new_position[0]) > 1:
            return 'Castled'


class Team:
    def __init__(self, team_index, phantom_copy=False):
        back_row_list = [7, 0]
        self.back_row = back_row_list[team_index]
        self.prev_turn_jump = None
        self.index = team_index
        self.Pieces = None
        self.piece_positions = []
        self.piece_list = []
        if not phantom_copy:
            self.initialize_pieces()

    def initialize_pieces(self):
        step_list = [-1, 1]
        self.Pieces = {
            'Pawn': [Pawn(coords=np.array([cnt, self.back_row + step_list[self.index]]),
                          team=self.index, y_step=step_list[self.index]) for cnt in range(8)],
            'Rook': [Rook(coords=np.array([cnt, self.back_row]), team=self.index) for cnt in [0, 7]],
            'Knight': [Knight(coords=np.array([cnt, self.back_row]), team=self.index) for cnt in [1, 6]],
            'Bishop': [Bishop(coords=np.array([cnt, self.back_row]), team=self.index) for cnt in [2, 5]],
            'Queen': [Queen(coords=np.array([cnt, self.back_row]), team=self.index) for cnt in [4]],
            'King': [King(coords=np.array([cnt, self.back_row]), team=self.index) for cnt in [3]],
        }
        (self.piece_positions, self.piece_list) = self.list_pieces()

    def make_copy(self):
        """Creates a shallow copy object, with the back_row, prev_turn_jump, piece_list,
        and piece_positions properties"""

        phantom = Team(self.index, phantom_copy=True)
        phantom.back_row = self.back_row
        phantom.prev_turn_jump = self.prev_turn_jump
        phantom.piece_list = self.piece_list.copy()
        phantom.piece_positions = np.copy(self.piece_positions)
        phantom.Pieces = self.Pieces
        return phantom

    def attacked_cases(self, foe):
        team_moves = np.empty((0, 2))
        for piece_iter in self.piece_list:
            piece = self.Pieces[piece_iter[0]][piece_iter[1]]
            piece_moves = piece.attack_cases(allies=self.piece_positions, foes=foe.piece_positions,
                                             position=piece.position)
            if piece_moves.size > 1:
                team_moves = np.append(team_moves, piece_moves, axis=0)
        return team_moves

    def list_pieces(self):

        piece_positions = []
        piece_id = []
        for piece_type in self.Pieces.keys():
            current_piece = 0
            for piece in self.Pieces[piece_type]:
                piece_positions.append(piece.position)
                piece_id.append([piece.type, current_piece])
                current_piece = current_piece + 1
        return np.array(piece_positions), piece_id

    def find_piece(self, position):
        if self.piece_positions.size > 0:
            selected = np.all(self.piece_positions == position, axis=1)
            if any(selected):
                selected = np.nonzero(selected)
                return selected[0][0]  # Output of nonzero is tuple for each dimension

    def identify_lost_piece(self, foe_piece, hungry_position):
        edible_piece = self.find_piece(hungry_position)
        if edible_piece is None and foe_piece.can_en_passant:
            en_passant_position = np.copy(hungry_position)
            en_passant_position[1] = en_passant_position[1] - foe_piece.y_step
            edible_piece = self.find_piece(en_passant_position)
        return edible_piece

    def move_piece(self, piece_ind, position, foe):
        self.prev_turn_jump = None
        piece = self.Pieces[self.piece_list[piece_ind][0]][self.piece_list[piece_ind][1]]

        edible_piece = foe.identify_lost_piece(piece, position)
        foe.eat_piece(edible_piece)

        move_state = piece.move(position)

        if move_state == 'Castled':
            dist = np.inf
            rook_id = 0
            cont = 0
            for rook in self.Pieces['Rook']:
                if dist > np.abs(piece.position[0] - rook.position[0]):
                    rook_id = cont
                    dist = np.abs(piece.position[0] - rook.position[0])
                cont = cont + 1
            rook = self.Pieces['Rook'][rook_id]
            direction = np.sign(piece.position[0] - rook.position[0])
            rook.move([piece.position[0]+direction, piece.position[1]])

        if move_state == 'Pawn_Jump':
            self.prev_turn_jump = position

        self.piece_positions = self.list_pieces()[0]

    def eat_piece(self, piece_ind):
        if piece_ind is None:
            return
        self.Pieces[self.piece_list[piece_ind][0]][self.piece_list[piece_ind][1]].gets_eaten()
        del self.Pieces[self.piece_list[piece_ind][0]][self.piece_list[piece_ind][1]]
        self.piece_positions, self.piece_list = self.list_pieces()

    def test_move(self, piece_ind, target_pos, foe):
        phantom = self.make_copy()
        foe_phantom = foe.make_copy()
        piece = self.Pieces[self.piece_list[piece_ind][0]][self.piece_list[piece_ind][1]]
        edible_piece = foe.identify_lost_piece(piece, target_pos)

        phantom.piece_positions[piece_ind] = target_pos
        if edible_piece is not None:
            foe_phantom.piece_positions = np.delete(foe_phantom.piece_positions, edible_piece, axis=0)
            del foe_phantom.piece_list[edible_piece]

        if self.Pieces['King'][0].is_in_check(phantom, foe_phantom):
            return False
        return True

    def check_for_check_mate(self, foe):
        piece_id = 0
        for piece_list in self.piece_list:
            piece = self.Pieces[piece_list[0]][piece_list[1]]
            move_set = piece.movement_range(self, foe)
            for move in move_set:
                if self.test_move(piece_id, move, foe):
                    return False
            piece_id += 1
        return True


class Match:
    board_size = 8

    def __init__(self):
        self.player = None
        self.turn = 0
        self.turn_starts = True
        self.selected_piece = None
        self.possible_moves = None
        self.allowed_moves_visual = None
        self.target = None

    def initialize(self):
        fig, ax = plt.subplots()
        self.plot_board()
        self.player = [Team(0), Team(1)]
        self.allowed_moves_visual = plt.plot([], [], 'yo')
        self.target = self.initialize_target(self.player[0].Pieces['King'][0])
        self.target.set_visible(False)
        ax.set_xlim([-0.5, 7.5])
        ax.set_ylim([-0.5, 7.5])
        return fig

    def plot_board(self):
        board_color = np.zeros([self.board_size, self.board_size])
        for row in range(self.board_size):
            for col in range(self.board_size):
                board_color[row, col] = np.mod(row + col, 2)
        plt.imshow(board_color, cmap='Greys', extent=(-.5, 7.5, -.5, 7.5), alpha=0.5)

    @staticmethod
    def initialize_target(king):
        icon_file = 'icons/Target.png'
        icon_img = mpimg.imread(icon_file)
        img_pos = (-0.45 + king.position[0], 0.45 + king.position[0], -0.45 + king.position[1], 0.45 + king.position[1])
        return plt.imshow(icon_img, extent=img_pos, alpha=0.9)

    def select_own_piece(self, position):
        _piece_id = (self.player[self.turn].piece_positions == position).all(1)
        if not any(_piece_id):
            return None, None  # Returns both None if enemy piece or empty case is selected
        _piece_id = np.argwhere(_piece_id)[0, 0]

        current_piece = self.player[self.turn].Pieces[self.player[self.turn].piece_list[_piece_id][0]]
        current_piece = current_piece[self.player[self.turn].piece_list[_piece_id][1]]
        _possible_moves = current_piece.movement_range(self.player[self.turn], self.player[(self.turn + 1) % 2])

        if _possible_moves.size < 2:
            self.allowed_moves_visual[0].set_xdata([])
            self.allowed_moves_visual[0].set_ydata([])
            return _piece_id, None  # Returns one number and one None if no movable places are possible
        else:
            self.allowed_moves_visual[0].set_xdata(_possible_moves[:, 0])
            self.allowed_moves_visual[0].set_ydata(_possible_moves[:, 1])

        return _piece_id, _possible_moves  # Returns chosen piece and possible moves if both exist

    def select_destination(self, position):
        if any((position == self.player[self.turn].piece_positions).all(1)):
            return self.select_own_piece(position)  # If an allied piece is selected, behave as if turn is starting
        if any((position == self.possible_moves).all(1)):
            if self.player[self.turn].test_move(self.selected_piece, position, self.player[(self.turn + 1) % 2]):
                self.player[self.turn].move_piece(self.selected_piece, position, self.player[(self.turn+1) % 2])
                return None, None  # If movable space is selected, move the piece and "clean" the output variables
        return self.selected_piece, self.possible_moves
        # If no suitable move is chosen, nothing changes and wait for next call

    def onclick(self, event):
        position = [np.round(event.xdata), np.round(event.ydata)]
        if self.turn_starts:

            if self.player[self.turn].Pieces['King'][0].is_in_check(self.player[self.turn],
                                                                    self.player[(self.turn + 1) % 2]):
                if self.player[self.turn].check_for_check_mate(self.player[(self.turn + 1) % 2]):
                    self.player[self.turn].Pieces['King'][0].visual.set_visible(False)
                    plt.draw()
                    return None

            self.selected_piece, self.possible_moves = self.select_own_piece(position)
            if self.possible_moves is not None:
                self.turn_starts = False
        else:
            self.selected_piece, self.possible_moves = self.select_destination(position)
            if self.selected_piece is None:  # This happens if select_destination() succeed in finding movable case
                self.turn = (self.turn + 1) % 2
                self.turn_starts = True
                self.allowed_moves_visual[0].set_xdata([])
                self.allowed_moves_visual[0].set_ydata([])

                if self.player[self.turn].Pieces['King'][0].is_in_check(self.player[self.turn],
                                                                        self.player[(self.turn + 1) % 2]):
                    self.target.set_extent(self.player[self.turn].Pieces['King'][0].visual.get_extent())
                    self.target.set_visible(True)
                else:
                    self.target.set_visible(False)

        # player[turn].move_piece(0, position, player[1])
        plt.draw()


def is_member_rows(a, b):
    """Determine if the rows of an array are in another array.

    Inputs:
        a: numpy.array of dimension 2
        b: numpy.array of dimension 2
    Output: numpy.array of dimension 1 with bools, with the same size as the first dimension of a.
            True indicates if the corresponding row is contained in b."""

    if a.size < 2:
        return None
    output = []
    for row in a:
        output.append(any(np.all(row == b, axis=1)))
    return np.array(output)




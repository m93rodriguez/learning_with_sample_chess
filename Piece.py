import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


class Piece:
    type = 'Pawn'

    def __init__(self, coords, team):
        self.position = coords
        self.team = team
        self.has_moved = False
        self.visual = self.plot()

    def plot(self):
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


class Pawn(Piece):
    def __init__(self, coords, team, y_step):
        self.type = 'Pawn'
        self.y_step = y_step
        Piece.__init__(self, coords, team)

    def attack_cases(self, allies, foes):
        reachable = np.empty((0, 2))
        for direction in [-1, 1]:
            test_pos = self.position + np.array([direction, self.y_step])
            if (test_pos == foes).all(1).any():
                reachable = np.append(reachable, [test_pos], axis=0)
        return reachable

    def attack_en_passant(self, ally, foe):
        reachable = np.empty((0, 2))
        if foe.prev_turn_jump is None:
            return reachable
        attack_row = 7 - ally.back_row - 3*self.y_step
        if not self.position[1] == attack_row:
            return reachable
        if np.abs(foe.prev_turn_jump[0] - self.position[0]) < 2:
            reachable = foe.prev_turn_jump
            reachable[1] = reachable[1] + self.y_step
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


class Rook(Piece):
    def __init__(self, coords, team):
        self.type = 'Rook'
        Piece.__init__(self, coords, team)

    @classmethod
    def attack_cases(cls, position, allies, foes, max_movement=7):
        reachable = np.empty((0, 2))
        for direction in [[0, 1], [1, 0]]:
            for sense in [1, -1]:
                for mag in range(1, max_movement+1):
                    test_pos = position + np.array([direction])*sense*mag
                    if not cls.is_inside_board(test_pos):
                        break
                    if (test_pos == allies).all(1).any():
                        break
                    reachable = np.append(reachable, test_pos, axis=0)
                    if (test_pos == foes).all(1).any():
                        break

        return np.array(reachable)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = ally.piece_positions
        return self.attack_cases(self.position, ally_pos, foe_pos)


class Knight(Piece):
    def __init__(self, coords, team):
        self.type = 'Knight'
        Piece.__init__(self, coords, team)

    def attack_cases(self, allies, foes):
        reachable = []
        for x_mag in [1, 2]:
            y_mag = 3 - x_mag
            for x_dir in [-1, 1]:
                for y_dir in [-1, 1]:
                    test_pos = self.position + np.array([x_mag*x_dir, y_mag*y_dir])
                    if not self.is_inside_board(test_pos):
                        continue
                    if (test_pos == allies).all(1).any():
                        continue
                    reachable.append(test_pos)

        return np.array(reachable)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        return self.attack_cases(ally_pos, foe_pos)


class Bishop(Piece):
    def __init__(self, coords, team):
        self.type = 'Bishop'
        Piece.__init__(self, coords, team)

    @classmethod
    def attack_cases(cls, position, allies, foes, max_movement=7):
        reachable = np.empty((0, 2))
        for x_dir in [-1, 1]:
            for y_dir in [-1, 1]:
                for mag in range(1, max_movement+1):
                    test_pos = position + np.array([[x_dir, y_dir]]) * mag
                    if not cls.is_inside_board(test_pos):
                        break
                    if (test_pos == allies).all(1).any():
                        break

                    reachable = np.append(reachable, test_pos, axis=0)

                    if (test_pos == foes).all(1).any():
                        break

        return np.array(reachable)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        return self.attack_cases(self.position, ally_pos, foe_pos)


class Queen(Piece):
    def __init__(self, coords, team):
        self.type = 'Queen'
        Piece.__init__(self, coords, team)

    def movement_range(self, ally, foe, **kwargs):
        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        reachable_rook = Rook.attack_cases(self.position, ally_pos, foe_pos)
        reachable_bishop = Bishop.attack_cases(self.position, ally_pos, foe_pos)
        if reachable_rook.size == 0:
            return reachable_bishop
        if reachable_bishop.size == 0:
            return reachable_rook
        return np.append(reachable_rook, reachable_bishop, axis=0)


class King(Piece):
    def __init__(self, coords, team):
        self.type = 'King'
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

    def movement_range(self, ally, foe, **kwargs):
        # kwargs:
        #   castling (= True): if True, calculates the possible moves due to castling with rooks

        ally_pos = ally.piece_positions
        foe_pos = foe.piece_positions
        reachable_rook = Rook.attack_cases(self.position, ally_pos, foe_pos, max_movement=1)
        reachable_bishop = Bishop.attack_cases(self.position, ally_pos, foe_pos, max_movement=1)
        reachable = np.append(reachable_rook, reachable_bishop, axis=0)

        if 'castling' not in kwargs.keys():
            return np.append(reachable, self.castling_moves(ally, foe), axis=0)

        if kwargs['castling']:
            return np.append(reachable, self.castling_moves(ally, foe), axis=0)

        return reachable

    def move(self, new_position):
        old_position = self.position
        super().move(new_position)
        if np.abs(old_position[0] - new_position[0]) > 1:
            return 'Castled'


class Team:
    def __init__(self, team_index):
        step_list = [-1, 1]
        back_row_list = [7, 0]
        self.back_row = back_row_list[team_index]
        self.prev_turn_jump = None
        self.index = team_index
        self.Pieces = {
            'Pawn': [Pawn(coords=np.array([cnt, self.back_row+step_list[team_index]]),
                          team=team_index, y_step=step_list[team_index]) for cnt in range(8)],
            'Rook': [Rook(coords=np.array([cnt, self.back_row]), team=team_index) for cnt in [0, 7]],
            'Knight': [Knight(coords=np.array([cnt, self.back_row]), team=team_index) for cnt in [1, 6]],
            'Bishop': [Bishop(coords=np.array([cnt, self.back_row]), team=team_index) for cnt in [2, 5]],
            'Queen': [Queen(coords=np.array([cnt, self.back_row]), team=team_index) for cnt in [4]],
            'King': [King(coords=np.array([cnt, self.back_row]), team=team_index) for cnt in [3]],
        }
        (self.piece_positions, self.piece_list) = self.list_pieces()

    def attacked_cases(self, foe):
        team_moves = np.empty((0, 2))
        for piece_iter in self.piece_list:
            piece_moves = self.Pieces[piece_iter[0]][piece_iter[1]].movement_range(self, foe, castling=False)
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

    def move_piece(self, piece_ind, position, foe):
        self.prev_turn_jump = None

        piece = self.Pieces[self.piece_list[piece_ind][0]][self.piece_list[piece_ind][1]]

        foe.eat_piece(foe.find_piece(position))

        if foe.piece_positions.size > 0:
            conflict_piece = (foe.piece_positions == position).all(1)
            if any(conflict_piece):
                eaten_piece_ind = np.nonzero(conflict_piece)
                foe.eat_piece(eaten_piece_ind[0][0])
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

import matplotlib.pyplot as plt
import numpy as np


class Piece:
    def __init__(self, coords, team):
        self.position = coords
        self.team = team
        self.has_moved = False
        self.visual = self.plot()

    def plot(self):
        if self.team == 1:
            line_spec = 'r'
        else:
            line_spec = 'b'

        return plt.text(self.position[0], self.position[1], 'p', color=line_spec,
                        horizontalalignment='center', verticalalignment='center')

    def move(self, new_position):
        self.position = new_position
        self.has_moved = True
        self.visual.set_x(new_position[0])
        self.visual.set_y(new_position[1])

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
        self.visual.set_text('P')

    def attack_cases(self, allies, foes):
        reachable = np.array([[x, self.y_step] for x in [-1, 1]]) + self.position
        return self.moves_inside_board(reachable)

    def movement_range(self, allies, foes):

        forward_move = self.moves_inside_board(self.position + np.array([[0, self.y_step]]))

        if not self.has_moved:
            forward_move = np.append(forward_move, np.array([[0, 2*self.y_step]]) + self.position, axis=0)

        return np.append(forward_move, self.attack_cases(allies, foes), axis=0)


class Rook(Piece):
    def __init__(self, coords, team):
        self.type = 'Rook'
        Piece.__init__(self, coords, team)
        self.visual.set_text('R')

    @classmethod
    def attack_cases(cls, position, allies, foes, max_movement=7):
        reachable = []
        for direction in [[0, 1], [1, 0]]:
            for sense in [1, -1]:
                for mag in range(1, max_movement+1):
                    test_pos = position + np.array(direction)*sense*mag
                    if not cls.is_inside_board(test_pos):
                        break
                    if (test_pos == allies).all(1).any():
                        break
                    reachable.append(test_pos)
                    if (test_pos == foes).all(1).any():
                        break

        return np.array(reachable)

    def movement_range(self, allies, foes):
        return self.attack_cases(self.position, allies, foes)


class Knight(Piece):
    def __init__(self, coords, team):
        self.type = 'Knight'
        Piece.__init__(self, coords, team)
        self.visual.set_text('H')

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

    def movement_range(self, allies, foes):
        return self.attack_cases(allies, foes)


class Bishop(Piece):
    def __init__(self, coords, team):
        self.type = 'Knight'
        Piece.__init__(self, coords, team)
        self.visual.set_text('B')

    @classmethod
    def attack_cases(cls, position, allies, foes, max_movement=7):
        reachable = []
        for x_dir in [-1, 1]:
            for y_dir in [-1, 1]:
                for mag in range(1, max_movement+1):
                    test_pos = position + np.array([x_dir, y_dir]) * mag
                    if not cls.is_inside_board(test_pos):
                        break
                    if (test_pos == allies).all(1).any():
                        break
                    reachable.append(test_pos)
                    if (test_pos == foes).all(1).any():
                        break

        return np.array(reachable)

    def movement_range(self, allies, foes):
        return self.attack_cases(self.position, allies, foes)


class Queen(Piece):
    def __init__(self, coords, team):
        self.type = 'Queen'
        Piece.__init__(self, coords, team)
        self.visual.set_text('Q')

    def movement_range(self, allies, foes):
        return np.append(Rook.attack_cases(self.position, allies, foes),
                         Bishop.attack_cases(self.position, allies, foes), axis=0)


class King(Piece):
    def __init__(self, coords, team):
        self.type = 'King'
        Piece.__init__(self, coords, team)
        self.visual.set_text('K')

    def movement_range(self, allies, foes):
        return np.append(Rook.attack_cases(self.position, allies, foes, max_movement=1),
                         Bishop.attack_cases(self.position, allies, foes, max_movement=1), axis=0)


class Team:
    def __init__(self, back_row, front_row, team_index):
        self.back_row = back_row
        self.index = team_index
        self.Pieces = {
            'Pawn': [Pawn(coords=np.array([cnt, front_row]), team=team_index, y_step=team_index) for cnt in range(8)],
            'Rook': [Rook(coords=np.array([cnt, back_row]), team=team_index) for cnt in [0, 7]],
            'Knight': [Knight(coords=np.array([cnt, back_row]), team=team_index) for cnt in [1, 6]],
            'Bishop': [Bishop(coords=np.array([cnt, back_row]), team=team_index) for cnt in [2, 5]],
            'Queen': [Queen(coords=np.array([cnt, back_row]), team=team_index) for cnt in [4]],
            'King': [King(coords=np.array([cnt, back_row]), team=team_index) for cnt in [3]],
        }

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

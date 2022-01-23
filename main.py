# Note: Some AI concepts and GUI settings were taken from:
# https://github.com/johnafish/othello/blob/master/othello.py
# https://github.com/arminkz/Reversi
# importing necessary libraries
from tkinter import *
from math import *
from time import *
from random import *
import numpy as np
from copy import deepcopy
import datetime

# Variable setup
nodes = 0
depth = 6
moves = 0

# creating the screen
start = Tk()
# background image
img = PhotoImage(file="Othello_and_Desdemona_by_Daniel_Maclis.png")
canvas_height = 600
canvas_width = 500
window = Canvas(start, width=canvas_width, height=canvas_height)
window.pack(fill=BOTH, expand=True)
window.create_image(0, 0, image=img, anchor="nw")

unit_time = []


# creating the board
class create_board:
    def __init__(self):
        self.player = 0
        self.passed = False
        self.won = False
        # array is the board
        self.array = []
        for x in range(8):
            self.array.append([])
            for y in range(8):
                self.array[x].append(None)

        # four starting discs
        self.array[3][3] = "w"
        self.array[4][4] = "w"
        self.array[3][4] = "b"
        self.array[4][3] = "b"

        self.oldarray = self.array

    # updates the board
    def update(self):
        window.configure(background="darkgreen")
        window.delete("highlight")
        window.delete("tile")
        for x in range(8):
            for y in range(8):
                if self.oldarray[x][y] == "w":
                    window.create_oval(54 + 50 * x, 54 + 50 * y, 96 + 50 * x, 96 + 50 * y,
                                       tags="tile {0}-{1}".format(x, y), fill="#aaa", outline="#aaa")
                    window.create_oval(54 + 50 * x, 52 + 50 * y, 96 + 50 * x, 94 + 50 * y,
                                       tags="tile {0}-{1}".format(x, y), fill="#fff", outline="#fff")

                elif self.oldarray[x][y] == "b":
                    window.create_oval(54 + 50 * x, 54 + 50 * y, 96 + 50 * x, 96 + 50 * y,
                                       tags="tile {0}-{1}".format(x, y), fill="#000", outline="#000")
                    window.create_oval(54 + 50 * x, 52 + 50 * y, 96 + 50 * x, 94 + 50 * y,
                                       tags="tile {0}-{1}".format(x, y), fill="#111", outline="#111")
        window.update()
        for x in range(8):
            for y in range(8):
                if self.array[x][y] != self.oldarray[x][y] and self.array[x][y] == "w":
                    window.delete("{0}-{1}".format(x, y))
                    for i in range(21):
                        window.create_oval(54 + i + 50 * x, 54 + i + 50 * y, 96 - i + 50 * x, 96 - i + 50 * y,
                                           tags="tile animated", fill="#000", outline="#000")
                        window.create_oval(54 + i + 50 * x, 52 + i + 50 * y, 96 - i + 50 * x, 94 - i + 50 * y,
                                           tags="tile animated", fill="#111", outline="#111")
                        if i % 3 == 0:
                            sleep(0.01)
                        window.update()
                        window.delete("animated")
                    for i in reversed(range(21)):
                        window.create_oval(54 + i + 50 * x, 54 + i + 50 * y, 96 - i + 50 * x, 96 - i + 50 * y,
                                           tags="tile animated", fill="#aaa", outline="#aaa")
                        window.create_oval(54 + i + 50 * x, 52 + i + 50 * y, 96 - i + 50 * x, 94 - i + 50 * y,
                                           tags="tile animated", fill="#fff", outline="#fff")
                        if i % 3 == 0:
                            sleep(0.01)
                        window.update()
                        window.delete("animated")
                    window.create_oval(54 + 50 * x, 54 + 50 * y, 96 + 50 * x, 96 + 50 * y, tags="tile", fill="#aaa",
                                       outline="#aaa")
                    window.create_oval(54 + 50 * x, 52 + 50 * y, 96 + 50 * x, 94 + 50 * y, tags="tile", fill="#fff",
                                       outline="#fff")
                    window.update()

                elif self.array[x][y] != self.oldarray[x][y] and self.array[x][y] == "b":
                    window.delete("{0}-{1}".format(x, y))
                    for i in range(21):
                        window.create_oval(54 + i + 50 * x, 54 + i + 50 * y, 96 - i + 50 * x, 96 - i + 50 * y,
                                           tags="tile animated", fill="#aaa", outline="#aaa")
                        window.create_oval(54 + i + 50 * x, 52 + i + 50 * y, 96 - i + 50 * x, 94 - i + 50 * y,
                                           tags="tile animated", fill="#fff", outline="#fff")
                        if i % 3 == 0:
                            sleep(0.01)
                        window.update()
                        window.delete("animated")
                    for i in reversed(range(21)):
                        window.create_oval(54 + i + 50 * x, 54 + i + 50 * y, 96 - i + 50 * x, 96 - i + 50 * y,
                                           tags="tile animated", fill="#000", outline="#000")
                        window.create_oval(54 + i + 50 * x, 52 + i + 50 * y, 96 - i + 50 * x, 94 - i + 50 * y,
                                           tags="tile animated", fill="#111", outline="#111")
                        if i % 3 == 0:
                            sleep(0.01)
                        window.update()
                        window.delete("animated")

                    window.create_oval(54 + 50 * x, 54 + 50 * y, 96 + 50 * x, 96 + 50 * y, tags="tile", fill="#000",
                                       outline="#000")
                    window.create_oval(54 + 50 * x, 52 + 50 * y, 96 + 50 * x, 94 + 50 * y, tags="tile", fill="#111",
                                       outline="#111")
                    window.update()

        for x in range(8):
            for y in range(8):
                if self.player == 0:
                    if valid(self.array, self.player, x, y):
                        window.create_oval(68 + 50 * x, 68 + 50 * y, 32 + 50 * (x + 1), 32 + 50 * (y + 1),
                                           tags="highlight", fill="#008000", outline="#008000")

        if not self.won:
            self.drawScoreBoard()
            window.update()
            if self.player == 1:
                startTime = time()
                self.oldarray = self.array
                begin_time = datetime.datetime.now()
                # calls for evaluator function in Iago at a given depth
                alphaBetaResult = Iago(self.array, self.player, depth, Evaluator())
                # finding the run time of choosing the next move with a given heuristic
                print(datetime.datetime.now() - begin_time)
                unit_time.append(datetime.datetime.now() - begin_time)
                self.array = alphaBetaResult[1]
                if len(alphaBetaResult) == 3:
                    position = alphaBetaResult[2]
                    self.oldarray[position[0]][position[1]] = "w"

                self.player = 1 - self.player
                deltaTime = round((time() - startTime) * 100) / 100
                if deltaTime < 2:
                    sleep(2 - deltaTime)
                nodes = 0
                self.pass_test()
        else:
            window.create_text(270, 540, anchor="c", font=("Consolas", 15), text="The Game is Done!!! \n Play again?")

    def boardMove(self, x, y):
        global nodes
        # Move and update screen
        self.oldarray = self.array
        self.oldarray[x][y] = "b"
        self.array = move(self.array, x, y)

        self.player = 1 - self.player
        self.update()

        self.pass_test()
        self.update()

    def drawScoreBoard(self):
        global moves
        window.delete("score")
        player_score = 0
        computer_score = 0
        for x in range(8):
            for y in range(8):
                if self.array[x][y] == "b":
                    player_score += 1
                elif self.array[x][y] == "w":
                    computer_score += 1

        window.create_text(80, 520, anchor="w", tags="score", font=("Times New Roman", 50, "bold"),
                           fill="black", text=player_score)
        window.create_text(386, 520, anchor="w", tags="score", font=("Times New Roman", 50, "bold"),
                           fill="white", text=computer_score)

        moves = player_score + computer_score

    def pass_test(self):
        must_pass = True
        for x in range(8):
            for y in range(8):
                if valid(self.array, self.player, x, y):
                    must_pass = False
        if must_pass:
            self.player = 1 - self.player
            if self.passed == True:
                self.won = True
            else:
                self.passed = True
            self.update()
        else:
            self.passed = False


nodes_set = []

'''
Minimax Search that has two static methods
one is Solve: given the board, player(0 or 1), depth and certain heuristics
gives back the best score, move and correspondingly the board
calls alpha_beta method for pruning
'''


class Minimax:
    nodes_expanded = 0

    @staticmethod
    def solve(array, player, d, evaluator):
        Minimax.nodes_expanded = 0
        best_board = []
        best_score = -float("inf")
        best_move = None
        all_moves = get_all_possible_moves(array, player)
        for m in all_moves:
            new_node = move(array, m[0], m[1], player)
            child_score = Minimax.alpha_beta(new_node, player, d - 1, False, -float("inf"), float("inf"), evaluator)
            print("Child Score: ", child_score)
            if child_score > best_score:
                best_score = child_score
                best_move = m
                best_board = new_node
        print("Nodes Expanded: ", Minimax.nodes_expanded, "Score chosen: ", best_score)
        nodes_set.append(Minimax.nodes_expanded)
        if best_move is None:
            return [best_score, array]
        return [best_score, best_board, best_move]

    @staticmethod
    def alpha_beta(node, player, d, mx, alpha, beta, evaluator):
        Minimax.nodes_expanded += 1
        if d <= 0 or is_game_over(node):
            return evaluator.eval(node, player)
        if (mx and not has_any_moves(node, player)) or (not mx and not has_any_moves(node, 1 - player)):
            return Minimax.alpha_beta(node, player, d - 1, not mx, alpha, beta, evaluator)
        if mx:
            score = -float("inf")
            for m in get_all_possible_moves(node, player):
                new_node = move(node, m[0], m[1], 1 - player)
                child_score = Minimax.alpha_beta(new_node, player, d - 1, False, alpha, beta, evaluator)
                if child_score > score:
                    score = child_score
                if score > alpha:
                    alpha = score
                if beta <= alpha:
                    break
        else:
            score = float("inf")
            for m in get_all_possible_moves(node, player):
                new_node = move(node, m[0], m[1], 1 - player)
                child_score = Minimax.alpha_beta(new_node, player, d - 1, True, alpha, beta, evaluator)
                if child_score < score:
                    score = child_score
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break
        return score


# game is over when np player has any moves
def is_game_over(array):
    return not (has_any_moves(array, 0) or has_any_moves(array, 1))


# the player has a move if there are possible moves at a given state
def has_any_moves(array, player):
    return len(get_all_possible_moves(array, player)) > 0


# changes the board after the m move and gives back the new board
def board_after_move(array, player, m):
    new_board = []
    for i in range(8):
        new_board.append([])
        for j in range(8):
            new_board[i].append(array[i][j])

    if player == 0:
        colour = "b"
    else:
        colour = "w"

    new_board[m[0]][m[1]] = colour
    rev = get_reverse_points(new_board, player, m[0], m[1])
    for flip in rev:
        new_board[flip[0]][flip[1]] = colour

    return new_board


# given the board, player, i and j gives back all the points that can be reversed
def get_reverse_points(array, player, i, j):
    if player == 1:
        colour = "w"
        opponent = "b"
    else:
        colour = "b"
        opponent = "w"
    all_reverse_points = []

    # move up
    ms = []
    mi = i - 1
    mj = j
    while mi > 0 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mi -= 1
    if mi >= 0 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move down
    ms = []
    mi = i + 1
    mj = j
    while mi < 7 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mi += 1
    if mi <= 7 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move left
    ms = []
    mi = i
    mj = j - 1
    while mj > 0 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mj -= 1
    if mj >= 0 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move right
    ms = []
    mi = i
    mj = j + 1
    while mj < 7 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mj += 1
    if mj <= 7 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move up left
    ms = []
    mi = i - 1
    mj = j - 1
    while mj > 0 and mi > 0 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mi -= 1
        mj -= 1
    if mj >= 0 and mi >= 0 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move up right
    ms = []
    mi = i - 1
    mj = j + 1
    while mj < 7 and mi > 0 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mi -= 1
        mj += 1
    if mj <= 7 and mi >= 0 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move down left
    ms = []
    mi = i + 1
    mj = j - 1
    while mj > 0 and mi < 7 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mi += 1
        mj -= 1
    if mj >= 0 and mi <= 7 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    # move down right
    ms = []
    mi = i + 1
    mj = j + 1
    while mj < 7 and mi < 7 and array[mi][mj] == opponent:
        ms.append((mi, mj))
        mi += 1
        mj += 1
    if mj <= 7 and mi <= 7 and array[mi][mj] == colour and ms:
        for m in ms:
            all_reverse_points.append(m)

    return all_reverse_points


# given the player, board, coordinates moves and returns the new board
def move(passed_array, x, y, player=-1):
    array = deepcopy(passed_array)
    if player == -1:
        player = board.player
    if player == 1:
        colour = "w"
    else:
        colour = "b"
    array[x][y] = colour

    # Determining the neighbours to the square
    neighbours = []
    for i in range(max(0, x - 1), min(x + 2, 8)):
        for j in range(max(0, y - 1), min(y + 2, 8)):
            if array[i][j] != None:
                neighbours.append([i, j])

    # Which tiles to convert
    convert = []

    # For all the generated neighbours, determine if they form a line
    # If a line is formed, we will add it to the convert array
    for neighbour in neighbours:
        neighX = neighbour[0]
        neighY = neighbour[1]
        # Check if the neighbour is of a different colour - it must be to form a line
        if array[neighX][neighY] != colour:
            # The path of each individual line
            path = []

            # Determining direction to move
            deltaX = neighX - x
            deltaY = neighY - y

            tempX = neighX
            tempY = neighY

            # While we are in the bounds of the board
            while 0 <= tempX <= 7 and 0 <= tempY <= 7:
                path.append([tempX, tempY])
                value = array[tempX][tempY]
                # If we reach a blank tile, we're done and there's no line
                if value == None:
                    break
                # If we reach a tile of the player's colour, a line is formed
                if value == colour:
                    # Append all of our path nodes to the convert array
                    for node in path:
                        convert.append(node)
                    break
                # Move the tile
                tempX += deltaX
                tempY += deltaY

    # Convert all the appropriate tiles
    for node in convert:
        array[node[0]][node[1]] = colour

    return array


# Method for drawing the gridlines
def drawGridBackground(outline=False):
    if outline:
        window.create_rectangle(50, 50, 450, 450, outline="#111")

    for i in range(7):
        lineShift = 50 + 50 * (i + 1)

        window.create_line(50, lineShift, 450, lineShift, fill="#111")

        window.create_line(lineShift, 50, lineShift, 450, fill="#111")

    window.update()


'''
mobility heuristics finds the difference of the number of allowable moves
'''


def mobility(array, player):
    my_move_count = len(get_all_possible_moves(array, player))
    opp_move_count = len(get_all_possible_moves(array, 1 - player))

    return my_move_count - opp_move_count


'''
goes into eight directions for each disc and finds the empty ones
'''


def get_frontier_squares(array, player):
    if player == 1:
        colour = "w"
    else:
        colour = "b"
    arr = []

    for i in range(8):
        for j in range(8):
            if array[i][j] == colour:
                possible_frontiers = []
                if i > 0 and not array[i - 1][j]:
                    possible_frontiers.append((i - 1, j))
                if i < 7 and not array[i + 1][j]:
                    possible_frontiers.append((i + 1, j))
                if j > 0 and not array[i][j - 1]:
                    possible_frontiers.append((i, j - 1))
                if j < 7 and not array[i][j + 1]:
                    possible_frontiers.append((i, j + 1))
                if i > 0 and j > 0 and not array[i - 1][j - 1]:
                    possible_frontiers.append((i - 1, j - 1))
                if i > 0 and j < 7 and not array[i - 1][j + 1]:
                    possible_frontiers.append((i - 1, j + 1))
                if i < 7 and j > 0 and not array[i + 1][j - 1]:
                    possible_frontiers.append((i + 1, j - 1))
                if i < 7 and j < 7 and not array[i + 1][j + 1]:
                    possible_frontiers.append((i + 1, j + 1))

                for k in possible_frontiers:
                    if k not in arr:
                        arr.append(k)

    return arr


'''
frontier heuristics finds the difference of the above 
function for the both player
'''


def frontier_heuristic(array, player):
    my_frontier = len(get_frontier_squares(array, player))
    opp_frontier = len(get_frontier_squares(array, 1 - player))

    return my_frontier - opp_frontier


'''
finds the number of discs
'''


def get_player_disc_count(array, player):
    if player == 1:
        colour = "w"
    else:
        colour = "b"
    count = 0

    for i in range(8):
        for j in range(8):
            if array[i][j] == colour:
                count += 1

    return count


'''
pieces heuristic which finds the difference of the number of discs for each player
'''


def pieces_heuristic(array, player):
    my_count = get_player_disc_count(array, player)
    opp_count = get_player_disc_count(array, 1 - player)

    return my_count - opp_count


# every square has a score shown below
square_score = [
    [100, -10, 8, 6, 6, 8, -10, 100],
    [-10, -25, -4, -4, -4, -4, -25, -10],
    [8, -4, 6, 4, 4, 6, -4, 8],
    [6, -4, 4, 0, 0, 4, -4, 6],
    [6, -4, 4, 0, 0, 4, -4, 6],
    [8, -4, 6, 4, 4, 6, -4, 8],
    [-10, -25, -4, -4, -4, -4, -25, -10],
    [100, -10, 8, 6, 6, 8, -10, 100]]

'''
according to the above matrix of scores, finds the score for each player
and finds the difference
'''


def placement_heuristic(array, player):
    if player == 1:
        colour = "w"
        opp = "b"
    else:
        colour = "b"
        opp = "w"

    my_w = 0
    opp_w = 0

    for i in range(8):
        for j in range(8):
            if array[i][j] == colour:
                my_w += square_score[i][j]
            if array[i][j] == opp:
                opp_w += square_score[i][j]

    return my_w - opp_w


'''
finds the number of discs that will not be
outflanked till the end of the game
'''


def get_stable_disc(array, player, i, j):
    stable_disc = []
    if player == 1:
        colour = "w"
        opp = "b"
    else:
        colour = "b"
        opp = "w"
    # move up points
    moves_p = []
    m_i = i - 1
    m_j = j

    while m_i > 0 and array[m_i][m_j] == colour:  # maybe colour
        moves_p.append((m_i, m_j))
        m_i -= 1

    for k in moves_p:
        if k not in stable_disc:
            stable_disc.append(k)

    moved_p = []
    m_i = i + 1
    m_j = j

    while m_i < 7 and array[m_i][m_j] == opp:
        moved_p.append((m_i, m_j))
        m_i += 1

    for k in moved_p:
        if k not in stable_disc:
            stable_disc.append(k)

    moveleft_p = []
    m_i = i
    m_j = j - 1

    while m_j > 0 and array[m_i][m_j] == opp:
        moveleft_p.append((m_i, m_j))
        m_j -= 1

    for k in moveleft_p:
        if k not in stable_disc:
            stable_disc.append(k)

    moveright_p = []
    m_i = i
    m_j = j + 1

    while m_j < 7 and array[m_i][m_j] == opp:
        moveright_p.append((m_i, m_j))
        m_j += 1

    for k in moveright_p:
        if k not in stable_disc:
            stable_disc.append(k)

    moveupleft_p = []
    m_i = i - 1
    m_j = j - 1

    while m_i > 0 and m_j > 0 and array[m_i][m_j] == opp:
        moveupleft_p.append((m_i, m_j))
        m_i -= 1
        m_j -= 1

    for k in moveupleft_p:
        if k not in stable_disc:
            stable_disc.append(k)

    moveupright_p = []
    m_i = i - 1
    m_j = j + 1

    while m_i > 0 and m_j < 7 and array[m_i][m_j] == opp:
        moveupleft_p.append((m_i, m_j))
        m_i -= 1
        m_j += 1

    for k in moveupright_p:
        if k not in stable_disc:
            stable_disc.append(k)

    movedownleft_p = []
    m_i = i + 1
    m_j = j - 1

    while m_i < 7 and m_j > 0 and array[m_i][m_j] == opp:
        movedownleft_p.append((m_i, m_j))
        m_i += 1
        m_j -= 1

    for k in movedownleft_p:
        if k not in stable_disc:
            stable_disc.append(k)

    movedownright_p = []
    m_i = i + 1
    m_j = j + 1

    while m_i < 7 and m_j < 7 and array[m_i][m_j] == opp:
        movedownright_p.append((m_i, m_j))
        m_i += 1
        m_j += 1

    for k in movedownright_p:
        if k not in stable_disc:
            stable_disc.append(k)

    return stable_disc


'''
stability heuristics finds the difference
of those squares from above functions
'''


def stability_heuristic(array, player):
    my_s = 0
    opp_s = 0
    if player == 1:
        colour = "w"
        opp = "b"
    else:
        colour = "b"
        opp = "w"

    if array[0][0] == colour:
        my_s += len(get_stable_disc(array, player, 0, 0))

    if array[0][7] == colour:
        my_s += len(get_stable_disc(array, player, 0, 7))

    if array[7][0] == colour:
        my_s += len(get_stable_disc(array, player, 7, 0))

    if array[7][7] == colour:
        my_s += len(get_stable_disc(array, player, 7, 7))

    if array[0][0] == opp:
        opp_s += len(get_stable_disc(array, 1 - player, 0, 0))

    if array[0][7] == opp:
        opp_s += len(get_stable_disc(array, 1 - player, 0, 7))

    if array[7][0] == opp:
        opp_s += len(get_stable_disc(array, 1 - player, 7, 0))

    if array[7][7] == opp:
        opp_s += len(get_stable_disc(array, 1 - player, 7, 7))

    return my_s - opp_s


'''
corner move heuristic checks whether it 
can move to corner on the next move
'''


def corner_move(array, player):
    moves = get_all_possible_moves(array, player)

    for m in moves:
        if m[0] == 0 and m[1] == 0:
            return 100
        if m[0] == 0 and m[1] == 7:
            return 100
        if m[0] == 7 and m[1] == 0:
            return 100
        if m[0] == 7 and m[1] == 7:
            return 100

    return 0


# Evaluator class that finds the score for minimax
class Evaluator:
    # uncomment the heuristic you want to use
    def eval(self, array, player):
        # score = mobility(array, player)
        # score = stability_heuristic(array, player)
        # score = corner_move(array, player)
        # score = frontier_heuristic(array, player)
        # score = placement_heuristic(array, player)
        score = pieces_heuristic(array, player)
        return score


# checks for validity of moves
def valid(array, player, x, y):
    if array[x][y] is not None:
        return False

    if player == 1:
        colour = "w"
        oplayer = "b"
    else:
        colour = "b"
        oplayer = "w"

    mi = x - 1
    mj = y
    c = 0
    while mi > 0 and array[mi][mj] == oplayer:
        mi -= 1
        c += 1
    if mi >= 0 and array[mi][mj] == colour and c > 0:
        return True

    mi = x + 1
    mj = y
    c = 0
    while mi < 7 and array[mi][mj] == oplayer:
        mi += 1
        c += 1
    if mi <= 7 and array[mi][mj] == colour and c > 0:
        return True

    mi = x
    mj = y - 1
    c = 0
    while mj > 0 and array[mi][mj] == oplayer:
        mj -= 1
        c += 1
    if mj >= 0 and array[mi][mj] == colour and c > 0:
        return True

    mi = x
    mj = y + 1
    c = 0
    while mj < 7 and array[mi][mj] == oplayer:
        mj += 1
        c += 1
    if mj <= 7 and array[mi][mj] == colour and c > 0:
        return True

    mi = x - 1
    mj = y - 1
    c = 0
    while mj > 0 and mi > 0 and array[mi][mj] == oplayer:
        mj -= 1
        mi -= 1
        c += 1
    if mj >= 0 and mi > 0 and array[mi][mj] == colour and c > 0:
        return True

    mi = x + 1
    mj = y + 1
    c = 0
    while mj < 7 and mi < 7 and array[mi][mj] == oplayer:
        mj += 1
        mi += 1
        c += 1
    if mj <= 7 and mi <= 7 and array[mi][mj] == colour and c > 0:
        return True

    mi = x + 1
    mj = y - 1
    c = 0
    while mj > 0 and mi < 7 and array[mi][mj] == oplayer:
        mj -= 1
        mi += 1
        c += 1
    if mj >= 0 and mi <= 7 and array[mi][mj] == colour and c > 0:
        return True

    mi = x - 1
    mj = y + 1
    c = 0
    while mj < 7 and mi > 0 and array[mi][mj] == oplayer:
        mj += 1
        mi -= 1
        c += 1
    if mj <= 7 and mi >= 0 and array[mi][mj] == colour and c > 0:
        return True


# return the possible moves according to validity
def get_all_possible_moves(array, player):
    arr = []
    for i in range(8):
        for j in range(8):
            if valid(array, player, i, j):
                arr.append((i, j))
    return arr

'''
Iago is the main evaluation
-checks whether it can do corner move without minimax
-checks whether can blovk the opponent (leave no valid move) without minimax
-otherwise minimax with heuristics
'''
def Iago(array, player, d, evaluator):
    ms = get_all_possible_moves(array, player)
    best_move = None
    best_value = -float("inf")
    best_board = None

    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    for m in ms:
        if m in corners:
            new_board = board_after_move(array, player, m)
            mval = evaluator.eval(new_board, player)
            if mval > best_value:
                best_value = mval
                best_board = new_board
                best_move = m
    if best_move is not None:
        print("\033[1;30;34m IAGO MOVE : CORNER \033[0m\n")
        return [best_value, best_board, best_move]

    for m in ms:
        new_board = board_after_move(array, player, m)
        if len(get_all_possible_moves(new_board, 1 - player)) == 0:
            mval = evaluator.eval(new_board, player)
            if mval > best_value:
                best_value = mval
                best_board = new_board
                best_move = m

    if best_move is not None:
        print("\033[1;30;34m IAGO MOVE : BLOCKING MOVE \033[0m\n")
        return [best_value, best_board, best_move]

    return Minimax.solve(array, player, d, evaluator)


# When the user clicks, if it's a valid move, make the move
def clickHandle(event):
    global depth
    xMouse = event.x
    yMouse = event.y
    if running:
        if xMouse >= 450 and yMouse <= 50:
            start.destroy()
        elif xMouse <= 50 and yMouse <= 50:
            playGame()
        else:
            if board.player == 0:
                x = int((event.x - 50) / 50)
                y = int((event.y - 50) / 50)
                if 0 <= x <= 7 and 0 <= y <= 7:
                    if valid(board.array, board.player, x, y):
                        board.boardMove(x, y)
    else:
        if 410 <= yMouse <= 495:
            # easy
            if 30 <= xMouse <= 150:
                depth = 1
                playGame()
            # medium
            elif 185 <= xMouse <= 305:
                depth = 4
                playGame()
            # hard
            elif 340 <= xMouse <= 460:
                # depth can be changed to bigger numbers
                depth = 6
                playGame()


def keyHandle(event):
    symbol = event.keysym
    if symbol.lower() == "r":
        playGame()
    elif symbol.lower() == "q":
        start.destroy()


def create_buttons():
    window.create_rectangle(0, 0, 50, 50, fill="CadetBlue4", outline="CadetBlue4")
    window.create_arc(5, 5, 45, 45, fill="navy", width="2", style="arc", outline="navy", extent=300)
    window.create_polygon(33, 38, 36, 45, 40, 39, fill="navy", outline="navy")
    window.create_rectangle(452, 2, 498, 48, fill="brown4", outline="brown4")
    window.create_rectangle(455, 5, 495, 45, fill="PeachPuff2", outline="PeachPuff2")
    window.create_line(460, 10, 490, 40, fill="brown4", width="4")
    window.create_line(490, 10, 460, 40, fill="brown4", width="4")


def runGame():
    global running
    running = False
    window.create_text(255, 151, anchor="c", text="Othello", font=("Purisa", 85, "bold"), fill="Ivory3")
    window.create_text(250, 148, anchor="c", text="Othello", font=("Purisa", 85, "bold"), fill="ghost white")
    for i in range(3):
        window.create_oval(30 + 155 * i, 405, 150 + 155 * i, 500, fill="white", outline="white")
        window.create_oval(35 + 155 * i, 410, 145 + 155 * i, 495, fill="#000", outline="#000")
        window.create_text(90, 450, anchor="c", text="easy", font=("Consolas", 25), fill="#ffd700")
        window.create_text(247, 450, anchor="c", text="medium", font=("Consolas", 22), fill="#ffd700")
        window.create_text(400, 450, anchor="c", text="hard", font=("Consolas", 25), fill="#ffd700")

    window.update()


def playGame():
    global board, running
    running = True
    window.delete(ALL)
    create_buttons()
    board = 0
    drawGridBackground()
    board = create_board()
    board.update()

#runs the game
runGame()

window.bind("<Button-1>", clickHandle)
window.bind("<Key>", keyHandle)
window.focus_set()

start.wm_title("Othello")
start.mainloop()

#for analyzing
print("avg runtime")
print(datetime.timedelta(seconds=sum(td.total_seconds() / len(unit_time) for td in unit_time)))
print("avg nodes expanded")
print(sum(nodes_set) / len(nodes_set))

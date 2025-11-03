import numpy as np
from copy import deepcopy

def heuristic_move(game):
    
    # Simple rule-based agent for Connect 4.
    # Evaluates possible moves by counting potential 2s, 3s, 4s.
    # Returns the column index to play.
    
    valid_moves = game.get_valid_moves()
    best_score = -float("inf")
    best_move = np.random.choice(valid_moves)

    for col in valid_moves:
        temp_game = deepcopy(game)
        row = temp_game.get_lowest_open_row(col)
        temp_game.board[row, col] = temp_game.current_player
        score = evaluate_board(temp_game.board, temp_game.current_player)
        if score > best_score:
            best_score = score
            best_move = col

    return best_move


def evaluate_board(board, player):
    opp = -player
    my_twos, my_threes, my_fours = count_windows(board, player)
    opp_twos, opp_threes, opp_fours = count_windows(board, opp)

    # Weighted scoring
    return (my_fours * 1000 + my_threes * 5 + my_twos * 2) - (
        opp_fours * 1000 + opp_threes * 5 + opp_twos * 2
    )


def count_windows(board, player):
    twos = threes = fours = 0
    for r in range(6):
        for c in range(7):
            # Horizontal
            if c + 3 < 7:
                window = list(board[r, c:c+4])
                twos, threes, fours = update_counts(window, player, twos, threes, fours)
            # Vertical
            if r + 3 < 6:
                window = list(board[r:r+4, c])
                twos, threes, fours = update_counts(window, player, twos, threes, fours)
            # Diagonal \
            if r + 3 < 6 and c + 3 < 7:
                window = [board[r+i][c+i] for i in range(4)]
                twos, threes, fours = update_counts(window, player, twos, threes, fours)
            # Diagonal /
            if r - 3 >= 0 and c + 3 < 7:
                window = [board[r-i][c+i] for i in range(4)]
                twos, threes, fours = update_counts(window, player, twos, threes, fours)
    return twos, threes, fours


def update_counts(window, player, twos, threes, fours):
    count = window.count(player)
    empty = window.count(0)
    if count == 2 and empty == 2:
        twos += 1
    elif count == 3 and empty == 1:
        threes += 1
    elif count == 4:
        fours += 1
    return twos, threes, fours


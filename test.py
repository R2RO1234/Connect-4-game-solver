import Minimax
import numpy as np
board_diagonal_right = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 1, 0, 0,-1, 0, 0]
], dtype=int),[4])
board_diagonal_left = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0,-1, 1, 0, 0],
    [ 0, 0, 0, 1,-1, 1, 0],
    [ 0, 0, 0, 1,-1, 0, 1]
], dtype=int),[3])
board_horizontal = (np.array([
    [0, 0, 0, 0, 0,0,0],
    [0, 0, 0, 0, 0,0,0],
    [0, 0, 0, 0, 0,0,0],
    [0, 0, 0, 0, 0,0,0],
    [0,-1,-1,-1, 0,0,0],
    [0, 1, 1, 1,-1,0,0]
], dtype=int),[0,4])
board_vertical = (np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0]
], dtype=int),[2])
board_vertical_closed = (np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,-1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0]
], dtype=int),[])
board_vertical_up = (np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0],
    [0,0,-1,0,0,0,0]
], dtype=int),[2])


board_winner_diagonal = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 1, 0, 0],
    [ 0, 0, 0, 1, 1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 1, 0, 0,-1, 0, 0]
], dtype=int),(True , 1))

board_winner_vertical = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0,-1, 0, 0],
    [ 0, 0, 0, 1,-1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 1, 0, 0,-1, 0, 0]
], dtype=int),(True , -1))

board_winner_horizontal = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0,-1, 0, 0],
    [ 0, 0, 0, 1, 1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 1, 1, 1, 1, 0, 0]
], dtype=int),(True , 1))

board_winner_diagonal_left = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 1, 0, 0, 0,-1, 0, 0],
    [-1, 1, 0, 1, 1, 0, 0],
    [-1,-1, 1, 0,-1, 0, 0],
    [ 1, 1,-1, 1,-1, 0, 0]
], dtype=int),(True , 1))



Infinity = Minimax.Infinity

board_evaluate = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 1, 0, 0,-1, 0, 0]
], dtype=int),1,Infinity)

board_evaluate1 = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1,-1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 1, 0, 0,-1, 0, 0]
], dtype=int),-1,-Infinity)


board_evaluate2 = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0]
], dtype=int),1,0)

board_evaluate3 = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 0, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0]
], dtype=int), -1,1)

board_evaluate4 = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0],
    [ 0, 0, 1, 0,-1, 0, 0]
], dtype=int), -1,2)

board_evaluate5 = (np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 1, 0, 0, 0, 0],
    [ 1, 0,-1, 0, 0, 0, 0],
    [ 1, 0,-1, 0, 1, 0, 0],
    [ 1, 0,-1, 0, 1, 0, 0]
], dtype=int), 1,Infinity)



mini = Minimax.minimax(4)
index = 0


def test_function(function , lst):
    for index, (board, solution) in enumerate(lst):
       
        moves = function(board)
        
        try:
            assert (moves == solution)
            print(f'checking {index} was successful')
        except:
            print(f'test {index} failed')
    
        index+=1

def test_evaluation(lst):
    for index, (board, player , solution) in enumerate(lst):
       
        moves = mini.evaluate_position(board , player)
        
        try:
            assert (moves == solution)
            print(f'checking {index} was successful')
        except:
            print(f'test {index} failed. expected {solution} but got {moves}')
    
        index+=1

if __name__ == "__main__":
    test_winner = [board_winner_diagonal, board_winner_diagonal_left ,board_winner_horizontal , board_winner_vertical]
    test_mandatory = [board_diagonal_right , board_diagonal_left, board_horizontal , board_vertical,board_vertical_closed,board_vertical_up]
    #test_function(mini.check_mandatory_moves , test_mandatory)
    #test_function(mini.check_winner_accross_board , test_winner)
    boards = [board_evaluate, board_evaluate1, board_evaluate2, board_evaluate3,board_evaluate4,board_evaluate5 ]
    test_evaluation(boards)
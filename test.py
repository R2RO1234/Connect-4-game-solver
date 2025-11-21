import time
import zlib
import Minimax
import numpy as np
import board_dataset
import os
import shutil
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
    agent = Minimax.minimax(7)
    bd = board_dataset.BoardDataset("with move ordering")
    bd.expand_dict_and_save(100)
    bd.evaluate_remaining_boards(agent,3)
    
    
    boards =  list(bd.load_dict().values())
    hashes = list(bd.load_dict().keys())
    evaluated = bd.evaluated_boards
    for i in range(130):
        try:
            assert zlib.adler32(boards[i].tobytes()) == hashes[i]
        except: 
            print("hashing did not work")
        board = boards[i]
        num_pos = np.sum(board == 1)
        num_neg = np.sum(board == -1)
        turn = 1 - 2 * (num_pos - num_neg)

        agent.choose_move(board, turn, True)
        try:
            before = time.time()
            evaluation = agent.choose_move(board, turn, True)[1]
            after = time.time()
            np.testing.assert_almost_equal(evaluation, evaluated[i])
            print(f'board number {i} worked. evaluated it in {(after - before):.4f}s')
        except:
            #print(board)
            print(f'board {i} is invalid. expected {int(evaluated[i])} but got {int(evaluation)}')

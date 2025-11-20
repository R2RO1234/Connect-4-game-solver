from typing import List
import Minimax
from logic import Connect4
import random
import numpy as np
import zlib
import time
def write_boards(board_list  ,src: str):
    with open(src , "wb") as f:
         np.save(f, board_list)
       
def read_boards(scr : str):
    with open(scr , "rb") as f:
        a = np.load(f)
    return a
def write_hashes(hash_list : List[str],src : str, mode: str):
    with open(src , mode) as f:
        for h in hash_list:
            f.write(f"{h}\n")
       
def read_hashes(src : str):
    with open(src , "r") as f: 
        hashes = [int(line.strip()) for line in f]
    return hashes

def get_dict_from_files(keys_src :str , values_src: str):
    keys = read_hashes(keys_src)
    values = read_boards(values_src)
    return dict(zip(keys,values))



def compute_board_states(board_states = {}, num_games = 10000):
    total_moves = len(board_states)
    for i in range(num_games): # simulate num_iterations games
        player_1 = Minimax.get_random_minimax()
        player_2 = Minimax.get_random_minimax()
        #print("player1: "+ str(player_1))
        #print("player2: "+ str(player_2))
        previous = len(board_states)
        current_player = 1
        game = Connect4()
        game.reset()
        num_moves = 0
        while True:
            agent = player_1 if current_player == 1 else player_2
            
            action =  agent.choose_move(game.board , current_player)
            #if random.random() < 0.005: game.print_state()
            state, reward, done = game.make_move(action)
            current_player*=-1
            
            hashed = zlib.adler32(game.board.tobytes())
            board_copy = game.board.copy()
            if hashed not in board_states: 
                board_states.setdefault(hashed ,board_copy)
                
            num_moves+=1
            if done:
                total_moves+= num_moves
                print(f'{i} game done, num_moves: {num_moves} {len(board_states)}  total generated boards. total num_moves: {total_moves}  new unique boards: {len(board_states) - previous}')
                
                break
    print(f'we have generated {len(board_states)} board states. total number of moves: {total_moves} ')
    return (board_states)


def compute_more_boards_and_write():
    before = time.time()
    dictionnary = get_dict_from_files("hashes.txt", "boards.npy")
    print(f'loading the dictionnary: {time.time()-before}')
    states = compute_board_states()


    board_list = list(states.values())
    before = time.time()
    write_hashes(list(states.keys()),"hashes.txt", "w")
    during = time.time()
    write_boards(board_list , "boards.npy")
    print(f'saving the hash: {during-before}, saving the boards: {time.time()-during}')
    
def load_boards_and_evaluate(depth ,num_evaluation):


    dictionnary = get_dict_from_files("hashes.txt", "boards.npy")
    subset = random.choices(list(dictionnary.values()) , k = num_evaluation)
    agent = Minimax.minimax(depth)

    total_time = 0
   
    for i , board in enumerate(subset):
        start = time.time()
        num_pos = np.sum(board == 1)   # count of 1
        num_neg = np.sum(board == -1)  # count of -1
        turn = 1-2*( num_pos - num_neg) # if num_pos - num_neg ==0, then its player(1) turn. if num_pos - num_neg == 1, then its player(-1) turn
        agent.choose_move(board, turn)
        elapsed = time.time() - start
        total_time += elapsed
        #print(board)
        
        print(f'evaluating board {i} took {elapsed:.4f}')
    print(f"Average time per choose_move: {total_time / num_evaluation:.6f} s")   

def evaluate_boards_and_save(agent, start_index, num_boards_to_compute):
    boards_dict = get_dict_from_files("hashes.txt" , "boards.npy")
    boards = List(boards_dict.values())
    size = len(boards)
    num_boards_to_compute = min(size - start_index , num_boards_to_compute) # ensure we dont go over the limit
    to_evaluate = to_evaluate[start_index , start_index + num_boards_to_compute]
    
load_boards_and_evaluate(7, 100)
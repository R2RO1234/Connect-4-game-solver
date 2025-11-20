import json
import math
import os
from typing import List
import Minimax
from logic import Connect4
import random
import numpy as np
import zlib
import time
import tqdm
def write_boards(board_list  ,src: str):
    with open(src , "wb") as f:
         np.save(f, board_list)
       
def read_boards(scr : str):
    with open(scr , "rb") as f:
        a = np.load(f)
    return a
def write_hashes(hash_list : list[str],src : str, mode: str):
    with open(src , mode) as f:
        for h in hash_list:
            f.write(f"{h}\n")
       
def read_hashes(src : str):
    with open(src , "r") as f: 
        hashes = [int(line.strip()) for line in f]
    return hashes

def get_dict_from_files(keys_src :str , values_src: str):
    if not (os.path.exists(keys_src) and os.path.exists(values_src)): return {} # if one of the two files is missing, return empty dict
    before = time.time()
    keys = read_hashes(keys_src)
    during = time.time()
    values = read_boards(values_src)
    print(f'time: loading the hashes list: {(during-before):.3f} s, loading the Boards list: {(time.time()-during):.3f} s')
    return dict(zip(keys,values))




def compute_board_states(board_states = {}, num_games = 10000):
    '''
    Simulates a number of Connect4 games to generate unique board states.
    Parameters:
        board_states (dict, optional): Existing dictionary of hashed board states 
            in the form {hash(board): board}. Defaults to an empty dict.
        num_games (int, optional): Number of games to simulate. Defaults to 10000.

    Returns:
        dict: Updated dictionary of unique board states with their hashes as keys.

    '''
    total_moves = len(board_states)
    for i in range(num_games): # simulate num_iterations games
        player_1 = Minimax.get_random_minimax()
        player_2 = Minimax.get_random_minimax()
        
        previous = len(board_states)
        current_player = 1
        game = Connect4()
        game.reset()
        num_moves = 0
        while True:
            agent = player_1 if current_player == 1 else player_2
            
            action =  agent.choose_move(game.board , current_player)
            state, reward, done = game.make_move(action)
            current_player*=-1
            
            hashed = zlib.adler32(game.board.tobytes())
            board_copy = game.board.copy()
            if hashed not in board_states: 
                board_states.setdefault(hashed ,board_copy)
                
            num_moves+=1
            if done:
                total_moves+= num_moves
                print(f"Game {i} done | Moves: {num_moves} | Total unique boards: {len(board_states)} | Total moves: {total_moves} | New unique boards: {len(board_states) - previous}")
                break
    print(f'End of function: Total unique boards: {len(board_states)} | Total moves: {total_moves} ')
    return board_states


def generate_and_save_additional_boards(num_games, key_src , values_src):
    """
    Generates additional unique Connect4 board states and saves them to disk.

    Parameters:
        num_games (int): Number of new games to simulate for generating boards.
        key_src (str): Path to the file containing saved board hashes.
        values_src (str): Path to the file containing saved board states (.npy).

    Returns:
        None
    """
    before = time.time()
    dictionnary = get_dict_from_files(key_src, values_src)
    print(f'loading the dictionnary: {time.time()-before}')
    states = compute_board_states(dictionnary, num_games)


    board_list = list(states.values())
    before = time.time()
    write_hashes(list(states.keys()),key_src, "w")
    during = time.time()
    write_boards(board_list ,values_src)
    print(f'time: saving the hashes list: {(during-before):.3f} s, saving the Boards list: {(time.time()-during):.3f} s')



def evaluate_boards(agent, boards_subset, global_start_index):
    """
    Evaluate a contiguous block of boards.
    Returns:
        (evaluations, count, interrupted)
    """
    n = len(boards_subset)
    evaluation_result = np.empty(n, dtype=float)

    total_time = 0

    try:
        for i, board in enumerate(tqdm.tqdm(boards_subset, desc=f"Evaluating boards {global_start_index}-{global_start_index+n-1}")):
            start = time.time()

            num_pos = np.sum(board == 1)
            num_neg = np.sum(board == -1)
            turn = 1 - 2 * (num_pos - num_neg)

            evaluation_result[i] = agent.choose_move(board, turn, True)[1]

            elapsed = time.time() - start
            total_time += elapsed

        print(f"Average time for choose_move(): {total_time / n:.6f} s")

        return (evaluation_result, n)

    except KeyboardInterrupt:
        # Return partial results
        print("Interrupted inside evaluate_boards()")
        return (evaluation_result[:i], i)


def save_progression(evaluation_list , start_index: int, end_index: int, evaluation_src : str, metadata_src : str , model : Minimax.minimax): 
    combined = {
        "start_index": start_index,
        "end_index":  end_index,
        "evaluation_src" : evaluation_src,
        "model" : model.to_dict()
    }
    with open(metadata_src, "w") as f:
        json.dump(combined, f)

    with open(evaluation_src , "wb") as f:
        np.save(f, evaluation_list)


def evaluate_boards_in_batches(agent, boards, n_batches, save_directory):
    total_boards = len(boards)
    batch_size = math.ceil(total_boards / n_batches)

    os.makedirs(save_directory, exist_ok=True)

    
    for batch_id in range(n_batches):
        start_index = batch_id * batch_size
        end_index = min(start_index + batch_size, total_boards)

        if start_index >= end_index:
            break

        print(f"Processing batch {batch_id+1}/{n_batches}, boards {start_index}-{end_index-1}")

        subset = boards[start_index:end_index]
        vals, count = evaluate_boards(agent, subset, start_index)
        print("==========================================")
        real_end = start_index + count
        file_base = f"{save_directory}/evaluated_{start_index}-{real_end}"
        
        if count < len(subset):
            save_now = input("Save partial results of this batch? (y/n) ").lower() == "y"
            if save_now :
                file_base = f"{save_directory}/evaluated_{start_index}-{real_end}"
                save_progression(
                vals,
                start_index,
                real_end,
                f"{file_base}.npy",
                f"{file_base}.json",
                agent
                )
                return
            
        save_progression(
            vals,
            start_index,
            real_end,
            f"{file_base}.npy",
            f"{file_base}.json",
            agent
        )

    print("Successfully evaluated all boards.")



# files names
output_directory = "output"
save_dict_directory = os.path.join(output_directory,"dict_output")
evaluation_folder = os.path.join(output_directory,"evaluation_output_1")
boards_file_name = os.path.join(save_dict_directory,"boards.npy")
hashes_file_name =  os.path.join(save_dict_directory,"hashes.txt")


# init
def make_directories(output_directory ,save_dict_directory ):
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(save_dict_directory, exist_ok=True)


make_directories(output_directory ,save_dict_directory )
# computing the boards
generate_and_save_additional_boards(10,hashes_file_name, boards_file_name )


# Once you are satisfied with the number of boards generated, run this 
#agent = Minimax.minimax(6)
#boards_to_evaluate = list(get_dict_from_files(hashes_file_name , boards_file_name).values())

#evaluate_boards_in_batches(agent,boards_to_evaluate, 2, evaluation_folder)
    
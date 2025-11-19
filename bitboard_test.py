
import numpy as np
import time
import zlib
import Minimax

def vertical_sequence_from_cell(player_bb, mask_bb, row, col):
    """
    Returns (sequence_len, is_open)
    player_bb: bitboard of the player
    mask_bb: bitboard of all occupied squares
    row, col: starting cell (0-indexed, bottom row = 0)
    """
    COL_SHIFT = col * 6
    col_bits = (player_bb >> COL_SHIFT) & 0b111111  # extract column
    mask_bits = (mask_bb >> COL_SHIFT) & 0b111111

    # Check that starting cell is occupied by player
    if not ((col_bits >> row) & 1):
        return 0, False

    length = 1
    for r in range(row - 1, -1, -1):  # go up
        if (col_bits >> r) & 1:
            length += 1
        elif ((mask_bits >> r) & 1) == 0:  # empty → open
            return length, True
        else:
            break

    return length, False

def board_to_bitboards_and_mask(board: np.ndarray):
    """
    Converts a 6x7 numpy array board to:
      - player1_bb: bits where player 1 has pieces
      - player2_bb: bits where player -1 has pieces
      - mask_bb: bits where any piece is present
    Bottom row = least significant bits. Column-major order.
    """
    player1_bb = 0
    player2_bb = 0
    mask_bb    = 0

    for row in range(6):
        for col in range(7):
            bit_index = col * 6 + row  # column-major
            val = board[row, col]
            if val == 1:
                player1_bb |= 1 << bit_index
                mask_bb    |= 1 << bit_index
            elif val == -1:
                player2_bb |= 1 << bit_index
                mask_bb    |= 1 << bit_index

    return player1_bb, player2_bb, mask_bb

def board_to_bitboards(board: np.ndarray):
    """
    Converts a 6x7 numpy array board to two bitboards (player 1 and player -1).
    Bottom row = least significant bits.
    """
    player1_bb = 0
    player2_bb = 0
    
    for row in range(6):
        for col in range(7):
            bit_index = col * 6 + row  # column-major, bottom-to-top
            if board[row, col] == 1:
                player1_bb |= 1 << bit_index
            elif board[row, col] == -1:
                player2_bb |= 1 << bit_index
                
    return player1_bb, player2_bb
def check_in_bounds( move : tuple): 
    row, column = move
    return 0<= row <= 5 and 0<= column <= 6

def check_horizontal_sequence(board, move):
    """
    Returns (sequence_length, left_open, right_open)
    """
    row, col = move
    if row < 0 or row >= board.shape[0] or col < 0 or col >= board.shape[1]:
        raise ValueError("Move out of bounds")
    
    player = board[row, col]
    if player == 0:
        return 0, False, False

    left_open = col > 0 and board[row, col-1] == 0
    sequence = 1
    c = col + 1

    while c < board.shape[1]:
        if board[row, c] == player:
            sequence += 1
        elif board[row, c] == 0:
            return sequence, left_open, True
        else:
            break
        c += 1

    return sequence, left_open, False

def horizontal_sequence_from_cell(player_bb, mask_bb, row, col):
    """
    Returns (sequence_len, left_open, right_open) for horizontal sequences
    player_bb: bitboard of the player
    mask_bb: bitboard of all occupied squares
    row, col: starting cell (0 = bottom row)
    """
    # Extract the row bits for all columns
    # Column-major layout: bit index = col*6 + row
    bits_in_row = 0
    mask_in_row = 0
    for c in range(7):
        idx = c*6 + row
        if (player_bb >> idx) & 1:
            bits_in_row |= 1 << c
        if (mask_bb >> idx) & 1:
            mask_in_row |= 1 << c

    # Check sequence to the right
    sequence = 1
    right_open = False
    for c in range(col+1, 7):
        if (bits_in_row >> c) & 1:
            sequence += 1
        elif (mask_in_row >> c) & 1 == 0:  # empty → open
            right_open = True
            break
        else:
            break

    # Check sequence to the left
    left_open = False
    for c in range(col-1, -1, -1):
        if (bits_in_row >> c) & 1:
            sequence += 1
        elif (mask_in_row >> c) & 1 == 0:  # empty → open
            left_open = True
            break
        else:
            break

    return sequence, left_open, right_open

def check_vertical_sequence(board , move : tuple): # return the lenght of the horizontal sequence from this move to up
        # dont forget the top of the board is at 0, so go in the direction of row -1
        if not check_in_bounds(move): raise ValueError(f'move not valid: ' + move)
        row , column = move
        player = board[row,column]
        if player == 0: return (0, False)
        
        sequence =1
        row -=1
        while row >=0:
            if board[row,column] == player:
                sequence +=1
                
            elif board[row,column] == 0: 
                return (sequence,True) 
            else:
                break
            row-=1
        
        return (sequence , False) # out of bounds, thus not open

def benchmark_vertical_sequence(board, move, num_iterations=10_000_000):
    # Compute correct value with normal function
    correct = check_vertical_sequence(board, move)
    
    # --- Bitboard version ---
    player1_bb, player2_bb, mask_bb = board_to_bitboards_and_mask(board)
    start = time.time()
    for _ in range(num_iterations):
        trying = vertical_sequence_from_cell(player1_bb, mask_bb, move[0], move[1])
    bitboard_time = time.time() - start
    
    print(f'Bitboard version time: {bitboard_time:.4f} s')
    print(f'Bitboard same value: {correct == trying}')
    
    # --- Normal array version ---
    start = time.time()
    for _ in range(num_iterations):
        check_vertical_sequence(board, move)
    normal_time = time.time() - start
    
    print(f'Array version time: {normal_time:.4f} s')
    
    return bitboard_time, normal_time, correct, trying

def benchmark_horizontal_sequence_bitboard_vs_array(board, move, num_iterations=10_000_000):
    # Compute correct array result once
    correct_array = check_horizontal_sequence(board, move)
    
    # --- Array-based version ---
    start = time.time()
    for _ in range(num_iterations):
        result_array = check_horizontal_sequence(board, move)
    array_time = time.time() - start
    
    # --- Bitboard preparation ---
    player1_bb, player2_bb, mask_bb = board_to_bitboards_and_mask(board)
    
    # --- Bitboard version ---
    start = time.time()
    for _ in range(num_iterations):
        result_bb = horizontal_sequence_from_cell(player1_bb, mask_bb, move[0], move[1])
    bitboard_time = time.time() - start
    
    # Output results
    print(f'Array version time: {array_time:.4f} s')
    print(f'Bitboard version time: {bitboard_time:.4f} s')
    print(f'Array correct: {correct_array}')
    print(f'Bitboard correct: {result_bb}')
    print(f'Match: {correct_array == result_bb}')
    
    return array_time, bitboard_time, correct_array, result_bb

def benchmark_hash_vs_adler(board, num_iterations=10_000_000):
    board_bytes = board.tobytes()
    
    # --- Python hash ---
    start = time.time()
    for _ in range(num_iterations):
        h = hash(board.tobytes())
    hash_time = time.time() - start
    print(f'Python hash(board.tobytes()): {hash_time:.4f} s')
    
    # --- zlib.adler32 ---
    start = time.time()
    for _ in range(num_iterations):
        player = 1
        h = (board.tobytes() ,player)
    adler_time = time.time() - start
    print(f'boards to byte only: {adler_time:.4f} s')
    
    return hash_time, adler_time

def test_minimax(depth, board2 , num_iterations = 10):
    algo = Minimax.minimax(depth)
    total = 0
    for i in range(num_iterations):
        start = time.time()
        algo.choose_move(board2,1)
        adler_time = time.time() - start
        print(f'compute {i} board: {adler_time:.4f} s')
        total+=adler_time
    print(f'mean of compute times: {(total/num_iterations):.4f}')

board_vertical_up = np.array([
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0]
], dtype=int)
board_horizontal = np.array([
    [0, 0, 0, 0, 0,0,0],
    [0, 0, 0, 0, 0,0,0],
    [0, 0, 0, 0, 0,0,0],
    [0, 0, 0, 0, 0,0,0],
    [0,-1,-1,-1, 0,0,0],
    [0, 1, 1, 1,-1,0,0]
], dtype=int)
        
test_minimax(9, board_vertical_up , 25)    
import numpy as np


class Connect4:
    
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.reset()
    
    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1 # Player 1 = 1, Player 2 = -1
        self.winner = None
        self.game_over = 0 # 0 = not over, 1 = win, -1 = draw
        self.last_move = None
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        state = np.zeros((2, 6, 7), dtype=int)

        if self.current_player == 1:
            state[0] = (self.board == 1).astype(int) # Player 1 pieces
            state[1] = (self.board == -1).astype(int) # Player 2 pieces
        else:
            state[0] = (self.board == -1).astype(int) # Player 2 pieces
            state[1] = (self.board == 1).astype(int) # Player 1 pieces

        return state
    
    def get_valid_moves(self):
        return [col for col in range(7) if self.board[0, col] == 0]
    
    def get_lowest_open_row(self, col):
        for row in range(5, -1, -1):
            if self.board[row, col] == 0:
                return row
        return None
    
    def make_move(self, col):
        if self.game_over != 0:
            raise ValueError("Cannot make a move when game is over!")
        
        if col not in self.get_valid_moves():
            raise ValueError(f"Cannot make illegal move: {col}!")
        
        row = self.get_lowest_open_row(col)
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.move_count += 1
        reward = 0

        # check for win
        if self.check_winner(row, col, self.current_player):
            self.winner = self.current_player
            self.game_over = 1
            reward = 1
            return self.get_state(), reward, self.game_over

        # check for draw
        if self.move_count == 42:
            self.game_over = -1
            reward = 0
            return self.get_state(), reward, self.game_over

        self.current_player *= -1 # Switch player
        return self.get_state(), reward, self.game_over
    
    def check_winner(self, row, col, player):
        board = self.board
        
        # check horizontal
        for c in range(max(0, col-3), min(4, col+1)):
            if (board[row, c] == player and
                board[row, c+1] == player and
                board[row, c+2] == player and
                board[row, c+3] == player):
                return True
            
        # check vertical
        if row <= 2:
            if (board[row, col] == player and
                board[row+1, col] == player and
                board[row+2, col] == player and
                board[row+3, col] == player):
                return True
            
        # check diagonal \
        for offset in range(-3, 1):
            r = row + offset
            c = col + offset
            if 0 <= r and r <= 2 and 0 <= c and c <= 3:
                if (board[r, c] == player and
                    board[r+1, c+1] == player and
                    board[r+2, c+2] == player and
                    board[r+3, c+3] == player):
                    return True
        
        # check diagonal /
        for offset in range(-3, 1):
            r = row - offset
            c = col + offset
            if 3 <= r and r <= 5 and 0 <= c and c <= 3:
                if (board[r, c] == player and
                    board[r-1, c+1] == player and
                    board[r-2, c+2] == player and
                    board[r-3, c+3] == player):
                    return True
        
        return False
    
    def print_state(self):
        print(f'state of the board after: {self.move_count}. last move: {0-self.current_player}')
        for row in self.board:
            for element in row:
                if element ==-1:
                    element =2
                print(element , end=' ')
            print("")
        print("")
        
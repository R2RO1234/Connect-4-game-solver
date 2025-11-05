
import numpy as np
import random
Infinity = 10000
class Minimax:
    def __init__(self, depth):
        self.depth = depth
        self.saved = {}

        
        # must also do a dictionnary to check if a position was already evaluated. dictionary is the form  board : (evaluation, depth)
        # will try to do some matrix operations to chekc if its a reversal (row 0 = row 6, row 1 = row 5, etc)

        # traverse the child nodes in a random order, prioritizing the center columns

        # make function to check mandatory moves

        # parameter : past a certain evaluation value, dont explore other values. initially set at infinity
        # parameter : list of weights of the value of sequences on these columns. example: [0.7 , 0.8 , 1 ,1 , 1 , 0.8, 0.7]
        # parameter : how much the o pponent sequences are worth compared to yourself. example, 0.5 means own sequence = 4, then opponent =2
        # parameter : how much each sequence is worth. ex : [6,13] means a sequence of 2 is worth 6, and a sequence of 3 is worth 13
        # parameter : minimum depth to save board evaluation
    
        # in the dict, if this is a winning board, the suggested move is -1
        # if this is a draw, the suggested move is -2
        

    
    def evaluate_position(self,board , player_to_play): # return infinity if player =1 and he is winner, -infinity player = -1 and he is loser, 
        # only consider the open sequences in the eval function
        # the sequences that are vertical may be worth less than horizontal and vertical
        # sequences on the edges are worth less than those on the center. value is taken based on the token the more close to the center

        # if there are more than 2 open sequences of 3 in a row that can be played immediately, return immediately with the correct value
        
        # if there are a open sequence of 3 that can be played immediately for the player_to_play, return with value infintiy or -infinity

        return 0
    def check_mandatory_moves(board): 
        # returns the column in which you need to play to continue playing, otherwise one of the players will win in the current or next move
        pass


    
    def minimax(self,board,current_player,depth , last_move):

       
        data = self.is_board_saved(board, current_player) # cbeck if board was already computed
        if data is not None:
            return data[0] # if so, return the evaluation
        
        winner = self.check_winner(last_move[0] , last_move[1] , current_player * -1 , board) # check if the game is over 
        if winner: #
            self.save_evaluation(board , Infinity*current_player * -1 , 100 , -1 , current_player)
            return Infinity*current_player * -1
        
        eval = self.evaluate_position(board,current_player)
        if depth ==0: return eval
      
        moves_to_check = self.check_mandatory_moves(board)
        if moves_to_check is []:
            moves_to_check = self.get_valid_moves()

        if moves_to_check is []: # no more valid moves, that means its a draw
            print("there is a draw in this position")
            self.save_evaluation(board , 0 , 100 , -2 , current_player)
            return 0 
        
        random.shuffle(moves_to_check) # randomize the child search order
        best_evaluation = None
        best_column = None
        
        for column in moves_to_check:
            
            open_row = self.get_lowest_open_row(column, board)  
            
            board[open_row,column] = current_player
            child_value = self.minimax(board ,current_player*-1, depth-1 , (open_row , column))
            board[open_row,column] = 0

            if child_value*current_player == Infinity:  # if the current player has a infinity value, break 
                self.save_evaluation(board , child_value , 100 , column , current_player)
                return child_value
            
            if child_value * current_player > best_evaluation * current_player or best_evaluation is None: # compares both and update is necessary
                best_evaluation = child_value
                best_column = column
            elif child_value * current_player > best_evaluation * current_player and abs(best_column-3) < abs(column-3): # if same evaluation, prioritize center move
                best_evaluation = child_value
                best_column = column
               
                
            
        self.save_evaluation(board , best_evaluation , depth , best_column , current_player)
        return best_evaluation
    

    

    


    def choose_move(self , board ,current_player):# calls minimax and returns the column in which to play 
        # calls minimax on all children, checks max value and return the column in which the max value was choosen
        # and also update the board

        # does not take previous move
        pass
    

    def get_valid_moves(self, board):
        return [col for col in range(7) if board[0, col] == 0]
    
    def save_evaluation(self , board , evaluation , depth, recommended_move , current_player):
        hashed = self.fast_hash(board)
        if hashed not in self.saved or depth >= self.saved[hashed][1]:
            self.saved[hashed] = (evaluation , depth , recommended_move , current_player)
        # maybe also dont save if reversed board is already in the dictionnary
    
    def is_board_saved(self,board , current_player):
        hashed = self.fast_hash(board)
        if hashed in self.saved:
          
            data  =self.saved[hashed]
            evaluation , depth , recommended_move , current_player= data
            print(f'board already computed at depth {depth} with evaluation = {evaluation}. Player {current_player} to move at column {recommended_move}')
            return data # return the data
        
        reverseHash = self.fast_hash(self.reverseBoard(board))
        
        if reverseHash in self.saved:
           
            data  =self.saved[reverseHash]
            evaluation , depth , recommended_move , current_player= data
            
            print(f'board already computed at depth {depth} with evaluation = {evaluation}. Player {current_player} to move at column {6-recommended_move}')
            data[2] = 6 - data[2]
            return data # return the recommended move
        
        return None# not found
    
    def reverseBoard(self , board): # column 0->6, 1->5, etc
        return board[:, ::-1]

    def fast_hash(self , board: np.ndarray , current_player) -> int:
        return hash((board.tobytes() , current_player))
    
    def get_lowest_open_row(self, col, board):
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row
        return None
    def check_winner(self, row, col, player , board):
       
        
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

    # will do a method to play aginst himself, showing how it plays against himself. with sleep method if it goes too fast
    # new syntax:
        # Board = connect4
        # make NN choose move
        # update common board
        # Minimax.chooseMove(other move, board)
        # update common board
    # we also know that the board is not changed in the methods if the connect4



    


import numpy as np
import random

Infinity = 200

class minimax:
    def __init__(self, depth , distance_weights =  [1,1,1,1], epsilon = Infinity ,opponent_multiplier = 1 , sequence_values  = [1,2] ):
        self.depth = depth
        self.saved = {}
        # parameter : list of weights of the value of sequences on these columns. example: [1 ,1 , 0.8 , 0.6], and the index is the distance from the middle
        if len(distance_weights) != 4: raise ValueError("the lenght of distance_weights must be 4")
        self.distance_weights = distance_weights
        
        # parameter : past a certain evaluation value, dont explore other values. initially set at infinity
        if epsilon <0: raise ValueError("epsilon must be higher than 0") 
        self.epsilon = abs(epsilon)
        
        # parameter : how much the opponent sequences are worth compared to yourself. example, 0.5 means own sequence = 4, then opponent =2
        self.opponent_multiplier = opponent_multiplier

        # parameter : how much each sequence is worth. ex : [6,13] means a sequence of 2 is worth 6, and a sequence of 3 is worth 13
        if len(sequence_values) != 2: raise ValueError("the lenght of sequence_values must be 2")
        self.sequence_values = sequence_values

       
    
        # in the dict, if this is a winning board, the suggested move is -1
        # if this is a draw, the suggested move is -2
        

    
    def evaluate_position(self,board , player_to_play): # return infinity if player =1 and he is winner, -infinity player = -1 and he is loser, 
        # only consider the open sequences in the eval function
        # the sequences that are vertical may be worth less than horizontal and vertical
        # sequences on the edges are worth less than those on the center. value is taken based on the token the more close to the center
        # if there are more than 2 open sequences of 3 in a row that can be played immediately, return immediately with the correct value
        # if there are a open sequence of 3 that can be played immediately for the player_to_play, return with value infintiy or -infinity
        # may also consider sequences of 1. will have to check

        column = 0 
        row = len(board)-1
        winning_chances = [[False] * 7 ,[False] * 7  ] # columns in which player(-1)/player(1) play to win 
        
        opponent_chances_of_win_now = 0 # open sequence of three for the opponent
        player_chances_of_win_now = 0 #
        sequences_players = [0,0]    
        open_sequences =[[0,0],
                        [0,0]] # [seq_2_player(-1), seq_3_player(-1)] , # [seq_2_player(1), seq_3_player(1)]
        
        while column <=6: # consider changing for board[0].length-1
            if board[row,column] == 0 or row <0:
                column+=1 # change column immediately since the remaining in that column are all 0
                row = len(board)-1
                continue 
            player = board[row,column]


            sequence  = self.check_horizontal_sequence(board ,(row,column))
            sequence_len, open_left, open_right = sequence
            
            start_of_sequence = (column ==0 or board[row,column-1] != board[row,column])
            
            if start_of_sequence and sequence_len > 1 and (open_left or open_right): 
                sequence_index = sequence_len-2
                player_index = int((player+1)/2)
                open_sequences[player_index][sequence_index]+= 2 if open_right and open_left else 1

                if sequence_len == 3:
                    right_playable_now = open_right and (row == len(board)-1 or board[row+1,column+3] != 0)
                    if open_right and right_playable_now: 
                        winning_chances[player_index][column+3] = True
                        
                
                    left_playable_now  = (row == len(board)-1 or board[row+1,column-1] != 0)
                    if open_left  and left_playable_now: 
                        winning_chances[player_index][column-1]
            
            
            sequence = self.check_vertical_sequence(board ,(row,column))
            sequence_len , open_sequence = sequence
            start_of_sequence = (row == len(board)-1 or board[row+1,column] != board[row,column]) 
            
            if start_of_sequence and sequence_len > 1 and open_sequence:
                sequence_index = sequence_len-2
                player_index = int((player+1)/2)
                open_sequences[player_index][sequence_index]+=1  

                if sequence_len ==3:
                    winning_chances[player_index][column] = True
            
            
            # diagonal, top right
            sequence = self.check_diagonal_sequence(board ,(row,column),1)
            sequence_len, open_down_left, open_top_right = sequence
            start_of_sequence = (row== len(board)-1 or column == 0  or board[row+1,column-1] != board[row,column])
            
            if start_of_sequence and sequence_len >1 and (open_down_left or open_top_right):
                sequence_index = sequence_len-2
                player_index = int((player+1)/2)
                
                open_sequences[player_index][sequence_index]+= 2 if open_top_right and open_down_left else 1

                if sequence_len ==3:
                    
                    top_right_playable_now = open_top_right and board[row-2,column+3] != 0
                    if open_top_right and top_right_playable_now: 
                        winning_chances[player_index][column+3] = True
            
                    down_left_playable_now = (row+2 <=  len(board) and  (row+2 ==  len(board) or board[row+2,column-1] != 0 ))
                    if open_down_left and down_left_playable_now: 
                        winning_chances[player_index][column-1]


          
            # diagonal, top left
            sequence = self.check_diagonal_sequence(board ,(row,column),-1)
            sequence_len, open_down_right, open_top_left = sequence
            
            start_of_sequence = (row== len(board)-1 or column == 6 or board[row+1,column+1] != board[row,column]) 
            
            if start_of_sequence and sequence_len >1 and (open_down_right or open_top_left):
                sequence_index = sequence_len-2
                player_index = int((player+1)/2)

                open_sequences[player_index][sequence_index]+= 2 if open_down_right and open_top_left else 1

                if sequence_len == 3:
                    top_left_playable_now = board[row-2,column-3] != 0
                    if open_top_left and top_left_playable_now : 
                        winning_chances[player_index][column-3] = True # add additional to update chances_of_win

                
                    down_right_playable_now = open_down_right and (row+2 <= len(board) and  (row+2 ==  len(board) or board[row+2,column+1] != 0 ))
                    if open_down_right and down_right_playable_now: 
                        winning_chances[player_index][column+1]

            
            row -=1
        negative_player_immediate_win = sum(winning_chances[0])
        positive_player_immediate_win = sum(winning_chances[1])

        if player_to_play == -1 and negative_player_immediate_win != 0: return -Infinity
        if player_to_play == -1 and positive_player_immediate_win >= 2: return Infinity
        if player_to_play == 1 and positive_player_immediate_win != 0: return Infinity
        if player_to_play == 1 and negative_player_immediate_win >=2: return -Infinity

        return self.calculate_score(open_sequences , player_to_play) # will eventually have to handle the position of the sequences, to account for one of the parameters
 
    
    
    def calculate_score(self, sequence_arr , player): # will eventually have to handle the position of the sequences, to account for one of the parameters
        
        if len(sequence_arr) != 2 or len(sequence_arr[0]) !=2: raise ValueError("the arr of sequences_values is not the right size")
        if player != 1 and player != -1: raise ValueError("player must be either -1 or 1 : " + str(player))
    
        
        self.sequence_values
        negative_player_score = np.dot(sequence_arr[0] , self.sequence_values)
        positive_player_score = np.dot(sequence_arr[1] , self.sequence_values)

        if player == -1: positive_player_score*= self.opponent_multiplier 
        else: negative_player_score*= self.opponent_multiplier

        return positive_player_score - negative_player_score
    


    
    
    
    def check_mandatory_moves(self,board): # return the columns to block a open 3 sequence

        column = 0 
        row = len(board)-1
        mandatory = [False] * 7
        
        while column <=6: # consider changing for board[0].length-1
            if board[row,column] == 0 or row <0:
                column+=1 # change column immediately since the remaining in that column are all 0
                row = len(board)-1
                continue 
           
            sequence  = self.check_horizontal_sequence(board ,(row,column))
            sequence_len, open_left, open_right = sequence
            start_of_sequence = (column ==0 or board[row,column-1] != board[row,column])
            if start_of_sequence and sequence_len ==3:
                
                right_playable_now = open_right and (row == len(board)-1 or board[row+1,column+3] != 0)
                if open_right and right_playable_now: mandatory[column+3] = True
                left_playable_now  = (row == len(board)-1 or board[row+1,column-1] != 0)
                if open_left  and left_playable_now: mandatory[column-1] = True
            
            
            sequence = self.check_vertical_sequence(board ,(row,column))
            sequence_len , open_sequence = sequence
            start_of_sequence = (row == len(board)-1 or board[row+1,column] != board[row,column]) 
            if start_of_sequence and sequence_len ==3 and open_sequence:
                mandatory[column] = True
            
            # diagonal, top right
            sequence = self.check_diagonal_sequence(board ,(row,column),1)
            sequence_len, open_down_left, open_top_right = sequence
            start_of_sequence = (row== len(board)-1 or column == 0  or board[row+1,column-1] != board[row,column])
            if  start_of_sequence and sequence_len ==3:
                
                top_right_playable_now = open_top_right and board[row-2,column+3] != 0
                if open_top_right and top_right_playable_now : mandatory[column+3] = True
                
                down_left_playable_now = (row+2 <=  len(board) and  (row+2 ==  len(board) or board[row+2,column-1] != 0 ))
                if open_down_left and down_left_playable_now: mandatory[column-1] = True
            
            # diagonal, top left
            sequence = self.check_diagonal_sequence(board ,(row,column),-1)
            sequence_len, open_down_right, open_top_left = sequence
            start_of_sequence = (row== len(board)-1 or column == 6 or board[row+1,column+1] != board[row,column]) 
            if start_of_sequence and sequence_len ==3:
                
                top_left_playable_now = board[row-2,column-3] != 0
                if open_top_left and top_left_playable_now : mandatory[column-3] = True
                
                down_right_playable_now = open_down_right and (row+2 <= len(board) and  (row+2 ==  len(board) or board[row+2,column+1] != 0 ))
                if open_down_right and down_right_playable_now: mandatory[column+1] = True
            
            row -=1
        return [i for i in range(len(mandatory)) if mandatory[i] ]    

            
        



    



    
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
        if moves_to_check == []:
            moves_to_check = self.get_valid_moves(board)

        if moves_to_check == []: # no more valid moves, that means its a draw
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
            
            if best_evaluation is None or child_value * current_player > best_evaluation * current_player: # compares both and update is necessary
                best_evaluation = child_value
                best_column = column
            elif child_value * current_player > best_evaluation * current_player and abs(best_column-3) < abs(column-3): # if same evaluation, prioritize center move
                best_evaluation = child_value
                best_column = column
               
                
            
        self.save_evaluation(board , best_evaluation , depth , best_column , current_player)
        return best_evaluation
    
   
    # the spot after the sequence must be empty  
    # if the sequence is not open (next spot is occupied by opponent), return 0
    def check_vertical_sequence(self, board , move : tuple): # return the lenght of the horizontal sequence from this move to up
        # dont forget the top of the board is at 0, so go in the direction of row -1
        if not self.check_in_bounds(move): raise ValueError(f'move not valid: ' + move)
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

    def check_horizontal_sequence(self,board,move : tuple): 
        # return the lenght of the horizontal sequence from the move to the right, and 
        if not self.check_in_bounds(move): raise ValueError(f'move not valid: ' + move)
        # Return a tuple (length, left_open, right_open)
        
        row, column= move
        left_open = column != 0 and board[row,column-1] == 0
        player = board[row,column]
        if player == 0: return (0, False ,False)
        
        sequence =1
        column +=1
        while column <=6:
            if board[row,column] == player:
                sequence +=1
                
            elif board[row,column] == 0: 
                return (sequence, left_open ,True)
            
            else:
               break
            column+=1
        
        return (sequence,left_open, False) # out of bounds, so right is not open
   
    
    def check_diagonal_sequence(self,board,move : tuple,direction : int):
        if not self.check_in_bounds(move): raise ValueError(f'move not valid: {move}')
        if direction != 1 and direction != -1: raise ValueError(f'direction must be either -1 or 1: {direction}')
        
        # dont forget the top of the board is at 0, so go in the direction of row -1
        # direction influences the column
        row , column = move
        player = board[row,column]
        if player == 0: return (0, False ,False)
        
        below_open = row != 5 and 0<=column - direction <=6 and board[row+1 ,column-direction] == 0

        sequence =1
        row -=1
        column+= direction
        while row >=0 and 0 <= column <= 6:
            if board[row,column] == player:
                sequence +=1      
            elif board[row,column] == 0:
                return (sequence,below_open , True)
                
            else:
                break
            row-=1
            column+= direction
        
        return (sequence,below_open, False) # out of bounds, thus not open
    
    
    def check_in_bounds(self , move : tuple): 
        row, column = move
        if row <0 or row >7 or  column < 0 or column > 6: return False
        return True

    

    


    def choose_move(self , board ,current_player):# calls minimax and returns the column in which to play 
        # calls minimax on all children, checks max value and return the column in which the max value was choosen
        # returns -1 if the game is a win for either player, or -2 if the game is a draw

        
        #data = self.is_board_saved(board, current_player) # cbeck if board was already computed
        #if data is not None:
        #    return data[3] # if it is saved, return the recommended move
        
        
        end_of_game = self.check_winner_accross_board(board) # (win , who_won)
        winner , player_winner = end_of_game
        if winner: 
            print("there is a winner")
            #self.save_evaluation(board , Infinity*player_winner , 100 , -1 , 1)
            #self.save_evaluation(board , Infinity*player_winner , 100 , -1 , -1)
            return -1 # no move to make,
        
        
        moves_to_do = self.check_mandatory_moves(board)
        if len(moves_to_do) == 1: return moves_to_do[0]

        if len(moves_to_do) ==0: # no mandatory moves to do, switch to valid moves only
            moves_to_do = self.get_valid_moves(board)
        
        if len(moves_to_do) == 1: return moves_to_do[0]    
        
        if len(moves_to_do) == 0: 
            print("its a draw")
            self.save_evaluation(board , 0 , 100 , -2 , current_player)
            return -2
        

         
        
        eval = self.evaluate_position(board,current_player)
        
        random.shuffle(moves_to_do) # randomize the child search order
        best_evaluation = None
        best_column = None
        
        for column in moves_to_do:
            
            open_row = self.get_lowest_open_row(column, board)  
            
            board[open_row,column] = current_player
            child_value = self.minimax(board ,current_player*-1, self.depth , (open_row , column))
            board[open_row,column] = 0

            if child_value*current_player == Infinity:  # if the current player has a infinity value, break 
                self.save_evaluation(board , child_value , 100 , column , current_player)
                return column
            
            if best_evaluation is None or child_value * current_player > best_evaluation * current_player: # compares both and update is necessary
                best_evaluation = child_value
                best_column = column
            elif child_value * current_player > best_evaluation * current_player and abs(best_column-3) < abs(column-3): # if same evaluation, prioritize center move
                best_evaluation = child_value
                best_column = column
               
                
        self.saved  = {}    
        
        return best_column
        

        


        

        # does not take previous move
        pass
    

    def get_valid_moves(self, board):
        return [col for col in range(7) if board[0, col] == 0]
    
    def save_evaluation(self , board , evaluation , depth, recommended_move , current_player):
        
        
        
        hashed = self.fast_hash(board, current_player)
        if hashed not in self.saved or depth >= self.saved[hashed][1]:
            self.saved[hashed] = (evaluation , depth , recommended_move , current_player)
        # maybe also dont save if reversed board is already in the dictionnary
    
    def is_board_saved(self,board , current_player):
        hashed = self.fast_hash(board,current_player)
        if hashed in self.saved:
          
            data  =self.saved[hashed]
            evaluation , depth , recommended_move , current_player= data
            #print(f'board already computed at depth {depth} with evaluation = {evaluation}. Player {current_player} to move at column {recommended_move}')
            return data # return the data
        
        reverseHash = self.fast_hash(self.reverseBoard(board),current_player)
        
        if reverseHash in self.saved:
           
            data  =self.saved[reverseHash]
            evaluation , depth , recommended_move , current_player= data
            
            #print(f'reverse board already computed at depth {depth} with evaluation = {evaluation}. Player {current_player} to move at column {6-recommended_move}')
         
            return (evaluation , depth , 6-recommended_move , current_player) # return the recommended move
        
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
    def check_winner_accross_board(self,board):
        
        column = 0 
        row = len(board)-1
  
        
        while column <=6: # consider changing for board[0].length-1
            if board[row,column] == 0 or row <0:
                column+=1 # change column immediately since the remaining in that column are all 0
                row = len(board)-1
                continue 
           
            sequence  = self.check_horizontal_sequence(board ,(row,column))
            sequence_len, open_left, open_right = sequence
            start_of_sequence = (column ==0 or board[row,column-1] != board[row,column])
            if start_of_sequence and sequence_len >=4:
                return (True,board[row,column])
            
            
            sequence = self.check_vertical_sequence(board ,(row,column))
            sequence_len , open_sequence = sequence
            start_of_sequence = (row == len(board)-1 or board[row+1,column] != board[row,column]) 
            if start_of_sequence and sequence_len >=4:
                return (True,board[row,column])
            
            # diagonal, top right
            sequence = self.check_diagonal_sequence(board ,(row,column),1)
            sequence_len, open_left, open_right = sequence
            start_of_sequence = (row== len(board)-1 or column == 0  or board[row+1,column-1] != board[row,column])
            if start_of_sequence and sequence_len >=4:
                return (True,board[row,column])
            
            # diagonal, top left
            sequence = self.check_diagonal_sequence(board ,(row,column),-1)
            sequence_len, open_left, open_right = sequence
            start_of_sequence = (row== len(board)-1 or column == 6 or board[row+1,column+1] != board[row,column]) 
            if start_of_sequence and sequence_len >=4:
                return (True,board[row,column])
            
            row -=1
        return (False,1)

    


    # will do a method to play aginst himself, showing how it plays against himself. with sleep method if it goes too fast
    # new syntax:
        # Board = connect4
        # make NN choose move
        # update common board
        # Minimax.chooseMove(other move, board)
        # update common board
    # we also know that the board is not changed in the methods if the connect4



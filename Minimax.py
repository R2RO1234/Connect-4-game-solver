
import numpy as np
import random

Infinity = 200
TRUE_INFINITY = 1000
# find a way to generate random board states, possibly using different parameters

class minimax:
    def __init__(self, depth : int , distance_weights =  [1,1,1,1], epsilon = Infinity ,opponent_multiplier = 1 , sequence_values  = [1,2], column_to_consider = True ):
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

        # move ordering
        # bitboard representation
        # should we use a permanent dictionnary for winning and drawing states? yes for sure
        # should we use a permanent dictionnary for saving the evaluation: ? no

        
        
    
        

    
    def evaluate_position(self,board , player_to_play): # return  infinity if player(1) wins, -infinity if player(-1) wins, 
        # only consider the open sequences in the eval function
        # the sequences that are vertical may be worth less than horizontal and vertical
        # sequences on the edges are worth less than those on the center. value is taken based on the token the more close to the center
        # if there are more than 2 open sequences of 3 in a row that can be played immediately, return immediately with the correct value
        # if there are a open sequence of 3 that can be played immediately for the player_to_play, return with value infintiy or -infinity
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
                    if right_playable_now: 
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
                    if top_right_playable_now: 
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
                    if down_right_playable_now: 
                        winning_chances[player_index][column+1]

            
            row -=1
        negative_player_immediate_win = sum(winning_chances[0])
        positive_player_immediate_win = sum(winning_chances[1])

        if player_to_play == -1 and negative_player_immediate_win != 0: return -Infinity
        if player_to_play == -1 and positive_player_immediate_win >= 2: return Infinity
        if player_to_play == 1 and positive_player_immediate_win != 0: return Infinity
        if player_to_play == 1 and negative_player_immediate_win >=2: return -Infinity

        return self.calculate_score(open_sequences , player_to_play) 
 
    
    
    def calculate_score(self, sequence_arr , player): # will eventually have to handle the position of the sequences, to account for one of the parameters
        
        if len(sequence_arr) != 2 or len(sequence_arr[0]) !=2: raise ValueError("the arr of sequences_values is not the right size")
        if player != 1 and player != -1: raise ValueError("player must be either -1 or 1 : " + str(player))
    
        
        self.sequence_values
        negative_player_score = np.dot(sequence_arr[0] , self.sequence_values)
        positive_player_score = np.dot(sequence_arr[1] , self.sequence_values)

        if player == -1: positive_player_score*= self.opponent_multiplier 
        else: negative_player_score*= self.opponent_multiplier

        return positive_player_score - negative_player_score
    


    
    
    
    def check_mandatory_moves(self,board , valid_moves): # return the columns to block a open 3 sequence
        # alternate: get all the valid moves, play them all on both players, and check if there is a winner. Dont know if it will be fast
        # valid moves is an int array with column[open row]
        
        mandatory = []
        for col, row in enumerate(valid_moves):
            if row >= 0:  # column not full
                # simulate move
                board[row, col] = 1
                if check_winner(board, row, col, 1):
                    mandatory.append(col)
                board[row, col] = 0  # undo

                board[row, col] = -1
                if check_winner(board, row, col, -1):
                    mandatory.append(col)
                board[row, col] = 0  # undo
        return mandatory

    



    
    def minimax(self,board,current_player,depth , last_move , alpha, beta , possible_moves):

        
        if (data := self.is_board_saved(board, current_player)) is not None:# cbeck if board was already computed
            return data[0] # if so, return the evaluation
        
        winner = check_winner(board , last_move[0] , last_move[1] , current_player * -1 ) # check if the game is over 
        if winner: #
            self.save_evaluation(board , Infinity*current_player * -1 , 100 , -1 , current_player  )
            return Infinity*current_player * -1
        
        #eval = self.evaluate_position(board,current_player)
        if depth ==0: return self.evaluate_position(board,current_player)
   
        moves_to_check = self.check_mandatory_moves(board , possible_moves)
        if moves_to_check == []:
            moves_to_check = self.get_valid_moves(possible_moves)

        if moves_to_check == []: # no more valid moves, that means its a draw
            self.save_evaluation(board , 0 , 100 , -2 , current_player )
            return 0 
        
        random.shuffle(moves_to_check) # randomize the child search order
        best_evaluation = None
        best_column = None
        
        for column in moves_to_check:
            
            open_row = possible_moves[column]
            
            board[open_row,column] = current_player
            possible_moves[column] -=1
            child_value = self.minimax(board ,current_player*-1, depth-1 , (open_row , column) , alpha , beta, possible_moves)
            board[open_row,column] = 0
            possible_moves[column] +=1
            
            me = child_value * current_player
            

            if me == Infinity:  # if the current player has a infinity value, break 
                self.save_evaluation(board , child_value , 100 , column , current_player)
                return child_value
            
            if best_evaluation is None or me > best_evaluation * current_player: # compares both and update is necessary
                best_evaluation = child_value
                best_column = column
            elif me == best_evaluation * current_player and abs(best_column-3) < abs(column-3): # if same evaluation, prioritize center move
                best_column = column
            
            if current_player == 1:
                alpha = max(alpha, child_value)
            else:
                beta = min(beta, child_value)

            if alpha >= beta:
                break
                        
                    
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
        return 0<= row <= 5 and 0<= column <= 6
      

    

    


    def choose_move(self , board ,current_player , return_eval = False):# calls minimax and returns the column in which to play 
        # calls minimax on all children, checks max value and return the column in which the max value was choosen
        # returns -1 if the game is a win for either player, or -2 if the game is a draw
        
        self.saved  = {} # if we dont reset it everytime, It will be bad  
        
        end_of_game = self.check_winner_accross_board(board) # (win , who_won)
        winner , player_winner = end_of_game
        if winner: 
            if return_eval: return (-1 , player_winner)
            return -1 # no move to make,
        
        possible_moves = compute_open_rows(board)

      
        moves_to_do = self.check_mandatory_moves(board,possible_moves)
        # what to do here? either call choose move on this one and return the evaluation. or return minimax
        
        if len(moves_to_do) ==0: # no mandatory moves to do, switch to valid moves only
            moves_to_do = self.get_valid_moves(possible_moves)


        if len(moves_to_do) == 1: 
            if not return_eval: return moves_to_do[0]

            move = moves_to_do[0]        # single column index
            row = possible_moves[move]   # row to play in that column
            column = move

            board[row,column] = current_player
            possible_moves[column] -=1
            child_value = self.minimax(board ,current_player*-1, self.depth , (row , column), -TRUE_INFINITY , TRUE_INFINITY,possible_moves)
            possible_moves[column] +=1
            board[row,column] = 0
            return (moves_to_do[0] , child_value)  
        
        
        if len(moves_to_do) == 0: 
            self.save_evaluation(board , 0 , 100 , -2 , current_player)
            if return_eval: return (-2 , 0)
            return -2 
        

        alpha,beta = -TRUE_INFINITY , TRUE_INFINITY
        
        eval = self.evaluate_position(board,current_player)
        
        random.shuffle(moves_to_do) # randomize the child search order
        best_evaluation = None
        best_column = None
        
        for column in moves_to_do:


            open_row = possible_moves[column] # open row assumes its legal to play
            
            board[open_row,column] = current_player
            possible_moves[column] -=1
            child_value = self.minimax(board ,current_player*-1, self.depth , (open_row , column),alpha , beta,possible_moves)
            possible_moves[column] +=1
            board[open_row,column] = 0

            me = child_value*current_player

            if me == Infinity:  # if the current player has a infinity value, break 
                self.save_evaluation(board , child_value , 100 , column , current_player)
                if return_eval: return (column , child_value)
                return column
            
            if best_evaluation is None or me > best_evaluation * current_player: # compares both and update is necessary
                best_evaluation = child_value
                best_column = column
            elif me > best_evaluation * current_player and abs(best_column-3) > abs(column-3): # if same evaluation, prioritize center move
                best_evaluation = child_value
                best_column = column
            
            if current_player == 1:
                alpha = max(alpha, child_value)
            else:
                beta = min(beta, child_value)

            if alpha >= beta:
                print("in choose move alpha was chosen")
                break
            
               
                
        
        if return_eval: return (best_column , best_evaluation)
        
        return best_column
        

        


        

        # does not take previous move
        pass
    

    def get_valid_moves(self, row_index):
        return [col for col in range(7) if row_index[col] >=0] # same format as mandatory moves
    

    def save_evaluation(self , board , minimax_evaluation , depth, recommended_move , current_player):
        
        
        
        hashed = fast_hash(board, current_player)
        if hashed not in self.saved or depth >= self.saved[hashed][1]:
            self.saved[hashed] = (minimax_evaluation , depth , recommended_move , current_player)
        # maybe also dont save if reversed board is already in the dictionnary
    
    def is_board_saved(self,board , current_player):
        hashed = fast_hash(board,current_player)
        if hashed in self.saved:
          
            data  = self.saved[hashed]
            #minimax_evaluation , depth , recommended_move , current_player ,evaluation_function_score = data
            #print(f'board already computed at depth {depth} with evaluation = {evaluation}. Player {current_player} to move at column {recommended_move}')
            return data # return the data
        
        reverseHash = fast_hash(reverseBoard(board),current_player)
        
        if reverseHash in self.saved:
           
            data  =self.saved[reverseHash]
            minimax_evaluation , depth , recommended_move , current_player= data
            
            #print(f'reverse board already computed at depth {depth} with evaluation = {evaluation}. Player {current_player} to move at column {6-recommended_move}')

            return (minimax_evaluation , depth , 6-recommended_move , current_player) # return the recommended move
        
        return None# not found
    
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
    
    def to_dict(self):
        return {
            "depth": self.depth,
            "distance_weights": self.distance_weights,
            "epsilon": self.epsilon,
            "opponent_multiplier": self.opponent_multiplier,
            "sequence_values": self.sequence_values
        }
    @classmethod
    def from_dict(cls, d):
        return cls(
            depth=d["depth"],
            distance_weights=d["distance_weights"],
            epsilon=d["epsilon"],
            opponent_multiplier=d["opponent_multiplier"],
            sequence_values=d["sequence_values"]
        )



def reverseBoard(board): # column 0->6, 1->5, etc
    return board[:, ::-1]




def check_winner(board , row, col, player):  
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
   
def compute_open_rows(board):
    return [get_lowest_open_row(i , board) for i in range(7)]   
    
def get_closest_from_center(column1, column2):
    return min(get_distance_from_center(column1) ,get_distance_from_center(column2) )
def get_farthest_from_center(column1, column2):
    return max(get_distance_from_center(column1) ,get_distance_from_center(column2) )

def get_distance_from_center(column):
        return abs(column-3)
def get_lowest_open_row(col, board):
    for row in range(5, -1, -1):
        if board[row, col] == 0:
            return row
    return -1

def fast_hash(board: np.ndarray , current_player) :
        return hash((board.tobytes() , current_player))

def get_random_minimax():
    depth = random.randint(1,2)
    opponent_multiplier = random.uniform(0.2,3)
    sequence_values = [random.randint(1,10) , random.randint(1,10)]
    column_to_consider = bool(random.getrandbits(1))
    distance_weights = [random.uniform(0.1,3) for _ in range(4)]

    mini = minimax(depth)
    mini.opponent_multiplier = opponent_multiplier
    mini.sequence_values = sequence_values
    mini.column_to_consider = column_to_consider
    mini.distance_weights = distance_weights

    return mini




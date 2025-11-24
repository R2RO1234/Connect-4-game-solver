
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import Minimax
import numpy as np
from pydantic import BaseModel
from enum import Enum
from logic import Connect4
from fastapi.middleware.cors import CORSMiddleware

class PlayerType(Enum):
    USER = 1
    CNN = 2
    MINIMAX = 3


app = FastAPI()

model = Minimax.minimax(2)
model1 = Minimax.minimax(10)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class RequestMove(BaseModel):
    model: PlayerType
    board: list[list[int]]




# params: type of the model to choose move, board,
# player to play is already known: count number of 1 and -1
# must convert board since its inversed and also convert to numpy
# returns the new move, and the result of the move: or 3 for draw , 1 for player(1) win, or 2 for player(-1) win, 0 for nothing
@app.post("/api/getMove")
def getMove(request : RequestMove):
    board = convertArray(request.board)
    
   
    model_enum = PlayerType(request.model)
    
    winner = model.check_winner_accross_board(board) #return (True,board[row,column])
    if winner[0]: 
        if winner[1] == -1: return {"column" : -1, "flag" : 2}
        return  {"column" : -1, "flag" :1 }
    
    if not get_valid_moves(board): return {"column" : -1 , "flag" : 3}

    computing = model1

    if(model_enum == PlayerType.CNN):
        print("model_enum is a CNN")
        computing  = model
    
    elif model_enum == PlayerType.USER : print("its not supposed to be user")

    elif model_enum == PlayerType.MINIMAX: print("model_enum is a minimax")
    
    to_play = get_player_to_play(board)
    col = computing.choose_move(board,to_play)
    row = Minimax.get_lowest_open_row(col, board)
    board[row,col] = to_play
    
    flag = 0 # nothing
    if Minimax.check_winner(board , row, col, to_play): flag =  (1 - to_play) / 2 + 1 # somebody won
    elif not get_valid_moves(board): flag = 3 # a draw
    return  {"column" :col , "flag" : flag}



frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="frontend")


def convertArray(arr : list[list[int]]):
    return np.transpose(arr)

def get_player_to_play(board):
    num_pos = np.sum(board == 1)
    num_neg = np.sum(board == -1)
    return 1 - 2 * (num_pos - num_neg)

def get_valid_moves(board):
    return [col for col in range(7) if board[0, col] == 0]
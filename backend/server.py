
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import Minimax
from pydantic import BaseModel
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
import agent
import random


def get_decent_minimax():
    model = Minimax.minimax(random.randint(9,11))
    return model
class PlayerType(Enum):
    USER = 0
    MINIMAX = 1
    CNN = 2


app = FastAPI()

model  =get_decent_minimax()
ai_model = agent.AIPlayer("connect4_checkpoint_50.pth")



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
    board = Minimax.convertArray(request.board)
    
    global model
    model_enum = PlayerType(request.model)
    
    winner = model.check_winner_accross_board(board) 
    if winner[0]: 
        model  = get_decent_minimax()
        if winner[1] == -1: return {"column" : -1, "flag" : 2}
        return  {"column" : -1, "flag" :1 }
    
    if not Minimax.get_valid_moves(board): return {"column" : -1 , "flag" : 3}

    computing = model

    if(model_enum == PlayerType.CNN):
        print("model_enum is a CNN")
        computing  = ai_model
    
    elif model_enum == PlayerType.USER : print("its not supposed to be user")

    elif model_enum == PlayerType.MINIMAX: print("model_enum is a minimax")
    
    to_play = Minimax.get_player_to_play(board)
    col = computing.choose_move(board,to_play)
    row = Minimax.get_lowest_open_row(col, board)
    board[row,col] = to_play
    
    flag = 0 # nothing
    if Minimax.check_winner(board , row, col, to_play):
        model  =get_decent_minimax()
        flag =  (1 - to_play) / 2 + 1 # somebody won
    elif not Minimax.get_valid_moves(board): 
        flag = 3 # a draw
        model  =get_decent_minimax()
    return  {"column" :col , "flag" : flag}



frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="frontend")


print("server running on http://127.0.0.1:8000")



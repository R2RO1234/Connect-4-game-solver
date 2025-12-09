
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
import asyncio
from contextlib import asynccontextmanager


def get_decent_minimax():
    model = Minimax.minimax(8)
    return model
class PlayerType(Enum):
    USER = 0
    MINIMAX = 1
    CNN = 2



ai_model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_model
    loop = asyncio.get_running_loop()
    ai_model = await loop.run_in_executor(
        None, agent.AIPlayer, "connect4_checkpoint_50.pth"
    )
    print("CNN AI model loaded")
    yield
    # optional: cleanup code here

app = FastAPI(lifespan=lifespan)
dummy_model = Minimax.minimax(1)

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
    
   
    model_enum = PlayerType(request.model)
    
    winner = dummy_model.check_winner_accross_board(board) 
    if winner[0]: 
        
        if winner[1] == -1: return {"column" : -1, "flag" : 2}
        return  {"column" : -1, "flag" :1 }
    
    if not Minimax.get_valid_moves(board): return {"column" : -1 , "flag" : 3}

    

    if(model_enum == PlayerType.CNN):
        print("model_enum is a CNN")
        if ai_model is None:
        # fallback: either return an error or use minimax instead
            print("CNN not ready yet, using fallback minimax")
            computing = get_decent_minimax()
        else:
            computing = ai_model
    
    elif model_enum == PlayerType.USER : print("its not supposed to be user")

    elif model_enum == PlayerType.MINIMAX: 
        print("model_enum is a minimax")
        computing  = get_decent_minimax()
    
    to_play = Minimax.get_player_to_play(board)
    col = computing.choose_move(board,to_play)
    row = Minimax.get_lowest_open_row(col, board)
    board[row,col] = to_play
    
    flag = 0 # nothing
    if Minimax.check_winner(board , row, col, to_play):
        flag =  (1 - to_play) / 2 + 1 # somebody won
    elif not Minimax.get_valid_moves(board): 
        flag = 3 # a draw
    return  {"column" :col , "flag" : flag}



frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="frontend")


print("server running on http://127.0.0.1:8000")



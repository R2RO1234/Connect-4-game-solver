import neural_network
from logic import Connect4
import Minimax
import random
import torch as th
def play(player1 , player2):
    game = Connect4()
    state = game.reset()
    player = 1
     # Randomly decide who goes first
    current_player = "player1"
    #if random.random() < 0.5:
    #    current_player = "player1"
    #    print("player1 plays first")
    #else:
    #    current_player = "player2"
    #    print("player2 plays first")

    while True:
        game.print_state()

        agent = player1 if current_player =="player1" else player2

        if type(agent) is Minimax.minimax:
            action =  agent.choose_move(game.board , player)
        elif type(agent) is neural_network.DQNAgent:
            action = agent.select_action(game , game.get_valid_moves)

        else: 
            col = input("enter the index of the column you want to play in (0-7). enter 'exit' to stop playing ")
            if(col == "exit"): break
            valid_moves = game.get_valid_moves()
            try:
                action = int(col)
            except:
                continue

        state, reward, done = game.make_move(action)
        player *=-1
    
        if done:
        
            winner = 'player1 winning' if reward == '1' and current_player == 'player1' else 'player2 winning'
            if len(game.get_valid_moves()) == 0: winner = "a draw"
            game.print_state()

            play_again = input(f'game ended with {winner}. do you want to play again? (y/n)? ')
            if(play_again == 'y'):
                state = game.reset()
            else: return
        
        current_player = 'player1' if current_player == 'player2' else 'player2'






if __name__ == "__main__":

    #trained_agent = neural_network.train_dqn_agent()
    player1 = Minimax.minimax(0)
    player2 = Minimax.minimax(6)
    #neural_network.th.save(trained_agent.state_dict() , "saved_model.pth")
    play(player2, True)

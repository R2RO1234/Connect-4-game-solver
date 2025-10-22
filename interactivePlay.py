import neural_network
from logic import Connect4
import random
import torch as th
def play(agent):
    
    game = Connect4()
    state = game.reset()
     # Randomly decide who goes first
    if random.random() < 0.5:
        current_player = 'agent'
        print("the agent plays first")
    else:
        current_player = 'random'
        print("the agent plays second")

    while True:
        game.print_state()
        
        if(current_player == 'agent'): # Agent's turn
             
                valid_moves = game.get_valid_moves()
                action = agent.select_action(state, valid_moves)
                # Force exploitation (no exploration during test)
                agent.epsilon = 0
        
        else: # Player's turn
            col = input("enter the index of the column you want to play in (0-7). enter 'exit' to stop playing")
            if(col == "exit"): break
            valid_moves = game.get_valid_moves()
            try:
                action = int(col)
            except:
                continue

            if action not in valid_moves: continue
        
        state, reward, done = game.make_move(action)
    
        
        if done:
        
            winner = 'agent winning' if reward == '1' and current_player == 'agent' else 'player winning'
            game.print_state()
            play_again = input(f'game ended with {winner}. do you want to play again? (y/n)?')
            if(play_again == 'y'):
                state = game.reset()
            else: return
        
        current_player = 'player' if current_player == 'agent' else 'agent'
        
        

        






if __name__ == "__main__":

    trained_agent = neural_network.train_dqn_agent()
   
    #neural_network.th.save(trained_agent.state_dict() , "saved_model.pth")
    play(trained_agent)

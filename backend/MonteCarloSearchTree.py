###############################################
# Monte Carlo Tree Search (MCTS) Architecture #
###############################################

import math

import numpy as np
from logic import Connect4


class Node:
    """Notation:
    - N(s,a): # times node visited
    - W(s,a): total value accumulated
    - P(s,a): policy network previous probability
    """

    def __init__(self, prev):
        self.visit_count = 0 # N(s,a)
        self.val = 0         # W(s,a)
        self.prev = prev     # P(s,a)
        self.children = {}

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.val / self.visit_count

    def has_children(self):
        return len(self.children) > 0


class MCTS:
    """Monte Carlo Tree Search. NN-guided selection and expansion phase. Uses PUCT formula.
    Additional notation:
    - Q(s,a): mean action value = W(s,a) / N(s,a)
    - c_puct: exploration constant
    """

    def __init__(self, network, c_puct=1.5, num_simulations=200):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, game, temp=1.0):
        root = Node(prev=0)
        # Initialize root with NN policy probabilities P(s,a)
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        policy, _ = self.network.predict(state, valid_moves)

        for action in valid_moves:
            root.children[action] = Node(prev=policy[action])

        # MCTS simulations
        for _ in range(self.num_simulations):
            node = root
            sim_game = self.copy_game(game)
            total_path = [node]

            # Selection using PUCT
            while node.has_children() and not sim_game.game_over:
                action, node = self.PUCT_select(node)
                sim_game.make_move(action)
                total_path.append(node)

            # Expansion + Evaluation
            if sim_game.game_over:
                if sim_game.game_over == 1:
                    value = -1 # current player loss
                else:
                    value = 0  # draw
            else:
            # NN evaluation when game not over
                state = sim_game.get_state()
                valid_moves = sim_game.get_valid_moves()
                policy, value = self.network.predict(state, valid_moves)

                # Add children with prior probabilities P(s,a) from policy network
                for action in valid_moves:
                    node.children[action] = Node(prev=policy[action])

            self.update_path(total_path, value) # back propagate the path and update visited node attributes

        return self.get_probs(root, valid_moves, temp) # return new move probabilities based on visit counts

    def copy_game(self, game):
        new_game = Connect4()
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        new_game.game_over = game.game_over
        new_game.winner = game.winner
        new_game.move_count = game.move_count
        return new_game

    def PUCT_select(self, node):
        """Return action and child with best PUCT score"""
        best_score = float('-inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = self.PUCT(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def PUCT(self, parent, child): # PUCT formula
        prev_score = (self.c_puct * child.prev * math.sqrt(parent.visit_count) / (1 + child.visit_count))
        value_score = -child.value() if child.visit_count > 0 else 0 # value_score = Q(s,a)
        return value_score + prev_score

    def update_path(self, search_path, value):
        """Backpropagate W(s,a) and N(s,a) through search path."""
        for node in reversed(search_path):
            node.val += value
            node.visit_count += 1
            value = -value

    def get_probs(self, root, valid_moves, temp):
        """Compute updated move probabilities from visit counts. Formula from the AlphaGo Zero paper: π(a) ∝ N(s,a)^(1/t) with temp = t"""
        visits = np.array([root.children[a].visit_count if a in root.children else 0 for a in range(7)])

        if temp == 0:
            probs = np.zeros(7)
            probs[np.argmax(visits)] = 1
        else:
            visits_temp = visits ** (1 / temp)
            probs = visits_temp / visits_temp.sum()

        return probs
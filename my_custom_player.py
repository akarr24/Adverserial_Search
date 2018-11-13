## Monte Carlo Tree Search
## The following resources were used to learn MCTS
## https://towardsdatascience.com/monte-carlo-tree-search-158a917a8baa
## https://www.baeldung.com/java-monte-carlo-tree-search
## https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
from sample_players import DataPlayer
import copy
import random 
import math

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        
        if state.ply_count < 2:          #to initiate Monte Carlo simulations; Comment while running alpha beta search
            self.queue.put(random.choice(state.actions()))
        else:
            self.queue.put(self.mcts(state))
        
        #depth = 1 #to run alpha beta search; comment while running MCTS
        #while 1:
            #self.queue.put(self.alpha_beta_search(state, depth))
            #depth += 1
        
    def mcts(self, state):
        root = MCTS(state)
        if root.state.terminal_test():
            return random.choice(state.actions())
        for i in range(iterations):
            child = tree_policy(root)
            if not child:
                continue
            reward = default_policy(child.state)
            backup(child, reward)
    
        id = root.children.index(best_child(root))
        return root.actions[id]
        
        #self.queue.put(random.choice(state.actions()))
        
    def alpha_beta_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        actions = state.actions()
        if actions:
            best_move = actions[0]
        else:
            best_move = None
        player_id = 0
        for a in actions:
            new_state = state.result(a)
            v = self._alpha_beta_search(
                new_state, depth-1, alpha, beta, player_id)
            if v > alpha:
                alpha = v
                best_move = a
        return best_move

    def _alpha_beta_search(self, state, depth, alpha, beta, player_id):
        if state.terminal_test():
            return state.utility(self.player_id)
        if depth <= 0:
            return self.score(state)
        if player_id:
            v = -float('inf')
            for action in state.actions():
                new_state = state.result(action)
                v = max(v, self._alpha_beta_search(
                    new_state, depth-1, alpha, beta, player_id ^ 1))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:
            v = float('inf')
            for action in state.actions():
                new_state = state.result(action)
                v = min(v, self._alpha_beta_search(
                    new_state, depth-1, alpha, beta, player_id ^ 1))
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v
    
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

class MCTS():
    def __init__(self, state, parent = None):
        self.reward = 0.0       # reward value
        self.state = state     
        self.visits = 1         # visit value
        self.children = []      #empty list to store nodes
        self.actions = []       # empty list to store children action 
        self.parent = parent
        
    def add_children(self, child_state, action):
        child = MCTS(child_state, self)     #creating a child node
        self.children.append(child)         #adding child to children list
        self.actions.append(action)         #adding children action to action list
        
    def exploration_complete(self):
        '''
        returns true when all nodes are explored
        '''
        return len(self.actions) == len(self.state.actions())
    
    def update(self, reward):
        '''
        Updates reward and visit values
        '''
        self.reward += reward
        self.visits +=  1

Factor = 1.0
iterations = 100




def tree_policy(node):
    '''
    Gives a leaf node
    returns an unexplored child node if not fully explored
    Otherwise, returns the child with best score
    '''
    while not node.state.terminal_test():
        if not node.exploration_complete():
            return expand(node)
        node = best_child(node)
    return node

def expand(node):
    tried = node.actions
    legal_actions = node.state.actions()
    for action in legal_actions:
        if action not in tried:
            new_state = node.state.result(action)
            node.add_children(new_state, action)
            return node.children[-1]
    
def best_child(node):
    '''
    Child with best score
    '''
    best_score = float("-inf")
    best_children = []
    for child in node.children:
        exploit = child.reward / child.visits
        explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
        score = exploit + Factor * explore
        if score == best_score:
            best_children.append(child)
        elif score > best_score:
            best_children = [child]
            best_score = score
    return random.choice(best_children)

def default_policy(state):
    '''
    Search the state descendent randomly and return the reward
    '''
    
    initial_state = copy.deepcopy(state)
    while not state.terminal_test():
        action = random.choice(state.actions())
        state = state.result(action)
        
    if state._has_liberties(initial_state.player()):
        return -1
    else:
        return 1
    
def backup(node, reward):
    '''
    Using the result to update the information in the node present in path
    '''
    while node != None:
        node.update(reward)
        node = node.parent
        reward *= -1      


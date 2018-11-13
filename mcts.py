## Monte Carlo Tree Search
## The following resources were used to learn MCTS
## https://towardsdatascience.com/monte-carlo-tree-search-158a917a8baa
## https://www.baeldung.com/java-monte-carlo-tree-search
## https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
import random
import math
import copy

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
        reward *= 1
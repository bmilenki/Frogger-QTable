import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 2, self.frog_y - 1) or '_',  # row infront
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
            self.get(self.frog_x + 2, self.frog_y - 1) or '_',
            self.get(self.frog_x - 2, self.frog_y - 2) or '_',  # row 2 infront
            self.get(self.frog_x - 1, self.frog_y - 2) or '_',
            self.get(self.frog_x, self.frog_y - 2) or '_',
            self.get(self.frog_x + 1, self.frog_y - 2) or '_',
            self.get(self.frog_x + 2, self.frog_y - 2) or '_',
            self.get(self.frog_x - 2, self.frog_y) or '_',  # current frog row
            self.get(self.frog_x - 1, self.frog_y) or '_',
            self.get(self.frog_x, self.frog_y) or '_',
            self.get(self.frog_x + 1, self.frog_y) or '_',
            self.get(self.frog_x + 2, self.frog_y) or '_',
            self.get(self.frog_x - 2, self.frog_y + 1) or '_',  # behind current frog row
            self.get(self.frog_x - 1, self.frog_y + 1) or '_',
            self.get(self.frog_x, self.frog_y + 1) or '_',
            self.get(self.frog_x + 1, self.frog_y + 1) or '_',
            self.get(self.frog_x + 2, self.frog_y + 1) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()

        # learning params
        self.alpha = 0.1  # default =.1
        self.gamma = 0.6  # default =.6
        self.epsilon = 0  # default = .1 - changed to 0 for compet. so it only does best moves.

        self.prevState = None
        self.prevAction = None

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.

        The initial implementation of this method is simply a random
        choice among the possible actions. You will need to augment
        the code to implement Q-learning within the agent.
        '''
        currQState = Q_State(state_string)
        currStateKey = currQState.key
        if currStateKey not in self.q:
            self.addNewStateToQTable(currStateKey)

        # first run
        if self.prevState is None:

            bestAction = self.findBestAction(currStateKey)
            self.prevState = currQState
            self.prevAction = bestAction

            return bestAction

        # subsequent runs
        else:
            bestAction = self.qLearning(state_string)
            self.prevState = currQState
            self.prevAction = bestAction
            return bestAction

        #return random.choice(State.ACTIONS)

    def qLearning(self, state_string):
        # Will return the best action for the state
        currQState = Q_State(state_string)
        currStateKey = currQState.key


        if currStateKey in self.q:
            action = self.findBestAction(currStateKey)
            # executed previous action A, now update Qtable for that action
            # S’ = new state after executing A
            sPrime = currStateKey
            # R =  observed reward
            R = currQState.reward()
            prevStateKey = self.prevState.key
            prevStateAction = self.prevAction

            # Update Q table
            self.updateQTable(sPrime,action,prevStateKey,prevStateAction,R)

        return action

    def addNewStateToQTable(self, stateKey):
        self.q[stateKey] = [0,0,0,0,0]
        self.save()

    def updateQTable(self, currStateKey, currStateAction, prevStateKey, prevStateAction, R):
        prevActionIndex = self.actionToIndex(self.prevAction)
        prevStateKey = self.prevState.key
        maxOfCurrActions = max(self.q[currStateKey])

        newQValue = (1-self.alpha)*(self.q[prevStateKey][prevActionIndex]) + self.alpha*(R + self.gamma*maxOfCurrActions)
        self.q[prevStateKey][prevActionIndex] = newQValue

        self.save()

    def pickBestAction(self,stateKey):
        actionValues = self.q[stateKey]
        return State.ACTIONS[actionValues.index(max(actionValues))]

    def pickRandomAction(self):
        return random.choice(State.ACTIONS)

    def actionToIndex(self,actionString):
        # ACTIONS = ['u', 'd', 'l', 'r', '_']
        if actionString == 'u':
            return 0
        elif actionString == 'd':
            return 1
        elif actionString == 'l':
            return 2
        elif actionString == 'r':
            return 3
        elif actionString == '_':
            return 4

    def findBestAction(self,stateKey):
        # choose action based on Q table and current state S
        rand = random.randrange(0, 100)
        if rand <= self.epsilon * 100:
            # pick random
            action = self.pickRandomAction()
        else:
            # pick best action
            action = self.pickBestAction(stateKey)

        return action

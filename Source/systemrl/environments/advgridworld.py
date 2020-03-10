import numpy as np
import pickle

class AdvGridworld:

    def __init__(self):
        self.boardDim = ""
        self.startState = ""
        self.endState = ""
        self.fT = ""
        self.obst = ""
        self.keyLoc = ""
        self.movements = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'upri', 5: 'dori',
                 6: 'dole', 7: 'uple', 8: 'use', 9: 'break'}
        self._currentState = None
        self._rewards = 0
        self._inTerminal = False
        self.getGrid(2)
        self._name = "Advanced Gridworld"
        self._action = None
        self._gamma = 1
        #Turn stochastic on (True) or off (False)
        self.stoch = False
        #Key mechanic
        self.hasKey = False

    @property
    def name(self) -> str:
        return self._name

    '''
    getBoard method takes a numerical input to open a saved file with the grid information.
    The file information is parsed and passed to the generateBoard method which creates the matrix of
    information that is required to play.
    '''
    def getGrid(self, selection):
        fileName = ""
        if selection == 1:
            fileName = "grids/gridworld1.p"
        elif selection == 2:
            fileName = "grids/gridworld2.p" #TODO: implement
        else:
            fileName = "grids/gridworld3.p" #TODO: implement
        gridData = pickle.load(open(fileName, "rb"))
        self.generateGrid(gridData)

    '''
    generateGrid method takes an array of strings that make up the gridworld layout.
    The information within can define obstacles, barriers, and objects that require actions
    to interact with (eg. jump or hit).
    '''
    def generateGrid(self, gridInfo):
        self.boardDim = gridInfo["size"]
        self.startState = gridInfo["start"]
        self._currentState = self.startState
        self.endState = gridInfo["end"]
        fT = {}
        fT["left"] = gridInfo["left"]
        fT["right"] = gridInfo["right"]
        fT["up"] = gridInfo["up"]
        fT["down"] = gridInfo["down"]
        fT["upri"] = gridInfo["upri"]
        fT["dori"] = gridInfo["dori"]
        fT["dole"] = gridInfo["dole"]
        fT["uple"] = gridInfo["uple"]
        self.fT = fT
        self.obst = gridInfo["obs"]
        self.keyLoc = gridInfo["key"]
        self.doorLoc = gridInfo["door"]

    '''
    step will try to take a given action. If the action is unavailable, the player will
    keep the same position. For training purposes, movements are specified by numbers.
    This will likely be changed in the future as the combinations will get too large with
    additional action combinations.
    0 - Up
    1 - Right
    2 - Down
    3 - Left
    4 - Up+Right
    5 - Down+Right
    6 - Down+Left
    7 - Up+Left
    '''
    def step(self, poten_action):
        if self.stoch:
            action = self.alterMove(poten_action)
        else:
            action = poten_action
        stepReward = 0
        self._action = action
        actionCheck = self.movements[action]
        if action in [8, 9] or self._currentState not in self.fT[actionCheck] and not self._inTerminal:
            newState = ""
            #Up
            if action == 0:
                curState = self._currentState
                newState = [curState[0], curState[1] - 1]
            elif action == 1:
            #Right
                curState = self._currentState
                newState = [curState[0] + 1, curState[1]]
            #Down
            elif action == 2:
                curState = self._currentState
                newState = [curState[0], curState[1] + 1]
            #Left
            elif action == 3:
                curState = self._currentState
                newState = [curState[0] - 1, curState[1]]
            #Up and Right
            elif action == 4:
                curState = self._currentState
                newState = [curState[0] + 1, curState[1] - 1]
            #Down and Right
            elif action == 5:
                curState = self._currentState
                newState = [curState[0] + 1, curState[1] + 1]
            #Down and Left
            elif action == 6:
                curState = self._currentState
                newState = [curState[0] - 1, curState[1] + 1]
            #Up and Left
            elif action == 7:
                curState = self._currentState
                newState = [curState[0] - 1, curState[1] - 1]
            #Key Pick Up & Usage
            elif action == 8:
                if self._currentState in self.keyLoc:
                    self.hasKey = True
                    newState = self._currentState
                elif tuple(self._currentState) in self.doorLoc.keys() and self.hasKey:
                    self.doorRemoval(self._currentState)
                    newState = self._currentState
            self._currentState = newState
            stepReward = self.rewardCheck()
            self._rewards += stepReward
        if self._currentState == self.endState:
            self._inTerminal = True
        print(self._currentState)
        return self._currentState, stepReward, self._inTerminal

    '''
    doorRemoval takes an action and the coordinate, then removes both blockages
    that the door creates.
    '''
    def doorRemoval(self, state):
        #Doors block two paths, so both need to be removed.
        #Up (0) and Down (2) - Left (3) and Right (1)
        obstToRemove = self.doorLoc[tuple(state)]
        if obstToRemove == "down":
            self.obstacleRemoval(state, obstToRemove)
            otherDoor = [state[0], state[1]+1]
            self.obstacleRemoval(otherDoor, "up")
        elif obstToRemove == "up":
            self.obstacleRemoval(state, obstToRemove)
            otherDoor = [state[0], state[1]-1]
            self.obstacleRemoval(otherDoor, "down")
        elif obstToRemove == "left":
            self.obstacleRemoval(state, obstToRemove)
            otherDoor = [state[0]-1, state[1]]
            self.obstacleRemoval(otherDoor, "right")
        else:
            self.obstacleRemoval(state, obstToRemove)
            otherDoor = [state[0]+1, state[1]]
            self.obstacleRemoval(otherDoor, "left")


    def obstacleRemoval(self, state, action):
        oldForbid = self.fT[action]
        removeInd = oldForbid.index(state)
        oldForbid.pop(removeInd)
        newForbid = oldForbid
        self.fT[action] = newForbid


    '''
    rewardCheck checks the entered new state to see if there is any reward related to the state.
    If there is, it is added to the reward totals.
    '''
    def rewardCheck(self):
        if self._currentState in self.obst:
            return -10
        elif self._currentState == self.endState:
            return 10
        else:
            return 0

    '''
    getRewards returns the current reward total.
    '''
    @property
    def reward(self) -> float:
        return self._rewards

    @property
    def action(self) -> int:
        return self._action

    @property
    def state(self) -> []:
        return self._currentState

    @property
    def isEnd(self) -> bool:
        return self._inTerminal

    @property
    def gamma(self) -> float:
        return self._gamma

    def numActions(self):
        return 7

    '''
    reset resets the grid to the original start position, removes any rewards, and sets terminal check status to false.
    '''
    def reset(self):
        self._currentState = self.startState
        self._rewards = 0
        self._inTerminal = False
        self._action = None
        self.hasKey = False

    '''
    alterMove handles stochasicity within the gridworld.
    If percentages need to be changed to influence how the actions are changed,
    update them here!
    '''
    def alterMove(self, action):
        #75% - Do Action / 10% Veer / 10% Veer other way / 5% Random action
        doAction = 0.75
        veer = 0.1
        veerOther = 0.1
        randomAction = 0.05
        choice = np.random.choice([0, 1, 2, 3], 1, p=[doAction, veer, veerOther, randomAction])
        if choice[0] == 0:
            return action
        #Veer towards the right
        elif choice[0] == 1:
            #Up - 0
            if action == 0:
                return 1
            #Right - 1
            elif action == 1:
                return 2
            #Down - 2
            elif action == 2:
                return 3
            #Left - 3
            elif action == 3:
                return 0
            #Currently assumes use and break are just always input
            else:
                return action
        #Veer towards the left
        elif choice[0] == 2:
            #Up - 0
            if action == 0:
                return 3
            #Right - 1
            elif action == 1:
                return 0
            #Down - 2
            elif action == 2:
                return 1
            #Left - 3
            elif action == 3:
                return 2
            else:
                return action
        else:
            return np.random.choice([0, 1, 2, 3, 8, 9], 1)[0]

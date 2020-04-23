import numpy as np
import pickle
import os
from Source.systemrl.environments.obstacle import Obstacle

class AdvGridworld:

    def __init__(self, grid):
        self.boardDim = ""
        self.startState = ""
        self.endState = ""
        self.fT = ""
        self.obst = ""
        self.keyLoc = ""
        self.movements = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'use', 5: 'break'}
        #self.movements = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'upri', 5: 'dori', 6: 'dole', 7: 'uple', 8: 'use', 9: 'break'}
        self._currentState = None
        self._rewards = 0
        self._inTerminal = False
        #Track environment changes
        self._initBools = ""
        self._changes = ""
        #door list
        self._doors = ""
        #breakable list
        self._breakList = ""
        self._gridData = ""
        self.getGrid(grid)
        self._name = "Advanced Gridworld"
        self._action = None
        self._gamma = 0.99
        self._numSteps = 0
        #Turn stochastic on (True) or off (False)
        self.stoch = True
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
        ##os.chdir(os.path.split(__file__)[0])
        fileName = ""
        if selection == 1:
            fileName = "grids/gridworld1.p"
        elif selection == 2:
            fileName = "grids/gridworld2.p"
        elif selection == 3:
            fileName = "grids/gridworld3.p"
        else:
            fileName = "grids/gridworld4.p"
        with open(fileName, 'rb') as f:
            gridData = pickle.load(f)
        #gridData = pickle.load(open(fileName, "rb"))
        self._gridData = gridData
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
        self._doors = gridInfo["door"]
        self._breakList = gridInfo["break"]
        self._initBools = gridInfo["bools"]
        self._changes = self._initBools

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
    8 - Use (picks up key, and uses it on doors)
    9 - Break (breaks breakable objects, cool!)
    '''
    def step(self, poten_action):
        if self.stoch:
            action = self.alterMove(poten_action)
        else:
            action = poten_action
        stepReward = 0
        self._action = action
        actionCheck = self.movements[action]
        if action in [4, 5] or self._currentState not in self.fT[actionCheck] and not self._inTerminal:
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
            #Key Pick Up & Usage
            elif action == 4:
                if self._currentState == self.keyLoc:
                    self.hasKey = True
                    self._changes[0] = True
                    newState = self._currentState
                elif self.hasKey:
                    for door in self._doors:
                        if door.checkLocation(self._currentState):
                            remove = door.getBlockedPaths()
                            boolIndx = door.getBoolId()
                            self._changes[boolIndx] = True
                            self.removeBlock(remove)
                            remove = self._doors.index(door)
                            self._doors.pop(remove)
                            newState = self._currentState
                        else:
                            newState = self._currentState
                else:
                    newState = self._currentState
            #Break an object
            elif action == 5:
                for obstacle in self._breakList:
                    if obstacle.checkLocation(self._currentState):
                        remove = obstacle.getBlockedPaths()
                        boolIndx = obstacle.getBoolId()
                        self._changes[boolIndx] = True
                        self.removeBlock(remove)
                        remove = self._breakList.index(obstacle)
                        self._breakList.pop(remove)
                        newState = self._currentState
                    else:
                        newState = self._currentState
            if len(newState) < 1:
                newState = self._currentState
            self._currentState = newState
            stepReward = self.rewardCheck()
            self._rewards += stepReward * (self._gamma**self._numSteps)
        self._numSteps += 1
        if self._currentState == self.endState:
            self._inTerminal = True
        return self.state, stepReward, self._inTerminal

    #diagonal stuff had to be moved because python is a bad language
    '''
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
    '''

    '''
    removeBlock takes a dictionary of forbidden transitions and removes them.
    '''
    def removeBlock(self, blocked):
        actionList = blocked.keys()
        for action in actionList:
            coord = blocked[action]
            oldForbid = self.fT[action]
            removeInd = oldForbid.index(coord)
            oldForbid.pop(removeInd)
            newForbid = oldForbid
            self.fT[action] = newForbid



    '''
    rewardCheck checks the entered new state to see if there is any reward related to the state.
    If there is, it is added to the reward totals.
    '''
    def rewardCheck(self):
        if self._currentState in self.obst:
            return -50
        elif self._currentState == self.endState:
            return 100
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
        stateId = self.convertBoolToInt()
        fullState = [self._currentState, stateId]
        return tuple(fullState)

    @property
    def isEnd(self) -> bool:
        return self._inTerminal

    @property
    def gamma(self) -> float:
        return self._gamma

    #Currently set to 6 actions because Gridworld 2/3/4 cannot use diagonal movement.
    def numActions(self):
        return 6

    def getBoardDim(self):
        return self.boardDim

    def numSteps(self) -> int:
        return self._numSteps

    def numEnvChanges(self):
        return len(self._changes)

    def getStart(self):
        stateId = self.convertBoolToInt()
        starting = [self.startState, stateId]
        return starting

    '''
    reset resets the grid to the original start position, removes any rewards, and sets terminal check status to false.
    '''
    def reset(self):
        self._currentState = self.startState
        self._rewards = 0
        self._inTerminal = False
        self._action = None
        self.hasKey = False
        self._numSteps = 0
        self.obst = self._gridData["obs"]
        self.keyLoc = self._gridData["key"]
        self._doors = self._gridData["door"]
        self._breakList = self._gridData["break"]
        self._changes = self._initBools

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
            return np.random.choice([0, 1, 2, 3, 4, 5], 1)[0]

    def convertBoolToInt(self):
        stringNum = ""
        for bo in self._changes:
            if bo == True:
                stringNum = stringNum + str(2)
            else:
                stringNum = stringNum + str(1)
        return int(stringNum)

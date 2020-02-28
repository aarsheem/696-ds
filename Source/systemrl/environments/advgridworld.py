import numpy as np
import pickle
import os

class AdvGridworld:

    def __init__(self):
        self.boardDim = ""
        self.startState = ""
        self.endState = ""
        self.fT = ""
        self.obst = ""
        self.movements = {0: 'up', 1: 'right', 2: 'down', 3: 'left', 4: 'upri', 5: 'dori',
                 6: 'dole', 7: 'uple'}
        self._currentState = None
        self._rewards = 0
        self._inTerminal = False
        self.getGrid(1)
        self._name = "Advanced Gridworld"
        self._action = None
        self._gamma = 0.95
        self._numSteps = 0

    @property
    def name(self) -> str:
        return self._name

    '''
    getBoard method takes a numerical input to open a saved file with the grid information.
    The file information is parsed and passed to the generateBoard method which creates the matrix of
    information that is required to play.
    '''
    def getGrid(self, selection):
        os.chdir(os.path.split(__file__)[0])
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
    def step(self, action):
        #print(self.state)
        stepReward = 0
        self._action = action
        actionCheck = self.movements[action]
        if self._currentState not in self.fT[actionCheck] and not self._inTerminal:
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
            self._currentState = newState
            stepReward = self.rewardCheck()
            self._rewards += stepReward * (self._gamma**self._numSteps)
            self._numSteps += 1
        if self._currentState == self.endState:
            self._inTerminal = True
        return self._currentState, stepReward, self._inTerminal

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

    def getBoardDim(self):
        return self.boardDim

    def numActions(self):
        return 8

    '''
    reset resets the grid to the original start position, removes any rewards, and sets terminal check status to false.
    '''
    def reset(self):
        self._currentState = self.startState
        self._rewards = 0
        self._inTerminal = False
        self._action = None
        self._numSteps = 0

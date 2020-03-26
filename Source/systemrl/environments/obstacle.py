class Obstacle:
    def __init__(self):
        self.location = []
        self.blockedPaths = ""

    def setLocation(self, loc):
        self.location = loc

    def checkLocation(self, actionLoc):
        if actionLoc in self.location:
            return True
        else:
            return False

    def setBlock(self, blocked):
        self.blockedPaths = blocked

    def getBlockedPaths(self):
        return self.blockedPaths


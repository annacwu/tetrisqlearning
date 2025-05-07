from tetrisGame import Tetris
import numpy as np

# NOTE: im not sure if this is checking how we would want it to in normal gameplay like with ticks
class TetrisEnv: 
    def __init__(self):
        self.game = Tetris()
        self.score = 0
        self.state = None
        self.game.newBlock()
        self.term = False
        self.actions = ['left', 'right', 'down', 'rotateLeft', 'rotateRight', 'rotateFlip'] 
        self.num_actions = len(self.actions)
        self.state_dim = 240 # because using active board now
        self.currentCombo = 0
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.actions)}

    def actionToIdx(self, action):
        return self.action_to_idx[action]

    def idxToAction(self, index):
        return self.idx_to_action[index]

    def getState(self):
        return np.array(self.game.activeBoard).flatten()
    
    def reset(self):
        self.game = Tetris()
        self.currentCombo = 0
        self.game.newBlock()
        self.term = False
        self.score = 0
        return self.getState()

    def step(self, a):
        reward = -1

        rowsCleared = self.game.clearRows()
        if rowsCleared == 0:
            self.currentCombo = 0
        else:
            self.currentCombo += 1
        
        if self.game.checkTop():
            self.term = True

        # basic system that attempts to increase scoring for bigger clears and longer combos
        self.score += (rowsCleared + self.currentCombo) ** 2 
        reward += self.score

        # Create new block if no block exists
        if self.game.currentBlock is None:
            self.game.newBlock()

        if a == 'left': 
            self.game.moveLeft()
        elif a == 'right':
            self.game.moveRight()
        elif a == 'down':
            moved = self.game.moveDown()
            if moved == False:
                rowsCleared = self.game.clearRows() if hasattr(self.game, 'clearRows') else 0
                if rowsCleared == 0:
                    self.currentCombo = 0
                else:
                    self.currentCombo += 1
        elif a == 'rotateLeft':
            self.game.rotateLeft()
        elif a == 'rotateRight':
            self.game.rotateRight()
        elif a == 'rotateFlip':
            self.game.rotateFlip()
        
        return self.getState(), reward, self.term

    def getState(self):
        # FIXME: this should maybe be activeboard? or also include the block?
        board = np.array(self.game.board).flatten()
        # if self.game.currentBlock is not None: 
        #     block = np.array(self.game.currentBlock.shape).flatten()
        # else: 
        #     block = np.zeros((16), dtype=np.int32) # 16 because blocks are 4x4 grids

        # state = np.concatenate([board, block])
        return board


    def terminal(self):
        pass




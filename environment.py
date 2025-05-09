from tetris import Tetris
import numpy as np


# NOTE: there is no gravity currently. need to implement that for actual tetris
class TetrisEnv: 
    def __init__(self, stdscr=None):
        self.game = Tetris()
        self.state = None
        self.game.newBlock()
        self.term = False
        self.actions = ['left', 'right', 'down', 'rotateLeft', 'rotateRight', 'rotateFlip', 'hardDrop'] 
        self.num_actions = len(self.actions)
        self.state_dim = 240 # because using active board now
        self.score = 0 # implementing like how a basic tetris does. points for every cell moved down without gravity
        self.currentCombo = 0
        self.ticks = 0
        self.gravity = 5
        self.rotationCount = 0
        self.stdscr = stdscr
        self.blocks_placed = 0

    def getState(self):
        return np.array(self.game.activeBoard).flatten()
    
    def reset(self):
        self.game = Tetris()
        self.currentCombo = 0
        self.game.newBlock()
        self.term = False
        self.ticks = 0
        self.rotationCount = 0
        self.score = 0
        self.blocks_placed = 0
        return self.getState()

    def step(self, a):
        # Create new block if no block exists
        if self.game.currentBlock is None:
            self.rotationCount = 0
            self.game.newBlock()
            self.rotationCount = 0 # reset rotation count for every block right

        if a in ('rotateLeft', 'rotateRight', 'rotateFlip'):
            if self.rotationCount >= self.maxRotations:
                rotation_penalty = self.rotation_penalty
            else:
                self.rotationCount += 1
            getattr(self.game, a)()
        
        if a == 'left': 
            self.game.moveLeft()
        elif a == 'right':
            self.game.moveRight()
        elif a == 'down':
            moved = self.game.moveDown()
            self.score += 1
            if moved == False:
                rowsCleared = self.game.clearRows() if hasattr(self.game, 'clearRows') else 0
                if rowsCleared == 0:
                    self.currentCombo = 0
                else:
                    self.currentCombo += 1
                self.blocks_placed += 1
        elif a == 'rotateLeft':
            self.game.rotateLeft()
            self.rotationCount += 1
        elif a == 'rotateRight':
            self.game.rotateRight()
            self.rotationCount += 1
        elif a == 'rotateFlip':
            self.game.rotateFlip()
            self.rotationCount += 1
        elif a == 'hardDrop':
            self.score += self.game.hardDrop()
            self.blocks_placed += 1

        self.ticks += 1
        if self.ticks % self.gravity == 0:
            self.game.moveDown()
            
        if self.game.checkTop():
            self.term = True

        # Check if we have stdscr for rendering
        if self.stdscr: 
            self.render(self.stdscr)

        reward = self.calculateReward()

        return self.getState(), reward, self.term

    def getState(self):
        board = np.array(self.game.board).flatten()
        return board

    # This is essentially taken from the play function in tetrisGame.py 
    # to render the terminal screen
    def render(self, stdscr):
        stdscr.clear()  
        board_str = self.game.__str__().split('\n')

        screen_height, screen_width = stdscr.getmaxyx()
        board_height = len(board_str)
        board_width = len(board_str[0]) if board_str else 0
        start_y = max((screen_height - board_height) // 2, 0)
        start_x = max((screen_width - board_width) // 2, 0)

        for i, line in enumerate(board_str):
            stdscr.addstr(start_y + i, start_x, line)
        stdscr.refresh()

    def calculateReward(self):
        reward = 0

        rowsCleared = self.game.clearRows()
        if rowsCleared > 0:
            reward += 10000
            self.currentCombo += 1
            reward += (rowsCleared ** 2) * 10 + self.currentCombo * 2
        else:
            self.currentCombo = 0

        if self.rotationCount > 4:
            reward -= 100

        max_height = self.game.checkColumnHeight()
        reward -= max_height * 5 

        reward += self.score
        reward += self.blocks_placed

        # holes = self.game.countHoles()
        # bumpiness = self.game.computeBumpiness()
        # reward -= holes * 1.0
        # reward -= bumpiness * 0.5

        return reward

    def terminal(self):
        pass




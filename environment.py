from tetris import Tetris
import numpy as np


# NOTE: there is no gravity currently. need to implement that for actual tetris
class TetrisEnv: 
    def __init__(self, stdscr=None):
        self.game = Tetris()
        self.score = 0
        self.state = None
        self.game.newBlock()
        self.term = False
        self.actions = ['left', 'right', 'down', 'rotateLeft', 'rotateRight', 'rotateFlip', 'hardDrop'] 
        self.num_actions = len(self.actions)
        self.state_dim = 240 # because using active board now
        self.currentCombo = 0
        self.stdscr = stdscr
        self.stepPenalty = 0.01
        self.rotation_penalty = 0.01
        self.maxRotations = 4
        self.rotationCount = 0

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
        # if they clear no rows, reward should still be -1
        prev_score = self.score
        rotation_penalty = 0.0
        drop_bonus = 0.0

        # Create new block if no block exists
        if self.game.currentBlock is None:
            self.rotationCount = 0
            self.game.newBlock()

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
            if moved == False:
                rows = self.game.clearRows()
                self.currentCombo = self.currentCombo + 1 if rows > 0 else 0
                self.rotationCount = 0
                self.game.newBlock()
            

        moved = self.game.moveDown()
        if moved:
            reward += 0.005

        if not moved:
            rows = self.game.clearRows()
            if rows > 0:
                self.currentCombo += 1
            else:
                self.currentCombo = 0
            self.game.newBlock()
        else:
            rows = 0
        
        self.score += (rows + self.currentCombo) ** 2
        reward = (self.score - prev_score) - self.stepPenalty
        
        if self.game.checkTop():
            self.term = True

        # Check if we have stdscr for rendering
        if self.stdscr: 
            self.render(self.stdscr)

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

    def terminal(self):
        pass




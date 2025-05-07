from tetrisGame import Tetris
import numpy as np

# NOTE: im not sure if this is checking how we would want it to in normal gameplay like with ticks
class TetrisEnv: 
    def __init__(self, stdscr=None):
        self.game = Tetris()
        self.score = 0
        self.state = None
        self.game.newBlock()
        self.term = False
        self.actions = ['left', 'right', 'down', 'rotateLeft', 'rotateRight', 'rotateFlip'] 
        self.num_actions = len(self.actions)
        self.state_dim = 240 # because using active board now
        self.currentCombo = 0
        self.stdscr = stdscr

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

        # Check if we have stdscr (curses environment)
        if self.stdscr: 
            self.render(self.stdscr)

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

    def render(self, stdscr):
        stdscr.clear()  # Refresh screen
        board_str = self.game.__str__().split('\n')

        # Get terminal size for centering the game board
        screen_height, screen_width = stdscr.getmaxyx()
        board_height = len(board_str)
        board_width = len(board_str[0]) if board_str else 0
        start_y = max((screen_height - board_height) // 2, 0)
        start_x = max((screen_width - board_width) // 2, 0)

        # Draw board on terminal
        for i, line in enumerate(board_str):
            stdscr.addstr(start_y + i, start_x, line)
        stdscr.refresh()

    def terminal(self):
        pass




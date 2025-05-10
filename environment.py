from tetris import Tetris
import numpy as np

class TetrisEnv: 
    def __init__(self, stdscr=None):
        self.stdscr = stdscr
        self.gravity = 5
        self.actions = ['left', 'right', 'down', 'rotateLeft', 'rotateRight', 'rotateFlip', 'hardDrop'] 
        self.num_actions = len(self.actions)    
        self.state_dim = 240 # because using active board now

        self.reset_game_stuff()

        self.game = Tetris()
        self.state = None
        self.game.newBlock()
        self.term = False
        
        
    
    # separate call for resetting per-game variables to make it easier
    def reset_game_stuff(self):
        # dynamic reward properties
        self.score = 0 # implementing like how a basic tetris does. points for every cell moved down without gravity
        self.currentCombo = 0
        self.ticks = 0
        self.rotationCount = 0
        self.blocks_placed = 0 # try and get it to place more blocks so it stops building towers2

    def getState(self):
        return np.array(self.game.activeBoard).flatten()
    
    def reset(self):
        self.reset_game_stuff()
        self.game = Tetris()
        self.game.newBlock()
        self.term = False
        return self.getState()

    def step(self, a):
        # Create new block if no block exists
        if self.game.currentBlock is None:
            self.rotationCount = 0
            self.game.newBlock()
        
        locked = False
        if a == 'left': 
            self.game.moveLeft()
        elif a == 'right':
            self.game.moveRight()
        elif a == 'down':
            moved = self.game.moveDown()
            self.score += 1 # increase for moving down on purpose
            if not moved:
                locked = True
                self.blocks_placed += 1
        elif a == 'rotateLeft':
            self.game.rotateLeft()
            self.rotationCount += 1
        elif a == 'rotateRight':
            self.game.rotateRight()
            self.rotationCount += 1 # update rotation in here
        elif a == 'rotateFlip':
            self.game.rotateFlip()
            self.rotationCount += 1
        elif a == 'hardDrop':
            dropped = self.game.hardDrop()
            self.blocks_placed += 1
            locked = True

        # implementing gravity. i think this is working
        self.ticks += 1
        if self.ticks % self.gravity == 0:
            self.game.moveDown()
            
        if self.game.checkTop():
            self.term = True

        rowsCleared = 0
        if locked:
            rowsCleared = self.game.clearRows()
            if rowsCleared > 0:
                self.currentCombo += 1
            else:
                self.currentCombo = 0
        
        # check if we have stdscr for rendering
        if self.stdscr: 
            self.render(self.stdscr)

        reward = 0.0
        if rowsCleared:
            reward += ((rowsCleared * 1000) ** rowsCleared) + 0.5 * self.currentCombo
        
        if self.rotationCount > 4:
            reward -= (self.rotationCount - 4) * 0.1
        
        max_height = self.game.checkColumnHeight()
        reward -= 10 * max_height

        holes = self.countHoles()
        reward -= 0.2 * holes

        breadth = self.checkBreadth()
        reward += breadth * 10

        # reward += self.blocks_placed * 5


        return self.getState(), reward, self.term

    def countHoles(self):
        holes = 0
        b = self.game.board
        for column in range(self.game.width):
            seen_block = False

            for row in range(self.game.height):
                cell = b[row][column]
                
                if cell == 1:
                    seen_block = True
                elif seen_block and cell == 0:
                    holes += 1

        return holes

    def checkBreadth(self):
        breadth = 0
        intensity = 10
        for row in range(self.game.height):
            num_1s = 0
            for column in range(self.game.width):
                cell = self.game.board[row][column]

                if cell == 1:
                    num_1s += 1
            breadth += num_1s * intensity
            intensity // 2

        return breadth
                
    
    
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

    # Moved here for increased readability
    def calculateReward(self):
        reward = 0

        # if it clears a row we are ECSTATIC. it has never done this
        rowsCleared = self.game.clearRows()
        if rowsCleared > 0:
            reward += 10000
            self.currentCombo += 1
            reward += (rowsCleared ** 2) * 10 + self.currentCombo * 2
        else:
            self.currentCombo = 0

        # penalize for too many rotations per block
        if self.rotationCount > 4:
            reward -= 100 * self.rotationCount

        # penalize for building towers hopefully. 
        # in theory this should make it learn to stay low
        max_height = self.game.checkColumnHeight()
        reward -= max_height * 5 

        # positive reinforcement for playing the game
        reward += self.score
        reward += self.blocks_placed

        # things to maybe implement later to help it
        # holes = self.game.countHoles()
        # reward -= holes * 1.0

        return reward
    
    

            





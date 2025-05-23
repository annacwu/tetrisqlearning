from tetris import Tetris
from tetris import BLOCKS
import math
import copy
import numpy as np

""" Environment compatible with the neural network """
class TetrisEnv: 
    def __init__(self, stdscr=None, graphical=False):
        # choose how to render it
        self.stdscr = stdscr
        self.graphical = graphical

        # set up basics
        self.game = Tetris(graphical=self.graphical)
        self.state = None
        self.game.new_block()
        self.term = False
        self.gravity = 5
        self.state_dim = 240 # because using active board now
        # first attempt at training (without delayed reward)
        self.actions = ['left', 'right', 'down', 'rotate_left', 'rotate_right', 'rotate_flip', 'hard_drop'] 
        self.num_actions = len(self.actions)    
        # macro actions for training with a delayed reward
        self.macro_actions = self.get_macro_actions()
        self.num_macro_actions = len(self.macro_actions)

        # properties to track how well it is doing
        self.score = 0 # implementing like how a basic tetris does: points for every cell moved down without gravity
        self.current_combo = 0
        self.ticks = 0
        self.rotation_count = 0 # not used with macro actions
        self.blocks_placed = 0 # try and get it to place more blocks so it stops building towers

    # delayed training: place a block at a specific rotation in a chosen column (many actions at once)
    def get_macro_actions(self):
        macro_actions = []
        # for each of the 7 different blocks
        for block_type in BLOCKS:
            # get all possible rotations of the block
            for rotation in range(len(BLOCKS[block_type])):
                # and which column it can be in
                for col in range(self.game.width):
                    macro_actions.append((block_type, rotation, col))

        return macro_actions
    
    # separate call for resetting per-game variables to make it easier
    def reset_game_stuff(self):
        self.score = 0 
        self.current_combo = 0
        self.ticks = 0
        self.rotation_count = 0
        self.blocks_placed = 0 

    # state is given to the network as a 240 input array
    def get_state(self):
        return np.array(self.game.active_board).flatten()
    
    def reset(self):
        self.reset_game_stuff()
        self.game = Tetris(graphical=self.graphical)
        self.game.new_block()
        self.term = False
        return self.get_state()

    # handles doing all the steps in a macro step at once with existing game logic from tetris.py
    def macro_step(self, a):
        block_type, rotation, col = a

        if self.game.current_block is None:
            self.game.new_block()
        # if they chose a block that doesn't work
        if self.game.current_block.type != block_type:
            return self.get_state(), -1, self.term

        # for comparing how the new action affected the game
        previous_board = copy.deepcopy(self.game.board)

        # if valid action, do all the small steps at once
        new_rotation = (rotation - self.game.current_block.rotation) % len(BLOCKS[block_type])
        for _ in range(new_rotation):
            self.game.rotate_right()

        current_x = self.game.current_block.x
        while current_x < col:
            self.game.move_right()
            current_x += 1
        while current_x > col: 
            self.game.move_left()
            current_x -= 1

        dropped = self.game.hard_drop()
        self.score += dropped
        self.blocks_placed += 1

        # do all of the normal step checks
        if self.game.check_top():
            self.term = True

        locked = True
        reward = self.calculate_reward(locked, previous_board)

        if self.graphical: 
            self.game.render_pygame()

        if self.stdscr: 
            self.render(self.stdscr)

        return self.get_state(), reward, self.term


    # step for the simple actions
    def step(self, a):
        if self.game.current_block is None:
            self.rotation_count = 0
            self.game.new_block()
        
        locked = False
        if a == 'left': 
            self.game.move_left()
        elif a == 'right':
            self.game.move_right()
        elif a == 'down':
            moved = self.game.move_down()
            self.score += 1 # increase for moving down on purpose
            if not moved:
                locked = True
                self.blocks_placed += 1
        elif a == 'rotate_left':
            self.game.rotate_left()
            self.rotation_count += 1
        elif a == 'rotate_right':
            self.game.rotate_right()
            self.rotation_count += 1 # update rotation in here
        elif a == 'rotate_flip':
            self.game.rotate_flip()
            self.rotation_count += 1
        elif a == 'hard_drop':
            dropped = self.game.hard_drop()
            self.score += dropped
            self.blocks_placed += 1
            locked = True

        # implementing gravity. i think this is working
        self.ticks += 1
        if self.ticks % self.gravity == 0:
            self.game.move_down()
            
        if self.game.check_top():
            self.term = True

        if self.graphical:
            self.game.render_pygame()

        if self.stdscr: 
            self.render(self.stdscr)

        reward = -0.1
        if locked:
            reward = self.calculate_reward(locked)

        return self.get_state(), reward, self.term
        
    """ FUNCTIONS FOR CALCULATING THE REWARD """

    # check how many empty spaces are in each row to try and get it to fill them 
    def count_row_gaps(self):
        gaps = 0
        for row in range(self.game.height): 
            seen_block = False
            for column in range(self.game.width):
                cell = self.game.board[row][column]

                if cell != 0:
                    seen_block = True
            if seen_block:
                for column in range(self.game.width):
                    cell = self.game.board[row][column]

                    if cell == 0:
                        # penalize rows that are closer to the bottom (more important they are filled in)
                        penalty = (row + 1) / self.game.height * 10
                        gaps += penalty 
        return int(gaps)

    # check how many spaces are blocked off by a block above it
    def count_holes(self):
        holes = 0
        for column in range(self.game.width):
            seen_block = False

            for row in range(self.game.height):
                cell = self.game.board[row][column]
                
                if cell != 0:
                    seen_block = True
                elif seen_block and cell == 0:
                    holes += 1

        return holes

    # gives the maximum height of the columns (trying to keep it lower)
    def check_max_column_height(self):
        for i, row in enumerate(self.game.board):
            if any(cell != 0 for cell in row):
                return self.game.height - i + self.game.hidden
        return 0
    
    # calculate each individual column height
    def column_heights(self):
        heights = [0] * self.game.width
        for x in range(self.game.width):
            for y in range(self.game.height):
                if self.game.board[y][x] != 0:
                    heights[x] = self.game.height - y
                    break  # stop at first filled cell
        return heights

    # determine bumpiness by checking all the heights against each other
    def column_height_variance(self):
        heights = self.column_heights()
        mean = sum(heights) / len(heights)
        variance = sum((h - mean) ** 2 for h in heights) / len(heights)
        return math.sqrt(variance)  # return std deviation

    # tries to get it to spread the blocks out more to fill the rows wider
    def check_breadth(self):
        bonus = 0
        for row in self.game.board[self.game.hidden:]:
            filled = sum(1 for cell in row if cell != 0)
            if 7 <= filled:  
                bonus += filled  # encourage near-full rows
        return bonus

    # reward it getting closer to completing a row
    def near_full_rows(self):
        count = 0
        for row in self.game.board[self.game.hidden:]:
            filled = sum(1 for cell in row if cell != 0)
            if filled >= self.game.width - 2:
                count += 1
        return count

    # encourage it to place blocks against each other as tightly as possible
    def flush_contacts(self, previous_board):
        contacts = 0
    
        for y in range(self.game.height):
            for x in range(self.game.width):
                if self.game.board[y][x] != 0 and previous_board[y][x] == 0:
                    # only count if it is against another block not a wall 
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.game.width and 0 <= ny < self.game.height:
                            if self.game.board[ny][nx] != 0:
                                contacts += 1
    
                    # check the bottom
                    if y + 1 < self.game.height and self.game.board[y + 1][x] != 0:
                        contacts += 1
                    # extra reward for being flush with the bottom
                    elif y + 1 == self.game.height:
                        contacts += 5
        return contacts
    

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

    # reward calculation moved here for increased readability
    def calculate_reward(self, locked, previous):
        reward = 0

        # big reward for clearing rows
        rowsCleared = 0
        if locked:
            rowsCleared = self.game.clear_rows()
            if rowsCleared > 0:
                self.current_combo += 1
                print("ROW CLEARED !!!!!!!!!!!!!!!!!!!! ")
            else:
                self.current_combo = 0
        reward += rowsCleared * 20000        
        reward += self.current_combo * 10
        
        # negative rewards for bad behavior
        max_height = self.check_max_column_height()
        reward -= max_height * 2 

        gaps = self.count_row_gaps()
        reward -= gaps   

        reward -= int(self.column_height_variance()) * 1.5

        holes = self.count_holes()
        reward -= holes

        # positive rewards for good behavior
        breadth = self.check_breadth()
        reward += breadth * 100 

        reward += self.blocks_placed 
        
        flush_bonus = self.flush_contacts(previous)
        reward += flush_bonus * 5

        reward += self.near_full_rows() * 100

        reward += self.score 

        return reward



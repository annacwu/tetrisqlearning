import random
import pygame
import curses
import time

BLOCK_TYPES = [1, 2, 3, 4, 5, 6, 7]

BLOCK_COLORS = {
    1: (255, 255, 0),  # yellow
    2: (0, 255, 255),  # cyan
    3: (128, 0, 128),  # purple
    4: (255, 165, 0),  # orange
    5: (0, 0, 255),    # blue
    6: (0, 255, 0),    # green
    7: (255, 0, 0)     # red
}
# while playing i have a game tick set up, and gravity will be 
# move down if ticks % gravity == 0; higher gravity is slower 
GRAVITY = 5

REPLAY = True

BLOCKS = {
    1: [
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0]
        ]
    ],
    2: [
        # Vertical – occupies all four rows so no extra empty rows can be added at the top.
        [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ],
        # Horizontal – normalized so that the single filled row comes last.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]
    ],
    3: [
        # Rotation 0 (spawn state)
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0]
        ],
        # 90° rotation
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0]
        ],
        # 180° rotation
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 0, 0]
        ],
        # 270° rotation
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ]
    ],
    4: [
        # L spawn – vertical with extra block on the bottom right.
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0]
        ],
        # 90° rotation – normalized so that three nonempty rows appear at the bottom.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0]
        ],
        # 180° rotation – computed by rotating 180° then moving the bottom empty row to the top.
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ],
        # 270° rotation – normalized with all empty rows at the top.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 1, 1]
        ]
    ],
    5: [
        # J spawn – mirror image of L spawn.
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 1, 0]
        ],
        # 90° rotation for J.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0]
        ],
        # 180° rotation – normalized (note that after rotating 180° an empty row ended up at the bottom so it is moved to the top).
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0]
        ],
        # 270° rotation for J.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 1]
        ]
    ],
    6: [
        # S spawn.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0]
        ],
        # 90° rotation – normalized so that the three nonempty rows are at the bottom.
        [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0]
        ]
    ],
    7: [
        # Z spawn.
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 0]
        ],
        # 90° rotation – normalized so that the row(s) of zeros are up top.
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ]
    ]
}

class Block: 
    x = 3
    y = 0
    block = None
    rotation = 0

    def __init__(self, type, rotation=0):
        self.type = type
        self.rotation = rotation
        self.block = BLOCKS[type][rotation]

    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.block])
    
    # rotates CLOCKWISE x number of times
    def rotate(self, x):
        num_rotations = len(BLOCKS[self.type])
        self.rotation = (self.rotation + x) % num_rotations
        self.block = BLOCKS[self.type][self.rotation]
    
    def clone(self):
        new_block = Block(self.type, self.rotation)
        new_block.x = self.x
        new_block.y = self.y
        return new_block
            
    
"""set up the board and define methods that handle interaction between board and blocks"""
class GameBoard:
    
    # initialized to standard tetris board size plus padding
    def __init__(self, height = 20, width = 10, padding = 4, graphical=False): 
        self.height = height
        self.width = width 
        self.hidden = padding
        self.total_height = height + padding

        self.board = [] # board will be represented as 2d array (matrix)
        self.active_board = [] # this is the temporary board where we do the block rotation actions and stuff and check bounds
        self.current_block = None
        self.blocks = BLOCK_TYPES[:]
        self.active = False # for determining which board to show

        # pygame stuff for graphical rendering
        if graphical: 
            self.block_size = 16  # size of each block in pixels
            self.screen = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
            pygame.init()

        # make boards with given sizes
        for _ in range(self.total_height):
            row = [0] * width
            self.board.append(row)

        for _ in range(self.total_height):
            rows = [0] * (self.width)
            self.active_board.append(rows)

    def __str__(self):
        visible_board = self.board[self.hidden:]  # skip the top 'padding' rows
        if self.active: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in self.active_board[self.hidden:])
        else: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in visible_board)

    def new_block(self): 

        # implement bag: cycles through the 7 in random order
        if self.blocks == []:
            self.blocks = BLOCK_TYPES[:]
        random.shuffle(self.blocks)
        self.current_block = Block(self.blocks[0])
        self.blocks.pop(0)
        self.current_block.y = self.hidden - 1
        self.draw_block()
        
    # gets rid of rows that are full
    def clear_rows(self):
        visible_board = self.board[self.hidden:]
        updated_visible = [row for row in visible_board if not all(cell != 0 for cell in row)]
        rowsCleared = len(visible_board) - len(updated_visible)
        

        for _ in range(rowsCleared):
            updated_visible.insert(0, [0] * self.width)

        self.board = self.board[:self.hidden] + updated_visible
        return rowsCleared
    
    # game ending condition, checks if a block is placed above the top visible row
    def check_top(self):    
        for cell in self.board[self.hidden]:
            if cell != 0:
                return True
        return False
        
    # input current block x and y + 1 in whichever direction you want to move
    def can_move(self, block, newX, newY):
        # for loops figure out where the block actually is
        for y in range(4):
            for x in range(4):
                if block.block[y][x]: 
                    # need to adjust for where the 1 actually is in the 4x4 grid
                    boardX = newX + x
                    boardY = newY + y 

                    # check boundaries
                    if boardX < 0 or boardX >= self.width:
                        return False
                    if boardY < 0 or boardY >= self.total_height:
                        return False
                    
                    # it already is something
                    if self.board[boardY][boardX] != 0: 
                            return False
        return True

    # put the block in its location on the unlocked board
    def draw_block(self):
        self.active = True
        # when it draws active board it will look like the saved board
        self.active_board = [row[:] for row in self.board] 

        for i in range(4):
            for j in range(4):
                if self.current_block.block[i][j]:
                    x = self.current_block.x + j
                    y = self.current_block.y + i
                    if 0 <= y < self.total_height and 0 <= x < self.width:
                        self.active_board[y][x] = self.current_block.type

    # draw blocks permanent state on the locked board
    def lock_block(self):
        self.active = False
        for j in range(4):
            for i in range(4):
                if self.current_block.block[j][i]:
                    boardX = self.current_block.x + i
                    boardY = self.current_block.y + j
                    if 0 <= boardY < self.total_height and 0 <= boardX < self.width:
                        self.board[boardY][boardX] = self.current_block.type
        self.current_block = None

    # render the graphical representation using pygame
    def render_pygame(self):
        self.screen.fill((0, 0, 0))  # black background
        
        # current board state
        for row in range(self.hidden, self.total_height):
            for col in range(self.width):
                block_type = self.board[row][col]
                if block_type != 0:
                    color = BLOCK_COLORS.get(block_type, (255, 255, 255))
                else: 
                    color = (0,0,0)
                
                pygame.draw.rect(
                        self.screen, 
                        color,
                        (col * self.block_size,
                         (row - self.hidden) * self.block_size, 
                         self.block_size, self.block_size))
        
        # add the current block
        if self.current_block:
            block_color = BLOCK_COLORS[self.current_block.type]  
            for i in range(4):
                for j in range(4):
                    if self.current_block.block[i][j]:  
                        x = self.current_block.x + j
                        y = self.current_block.y + i
                        if 0 <= x < self.width and 0 <= y < self.total_height: 
                            pygame.draw.rect(
                                    self.screen, 
                                    block_color,
                                    (x * self.block_size, 
                                     (y - self.hidden) * self.block_size, 
                                     self.block_size, self.block_size))

        pygame.display.update()  

"""gameplay functionality methods"""
class Tetris(GameBoard): 
    score = 0

    def __init__(self, height=20, width=10, padding=4, graphical=False):
        super().__init__(height, width, padding, graphical)
    
    def move_right(self):
        if self.can_move(self.current_block, self.current_block.x + 1, self.current_block.y):
            self.current_block.x += 1
            self.draw_block()

    def move_left(self):
        if self.can_move(self.current_block, self.current_block.x - 1, self.current_block.y):
            self.current_block.x -= 1
            self.draw_block()

    def move_down(self):
        if self.current_block is None:
            return False
        if self.can_move(self.current_block, self.current_block.x, self.current_block.y + 1):
            self.current_block.y +=1
            self.draw_block()
        else: 
            self.lock_block()
            return False

    def rotate_left(self):
        newConfig = self.current_block.clone()
        newConfig.rotate(3)

        candidate_offsets = [0, -1, 1, -2, 2]

        for offset in candidate_offsets:
            tempX = newConfig.x + offset
            
            if self.can_move(newConfig, tempX, newConfig.y):
                newConfig.x = tempX
                self.current_block = newConfig
                self.draw_block()
                return

    def rotate_right(self):
        newConfig = self.current_block.clone()
        newConfig.rotate(1)

        candidate_offsets = [0, -1, 1, -2, 2]

        for offset in candidate_offsets:
            tempX = newConfig.x + offset
            if self.can_move(newConfig, tempX, newConfig.y):
                newConfig.x = tempX
                self.current_block = newConfig
                self.draw_block()
                return

    def rotate_flip(self):
        newConfig = self.current_block.clone()
        newConfig.rotate(2)

        candidate_offsets = [0, -1, 1, -2, 2]

        for offset in candidate_offsets:
            tempX = newConfig.x + offset
            if self.can_move(newConfig, tempX, newConfig.y):
                newConfig.x = tempX
                self.current_block = newConfig
                self.draw_block()
                return

    def hard_drop(self):
        distanceDropped = 0
        while self.can_move(self.current_block, self.current_block.x, self.current_block.y + 1):
            self.current_block.y +=1
            distanceDropped += 1
        self.lock_block()
        return distanceDropped
        
def game_over_screen(stdscr, score):
    stdscr.clear()
    screen_height, screen_width = stdscr.getmaxyx()
    msg1 = f"SCORE: {score}"
    msg2 = "Y: PLAY AGAIN, ANY KEY TO EXIT"
    time.sleep(0.2)
    stdscr.addstr(screen_height // 2 - 1, (screen_width - len(msg1)) // 2, msg1)
    stdscr.addstr(screen_height // 2, (screen_width - len(msg2)) // 2, msg2)
    stdscr.refresh()
    key = stdscr.getch()
    return key == ord('y')


# currently formatted for playing in terminal
# stdscr represents the terminal window
def play_game(stdscr):
    curses.curs_set(0) # hide blinking terminal cursor
    stdscr.nodelay(True) # doesn't wait for keypress to do calls
    stdscr.timeout(100) # refresh rate 100ms

    while True:
        playing = True
        game = Tetris()
        score = 0
        current_combo = 0
        ticks = 0
        
        while playing: 
            
            stdscr.clear() # refresh screen

            rowsCleared = game.clear_rows()
            if rowsCleared == 0:
                current_combo = 0
            else:
                current_combo += 1
            
            if game.check_top():
                playing = False
                break

            # basic system that attempts to increase scoring for bigger clears and longer combos
            score += (rowsCleared + current_combo) ** 2 
            
            if game.current_block == None:
                game.new_block()

            # this is which board we are drawing
            board_str = game.__str__().split('\n')

            # OPERATIONS BELOW TO CENTER IT IN THE TERMINAL
            # get terminal size
            screen_height, screen_width = stdscr.getmaxyx()
            # get board size
            board_height = len(board_str)
            board_width = len(board_str[0]) if board_str else 0
            # calculate starting y,x to center
            start_y = max((screen_height - board_height) // 2, 0)
            start_x = max((screen_width - board_width) // 2, 0)

            # draw board on terminal with stats
            for i, line in enumerate(board_str):
                stdscr.addstr(start_y + i, start_x, line)
            info_x = start_x + game.width * 2 + 4  
            stdscr.addstr(start_y, info_x, f"Score: {score}")
            stdscr.addstr(start_y + 2, info_x, f"Column Height: {game.checkColumnHeight()}")

            stdscr.refresh()

            try:
                key = stdscr.getch() 
            except:
                key = -1
            
            if key == curses.KEY_LEFT:
                game.move_left()
            elif key == curses.KEY_RIGHT:
                game.move_right()
            elif key == curses.KEY_DOWN:
                game.move_down()
            elif key == ord('a'):
                game.rotate_left()
            elif key == ord('s'):
                game.rotate_flip()
            elif key == ord('d'):
                game.rotate_right()
            elif key == ord(' '):
                game.hard_drop()
            
            # we can stop the game now
            elif key == ord('p'):
                game_over_screen(stdscr, score)

            ticks += 1
            if ticks % GRAVITY == 0:
                game.move_down()

            # time.sleep(0.05) 
            # doesn't loop too fast
        if game_over_screen(stdscr, score):
            continue
        else:
            break

""" UNCOMMENT TO PLAY MANUALLY IN TERMINAL """
# curses.wrapper(play_game)
    

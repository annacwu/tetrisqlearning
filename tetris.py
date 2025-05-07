import random
import curses
import time

BLOCK_TYPES = ['O', 'L', 'J', 'I', 'T', 'Z', 'S']

# while playing i have a game tick set up, and gravity will be 
# move down if ticks % gravity == 0; higher gravity is slower 
GRAVITY = 5

REPLAY = True

BLOCKS = {
    "O": [
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0]
        ]
    ],
    "I": [
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
    "T": [
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
    "L": [
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
    "J": [
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
    "S": [
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
    "Z": [
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
            
    

class GameBoard:

    def __init__(self, height = 20, width = 10, padding = 4): # initialized to standard tetris board size plus padding
        self.height = height
        self.width = width 
        self.hidden = padding
        self.totalHeight = height + padding

        self.board = [] # board will be represented as 2d array (matrix)
        self.activeBoard = [] # this is the temporary board where we do the block rotation actions and stuff and check bounds
        self.currentBlock = None
        self.blocks = BLOCK_TYPES[:]
        self.active = False # for determining which board to show

        for _ in range(self.totalHeight):
            row = [0] * width
            self.board.append(row)

        for _ in range(self.totalHeight):
            rows = [0] * (self.width)
            self.activeBoard.append(rows)

            
    def __str__(self):
        visible_board = self.board[self.hidden:]  # skip the top 'padding' rows
        if self.active: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in self.activeBoard[self.hidden:])
        else: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in visible_board)

    def newBlock(self): 
        if self.blocks == []:
            self.blocks = BLOCK_TYPES[:]
        random.shuffle(self.blocks)
        self.currentBlock = Block(self.blocks[0])
        self.blocks.pop(0)
        self.currentBlock.y = self.hidden
        self.drawBlock()
        
    def clearRows(self):
        visible_board = self.board[self.hidden:]
        updated_visible = [row for row in visible_board if not all(cell == 1 for cell in row)]
        rowsCleared = len(visible_board) - len(updated_visible)
        

        for _ in range(rowsCleared):
            updated_visible.insert(0, [0] * self.width)

        self.board = self.board[:self.hidden] + updated_visible
        return rowsCleared
    
    def checkTop(self):    
        for cell in self.board[self.hidden]:
            if cell != 0:
                return True
        return False
        
        
    # input current block x and y + 1 in whichever direction you want to move
    def canMove(self, block, newX, newY):
        # for loops figure out where the block actually is
        for y in range(4):
            for x in range(4):
                if block.block[y][x]: 
                    # need to adjust for where the 1 actually is in the 4x4 grid
                    boardX = newX + x
                    boardY = newY + y 

                    # check boundaries
                    if boardX < 0 or boardX >= self.width:
                        print('out of bounds')
                        return False
                    if boardY < 0 or boardY >= self.totalHeight:
                        print('at the top/bottom')
                        return False
                    
                    # it already is a 1
                    if self.board[boardY][boardX] == 1: 
                            return False
        return True


    def drawBlock(self):
        self.active = True
        # when it draws active board it will look like the saved board
        self.activeBoard = [row[:] for row in self.board] 

        for i in range(4):
            for j in range(4):
                if self.currentBlock.block[i][j]:
                    x = self.currentBlock.x + j
                    y = self.currentBlock.y + i
                    if 0 <= y < self.totalHeight and 0 <= x < self.width:
                        self.activeBoard[y][x] = 1

    def lockBlock(self):
        self.active = False
        for j in range(4):
            for i in range(4):
                if self.currentBlock.block[j][i]:
                    boardX = self.currentBlock.x + i
                    boardY = self.currentBlock.y + j
                    if 0 <= boardY < self.totalHeight and 0 <= boardX < self.width:
                        self.board[boardY][boardX] = 1
        self.currentBlock = None


class Tetris(GameBoard): 
    score = 0

    def __init__(self, height=20, width=10, padding=4):
        super().__init__(height, width, padding)

    def increaseScore(self, value):
        self.score += value
    
    def moveRight(self):
        if self.canMove(self.currentBlock, self.currentBlock.x + 1, self.currentBlock.y):
            self.currentBlock.x += 1
            self.drawBlock()
        else: 
            print('cant move right')

    def moveLeft(self):
        if self.canMove(self.currentBlock, self.currentBlock.x - 1, self.currentBlock.y):
            self.currentBlock.x -= 1
            self.drawBlock()
        else: 
            print('cant move left')

    def moveDown(self):
        if self.currentBlock is None:
            return False
        if self.canMove(self.currentBlock, self.currentBlock.x, self.currentBlock.y + 1):
            self.currentBlock.y +=1
            self.drawBlock()
        else: 
            self.lockBlock()
            print('cant move down')
            return False

    def rotateLeft(self):
        newConfig = self.currentBlock.clone()
        newConfig.rotate(3)

        candidate_offsets = [0, -1, 1, -2, 2]

        for offset in candidate_offsets:
            tempX = newConfig.x + offset
            
            if self.canMove(newConfig, tempX, newConfig.y):
                newConfig.x = tempX
                self.currentBlock = newConfig
                self.drawBlock()
                return
        print('cant rotate left')

    def rotateRight(self):
        newConfig = self.currentBlock.clone()
        newConfig.rotate(1)

        candidate_offsets = [0, -1, 1, -2, 2]

        for offset in candidate_offsets:
            tempX = newConfig.x + offset
            if self.canMove(newConfig, tempX, newConfig.y):
                newConfig.x = tempX
                self.currentBlock = newConfig
                self.drawBlock()
                return
        print('cant rotate right')

    def rotateFlip(self):
        newConfig = self.currentBlock.clone()
        newConfig.rotate(2)

        candidate_offsets = [0, -1, 1, -2, 2]

        for offset in candidate_offsets:
            tempX = newConfig.x + offset
            if self.canMove(newConfig, tempX, newConfig.y):
                newConfig.x = tempX
                self.currentBlock = newConfig
                self.drawBlock()
                return
        print("cant rotate 180")


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
def playGame(stdscr):
    curses.curs_set(0) # hide blinking terminal cursor
    stdscr.nodelay(True) # doesn't wait for keypress to do calls
    stdscr.timeout(100) # refresh rate 100ms

    while True:
        playing = True
        game = Tetris()
        score = 0
        currentCombo = 0
        ticks = 0
        
        while playing: 
            
            stdscr.clear() # refresh screen

            rowsCleared = game.clearRows()
            if rowsCleared == 0:
                currentCombo = 0
            else:
                currentCombo += 1
            
            if game.checkTop():
                playing = False
                break

            # basic system that attempts to increase scoring for bigger clears and longer combos
            score += (rowsCleared + currentCombo) ** 2 
            
            if game.currentBlock == None:
                game.newBlock()

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

            # draw board on terminal
            for i, line in enumerate(board_str):
                stdscr.addstr(start_y + i, start_x, line)
            stdscr.refresh()

            try:
                key = stdscr.getch() 
            except:
                key = -1
            
            if key == curses.KEY_LEFT:
                game.moveLeft()
            elif key == curses.KEY_RIGHT:
                game.moveRight()
            elif key == curses.KEY_DOWN:
                game.moveDown()
            elif key == ord('a'):
                game.rotateLeft()
            elif key == ord('s'):
                game.rotateFlip()
            elif key == ord('d'):
                game.rotateRight()
            
            # we can stop the game now
            elif key == ord('p'):
                game_over_screen(stdscr, score)

            ticks += 1
            if ticks % GRAVITY == 0:
                game.moveDown()

            time.sleep(0.05) # doesn't loop too fast
        if game_over_screen(stdscr, score):
            continue
        else:
            break


# curses.wrapper(playGame)
    
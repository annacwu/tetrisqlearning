import random

BLOCK_TYPES = ['O', 'L', 'J', 'I', 'T', 'Z', 'S']

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
            
    

class GameBoard:

    def __init__(self, height = 20, width = 10, padding = 4): # initialized to standard tetris board size plus padding
        self.height = height
        self.width = width 
        self.paddedHeight = height + padding
        self.paddedWidth = width + padding

        self.board = [] # board will be represented as 2d array (matrix)
        self.activeBoard = [] # this is the temporary board where we do the block rotation actions and stuff and check bounds
        self.currentBlock = None
        self.blocks = BLOCK_TYPES[:]
        self.active = False # for determining which board to show

        for _ in range(height):
            row = [0] * width
            self.board.append(row)

        for _ in range(self.paddedHeight):
            rows = [0] * (self.paddedWidth)
            self.activeBoard.append(rows)

            
    def __str__(self):
        if self.active: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in self.activeBoard) 
        else: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)

    def newBlock(self): 
        if self.blocks == []:
            self.blocks = BLOCK_TYPES[:]
        random.shuffle(self.blocks)
        print(self.blocks)
        self.currentBlock = Block(self.blocks[0])
        self.blocks.pop(0)
        
# GOAL: every move call this on all rows, and depending on how many rows return True increase score accordingly
    def checkClear(self, row):
        isFull = all(val == 1 for val in row) # checks if the whole row is 1s (full row)
        if isFull: 
            return True
        
    # input current block x and y + 1 in whichever direction you want to move
    def canMove(self, newX, newY):
        # for loops figure out where the block actually is
        for y in range(4):
            for x in range(4):
                if self.currentBlock.block[y][x]: 
                    # need to adjust for where the 1 actually is in the 4x4 grid
                    boardX = newX + x
                    boardY = newY + y 
                    # print('x: ' + str(boardX))
                    # print('y: ' + str(boardY))

                    # check boundaries
                    if boardX < 0 or boardX >= 10:
                        print('out of bounds')
                        return False
                    if boardY < 0 or boardY >= 20:
                        print('at the top/bottom')
                        return False
                    
                    # it already is a 1
                    if self.board[boardY][boardX] == 1: 
                            return False
        return True

    def draw(self):
        self.active = True
        self.activeBoard = [row[:] for row in self.board] # when it draws active board it will look like the saved board

        for i in range(4):
            for j in range(4):
                if self.currentBlock.block[i][j]:
                    x = self.currentBlock.x + j
                    y = self.currentBlock.y + i
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.activeBoard[y][x] = 1

    def lockBlock(self):
        # TODO: fix this so that it actually prints the board i wanna see
        self.active = False
        for j in range(4):
            for i in range(4):
                if self.currentBlock.block[j][i]:
                    boardX = self.currentBlock.x + i
                    boardY = self.currentBlock.y + j
                    self.board[boardY][boardX] = 1


class Tetris(GameBoard): 
    score = 0

    def __init__(self, height=20, width=10, padding=4):
        super().__init__(height, width, padding)

    def increaseScore(self, value):
        self.score += value

    def moveRight(self):
        if self.canMove(self.currentBlock.x + 1, self.currentBlock.y):
            self.currentBlock.x += 1
            self.draw()
        else: 
            print('cant move right')

    def moveLeft(self):
        if self.canMove(self.currentBlock.x - 1, self.currentBlock.y):
            self.currentBlock.x -= 1
            self.draw()
        else: 
            print('cant move left')

    def moveDown(self):
        if self.canMove(self.currentBlock.x, self.currentBlock.y + 1):
            self.currentBlock.y +=1
            self.draw()
        else: 
            self.lockBlock()
            print('cant move down')
            return False

    def rotateLeft(self):
        newConfig = self.currentBlock.rotate(3)
        if self.canMove(newConfig.x, newConfig.y):
            self.currentBlock = newConfig
            self.draw()
        else: 
            print('cant rotate left')

    def rotateRight(self):
        newConfig = self.currentBlock.rotate(1)
        if self.canMove(newConfig.x, newConfig.y):
            self.currentBlock = newConfig
            self.draw()
        else: 
            print('cant rotate right')

    def rotateFlip(self):
        newConfig = self.currentBlock.rotate(2)
        if self.canMove(newConfig.x, newConfig.y):
            self.currentBlock = newConfig
            self.draw()
        else: 
            print('cant rotate 180')


g = Tetris()
g.newBlock()
g.newBlock()
print(g)
    
###
# CURRENT FLOW: 
# g = Tetris() makes a new game board 
# g.newBlock() summons a block
# g.draw() turns on the active board and adds the new block to it
# print(g) will print the active board if run after draw()
# g.move_____ moves in the direction wanted unless it cant
#   if it can't move down it should lock, meaning it keeps it at the bottom
# 
# NOTE: any time you want to see the board as moves are happening, you must do: 
# g.move___
# g.draw()
# print(g) 
# and that will visualize the board after the move for you    
    
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
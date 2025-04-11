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
    board = [] # board will be represented as 2d array (matrix)
    activeBoard = [] # this is the temporary board where we do the block rotation actions and stuff and check bounds
    currentBlock = None
    blocks = BLOCK_TYPES
    active = False # for determining which board to show


    def __init__(self, height = 20, width = 10, padding = 4): # initialized to standard tetris board size plus padding
        self.height = height
        self.width = width 

        for _ in range(height):
            row = [0] * width
            self.board.append(row)

        for _ in range(height + padding):
            row = [0] * (width + padding)
            self.activeBoard.append(row)
            
    def __str__(self):
        if self.active: 
            return "\n".join(" ".join(str(cell) for cell in row) for row in self.activeBoard) 
        return "\n".join(" ".join(str(cell) for cell in row) for row in self.board)

    def newBlock(self): 
        if self.blocks == []:
            self.blocks = BLOCK_TYPES[:]
            random.shuffle(self.blocks)
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
                    print('x: ' + str(boardX))
                    print('y: ' + str(boardY))

                    # check boundaries
                    if boardX < 0 or boardX >= 10:
                        print('out of bounds')
                        return False
                    if boardY < 0 or boardY >= 20:
                        print('at the top')
                        return False
                    
                    # it already is a 1
                    if self.board[boardY][boardX] == 1: 
                            return False
        return True

    def draw(self):
        self.active = True
        self.activeBoard = [row[:] for row in self.board]

        for i in range(4):
            for j in range(4):
                if self.currentBlock.block[i][j]:
                    x = self.currentBlock.x + j
                    y = self.currentBlock.y + i
                    if 0 <= y < self.height and 0 <= x < self.width:
                        self.activeBoard[y][x] = 1

    def lockBlock(self):
        for j in range(4):
            for i in range(4):
                if self.currentBlock.block[j][i]:
                    boardX = self.currentBlock.x + i
                    boardY = self.currentBlock.y + j
                    self.board[boardY][boardX] = 1


class Tetris(GameBoard): 
    score = 0

    def __init__(self, height=24, width=10):
        super().__init__(height, width)

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


g = Tetris()
g.newBlock()
g.newBlock()
print(g.currentBlock)
g.draw()
print(g)
print(g.currentBlock.x)
g.moveRight()
g.moveRight()
g.moveRight()
g.moveRight()
g.moveRight()
print(g.currentBlock.x)
print(g)
    
    
    
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
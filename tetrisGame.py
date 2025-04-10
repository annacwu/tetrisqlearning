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
        num_rotations = len[BLOCKS[self.type]]
        self.rotation = (self.rotation + x) % num_rotations
        self.block = BLOCKS[self.type][self.rotation]
            
        
        
        
        
        
    

class GameBoard:
    board = [] # board will be represented as 2d array (matrix)
    currentBlock = None
    blocks = BLOCK_TYPES

    def __init__(self, height = 24, width = 10): # initialized to standard tetris board size
        self.height = height
        self.width = width + 4 # add padding for 4 x 4 blocks
        for i in range(height):
            row = [0] * width
            self.board.append(row)
            
    def __str__(self):
        return "\n".join([" ".join(map(str, row)) for row in self.board]) 

    def newBlock(self): 
        if self.blocks == []:
            self.blocks = BLOCK_TYPES
        else: 
            random.shuffle(self.blocks)
            self.currentBlock = Block(self.blocks[0])
            self.blocks.pop(0)
        
# TODO: CHANGE THIS BASED ON PADDING NOW        
# GOAL: every move call this on all rows, and depending on how many rows return True increase score accordingly
    def checkClear(self, row):
        isFull = all(val == 1 for val in row[2:12]) # checks if the whole row is 1s (full row)
        if isFull: 
            return True

    # def collides(self, pos1, pos2):
    #     if pos1 and pos2 == 1:
    #         return True
    #     return False
        
    # input current block x and y + 1 in whichever direction you want to move
    def canMove(self, newX, newY):
        # for loops figure out where the block actually is
        for y in range(4):
            for x in range(4):
                if self.currentBlock[y] and self.currentBlock[x]: 
                    # need to adjust for where the 1 actually is in the 4x4 grid
                    boardX = newX + x
                    boardY = newY + y 

                    # check boundaries
                    if boardX < 2 or boardX > 11:
                        return False
                    if boardY < 0 or boardY > 19:
                        return False
                    
                    # it already is a 1
                    if self.board[boardX][boardY] == 1: 
                        return False
        
        return True
    


        
        

class Tetris: 
    score = 0

    def increaseScore(self, value):
        self.score += value
    
    
    
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




def display():
    pass
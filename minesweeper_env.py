import pygame
import numpy as np
import torch
import random
from collections import deque



### NROWS, NCOLS, MINES
DIFFICULTY_LEVELS = {
    'VeryEasy': (5, 5, 3),
    'Beginner': (9,9,10),
    'Intermediate': (16,16,40),
    'Expert': (21, 21, 90)
}
NUM_ROWS = DIFFICULTY_LEVELS['VeryEasy'][0]
NUM_COLS = DIFFICULTY_LEVELS['VeryEasy'][1]
NUM_MINES = DIFFICULTY_LEVELS['VeryEasy'][2]
##

WINDOW_SIZE = (1000, 1000)
TILE_CODES = {'unrevealed': -3, 'revealed': 1, 'bomb': -1, 'flag': 2}
DROWS = [-1, 0, 1]
DCOLS = DROWS.copy()
pygame.init()
pygame.font.init()
pygame.display.set_caption("Minesweeper")
screen = pygame.display.set_mode(WINDOW_SIZE)
SIZE = WINDOW_SIZE[0] // NUM_ROWS
FONT = pygame.font.SysFont('Arial', int(WINDOW_SIZE[1] * 25 / 600))



#COLORS
CLICKED_TILE_COLOR = (20, 20, 20)
FLAGGED_TILE_COLOR = (200, 200, 0)
BG_COLOR = (0, 0, 0)
NUM_COLORS = {-1: (255, 192, 203), 0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128), 5: (128, 0, 32), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)}
TILE_COLOR = (100, 100, 100)



class MinesweeperEnv:
    def __init__(self, critic=None):
        self.critic = critic
        self.grid = self.create_grid(NUM_ROWS, NUM_COLS, NUM_MINES)
        self.cover_grid = [[-3 for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
        self.merged_grid = self.get_merged_grid(self.grid, self.cover_grid)
        self.game_started = False
        self.reset()

    def reset(self):
        self.grid = self.create_grid(NUM_ROWS, NUM_COLS, NUM_MINES)
        self.cover_grid = [[-3 for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
        self.merged_grid = self.get_merged_grid(self.grid, self.cover_grid)
        self.game_started = False
        return self.get_state()

    def get_state(self):
        return torch.FloatTensor(np.array(self.merged_grid).flatten()).unsqueeze(0)

    def get_merged_grid(self, grid, cover_grid):
        merged_grid = []
        for i in range(len(grid)):
            merged_row = []
            for j in range(len(grid[i])):
                if cover_grid[i][j] == 1:
                    merged_row.append(grid[i][j])
                else:
                    merged_row.append(-3) #-3 means tile is unrevealed. testing issues with ambiguity between cover_grid[r][c] = 0 (unrevealed tile) and grid[r][c] = 0 (revealed tile with 0 bombs nearby) in the merged_grid
            merged_grid.append(merged_row)
        return merged_grid
    
    def create_grid(self,rows,cols, mines, start_pos=(0,0)):
        grid = [[0 for _ in range(cols)] for _ in range(rows)]

        mine_indexes = set()
        safe_area = self.create_safearea(start_pos)

        while len(mine_indexes) < mines:
            row, col = random.randint(0,rows-1), random.randint(0,cols-1)
            pos = row, col

            if (row,col) in safe_area:
                continue
            if pos in mine_indexes:
                continue

            mine_indexes.add(pos)
            grid[row][col] = TILE_CODES['bomb']

        for mine in mine_indexes:
            neighbours = self.get_tile_neighbours(*mine, rows, cols)
            for r,c in neighbours:
                if grid[r][c] != TILE_CODES['bomb']:
                    grid[r][c] += 1

        return grid

    def get_tile_neighbours(self,row,col,rows,cols):
        neighbours = []

        for drow_offset in DROWS:
            for dcol_offset in DCOLS:
                if drow_offset == 0 and dcol_offset == 0:
                    continue
                drow = row + drow_offset
                dcol = col + dcol_offset
                if 0 <= drow < rows and 0 <= dcol < cols:
                    neighbours.append((drow, dcol))

        return neighbours

    def create_safearea(self,start_pos):
        safe_area = set()
        for drow_offset in DROWS:
            for dcol_offset in DCOLS:
                drow = start_pos[0] + drow_offset
                dcol = start_pos[1] + dcol_offset
                if 0 <= drow < NUM_ROWS and 0 <= dcol < NUM_COLS:    
                    safe_area.add((drow, dcol))
        return safe_area

    def step(self, action):
        won = False
        row = action // NUM_COLS
        col = action % NUM_COLS


        if self.cover_grid[row][col] == 1:
            reward = -1  
            done = True
            value = self.get_value_estimate()
            return self.get_state(), reward, done, won, value
        
        # print('row, col:',row,col, '\n', 'value at covergrid[row][col]: ',self.cover_grid[row][col])
        visited_mask = [[0 for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]

        if self.cover_grid[row][col] == -3:
            self.cover_grid[row][col] = 1
            list_neighbouring_empty_tiles = grid_bfs((row,col), self.grid, visited_mask)
            for r,c in list_neighbouring_empty_tiles:
                self.cover_grid[r][c] = 1

            if self.grid[row][col] > 0:
                neighbours = basic_middle_button((row,col),self.grid,self.cover_grid)
                for r,c in neighbours:
                    self.cover_grid[r][c] = 1
                    if self.grid[r][c] == 0:
                        visited_mask = [[0 for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]
                        list_neighbouring_empty_tiles = grid_bfs((r,c), self.grid, visited_mask)
                        for r,c in list_neighbouring_empty_tiles:
                            self.cover_grid[r][c] = 1
        
        reward = 0
        self.merged_grid = self.get_merged_grid(self.grid, self.cover_grid)
        done = self.check_gameover() or self.check_win()
        if self.check_win():
            reward = 1 
            won = True
        elif self.check_gameover():
            reward = -1
        elif self.check_random_guess(row,col):
            reward = -0.3
        else:
            tile_value = self.grid[row][col]
            if tile_value == 0:
                reward = 0.4
            elif tile_value > 0:
                reward = 0.3 + 0.1 * tile_value
            else:
                reward = 0.3

        value = self.get_value_estimate()
        return self.get_state(), reward, done, won, value


    def get_value_estimate(self):
        if self.critic is not None:
            state = self.get_state()
            with torch.no_grad():
                value = self.critic(state).item()
            return value
        return 0.0

    def check_random_guess(self, row, col):
        # Check if the tile is unrevealed
        if self.cover_grid[row][col] != -3:
            return False

        neighbours = self.get_tile_neighbours(row, col, NUM_ROWS, NUM_COLS)
        for r, c in neighbours:
            if self.cover_grid[r][c] == 1:
                return False  
        return True  
    
    def check_gameover(self):
        for i, row in enumerate(self.grid):
            for j, val in enumerate(row):
                if self.cover_grid[i][j] == 1 and val == -1:
                    return True
        return False

    def check_win(self):
        hidden_tiles = 0
        flagged_tiles = 0
        for i, row in enumerate(self.grid):
            for j, val in enumerate(row):
                if self.cover_grid[i][j] == -3:
                    hidden_tiles += 1
                if self.cover_grid[i][j] == 2:
                    flagged_tiles += 1
        return (hidden_tiles + flagged_tiles) == NUM_MINES

    def render(self):
        draw(screen, self.grid, self.cover_grid)


def basic_middle_button(pos,grid, cover_grid):
    x,y = pos
    ca8_neighbours = []
    clicked_tile_val = grid[x][y]
    flags_found = 0

    for drow_offset in DROWS:
            for dcol_offset in DCOLS:
                if drow_offset == 0 and dcol_offset == 0:
                    continue
                drow = x + drow_offset
                dcol = y + dcol_offset
                if 0 <= drow < NUM_ROWS and 0 <= dcol < NUM_COLS:
                    if cover_grid[drow][dcol] == 2:
                        flags_found += 1

    if flags_found < clicked_tile_val:
        return []

    for drow_offset in DROWS:
        for dcol_offset in DCOLS:
            if drow_offset == 0 and dcol_offset == 0:
                continue
            drow = x + drow_offset
            dcol = y + dcol_offset
            if 0 <= drow < NUM_ROWS and 0 <= dcol < NUM_COLS:
                if cover_grid[drow][dcol] != 2:
                    ca8_neighbours.append((drow, dcol))

    return ca8_neighbours

def grid_bfs(pos, grid, visited_mask):
    q = deque()
    x, y = pos
    ll=[]
    ll.append([(x,y)])
    q.append((x, y))

    while len(q) > 0:
        cell = q.popleft()
        x, y = cell

        if x < 0 or y < 0 or x >= len(grid) or y >= len(grid[0]):
            continue
        if visited_mask[x][y]:
            continue

        visited_mask[x][y] = 1
        if grid[x][y] == 0:
            neighbors = []

            for drow_offset in DROWS:
                for dcol_offset in DCOLS:
                    if drow_offset == 0 and dcol_offset == 0:
                        continue
                    drow = x + drow_offset
                    dcol = y + dcol_offset
                    if 0 <= drow < NUM_ROWS and 0 <= dcol < NUM_COLS:
                        if not visited_mask[drow][dcol]:
                            neighbors.append((drow, dcol))

            q.extend(neighbors)
            ll.append(neighbors)

    flat_ll = [x for xs in ll for x in xs]
    return flat_ll

def draw(window, field, cover):
    window.fill(BG_COLOR)
    flag = pygame.image.load(r"icons/pixil_flag.png").convert()
    bomb2= pygame.image.load(r"icons/pixil_bomb.png").convert()
    
    scaled_flag = pygame.transform.scale(flag,(SIZE,SIZE))
    scaled_bomb2 = pygame.transform.scale(bomb2,(SIZE,SIZE))
    for i, row in enumerate(field):
        x= i*SIZE
        for j, val in enumerate(row):
            y = j * SIZE

            rect = pygame.Rect(x,y,SIZE,SIZE)
            is_covered = cover[i][j] == -3
            is_flagged = cover[i][j] == 2

            if is_covered:
                pygame.draw.rect(screen,TILE_COLOR,rect)
                pygame.draw.rect(screen,'black',rect,2)
                continue
            elif is_flagged:
                pygame.draw.rect(screen,FLAGGED_TILE_COLOR,rect)
                window.blit(scaled_flag,(x,y))
                pygame.draw.rect(screen,'black',rect,2)
                continue
            else:
                pygame.draw.rect(screen,CLICKED_TILE_COLOR,rect)
                pygame.draw.rect(screen,'black',rect,2)

            if val != 0:
                txt = FONT.render(str(val),2,NUM_COLORS[val])
                window.blit(txt,(x + (SIZE/2 -txt.get_width()/2), y + (SIZE/2 - txt.get_height()/2)))
            if val == -1:
                window.blit(scaled_bomb2,(x,y))

    pygame.display.update()

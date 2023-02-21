from Board import SnakeGame
import numpy as np
import time
import pandas as pd


def V(x, w):
    return np.matmul(w.T, x)


boardDim = 20
game = SnakeGame(boardDim, boardDim)
w = np.fromfile('weights.dat')

print('weights: ', w)

game = SnakeGame(boardDim, boardDim)
x = []
down = []
up = []
right = []
left = []
gameOver = False
while not gameOver:
    game.display()
    x, down, right, up, left = game.calcAllValues()
    successors = np.array([V(down, w), V(right, w), V(up, w), V(left, w)])
    foodDirections = game.calcFoodDirection()
    direction = np.argmax(foodDirections)
    gameOver, length = game.makeMove(direction)

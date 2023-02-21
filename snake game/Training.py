'''
Sometimes the machine will get stuck in an infinite loop of non-scoring moves.
'''
import random
from Board import SnakeGame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import time


def V(x, w):
    return np.matmul(w.T, x)


boardDim = 20
game = SnakeGame(boardDim, boardDim)
w = np.zeros(4)
alpha = 0.000001
numEpochs = 5001
move_max_time = 7
print("Training for", numEpochs, "games...")

for epoch in range(numEpochs):
    print("number", epoch)
    start = datetime.now()
    game = SnakeGame(boardDim, boardDim)
    V_h = []
    V_train = []
    x = []
    down = []
    up = []
    right = []
    left = []
    gameOver = False
    start_time = time.time()
    while not gameOver:
        current_time = time.time()
        x, down, right, up, left = game.calcAllValues()

        V_h = V(x, w)

        successors = np.array([V(down, w), V(right, w), V(up, w), V(left, w)])
        elapsed_time = current_time - start_time

        if elapsed_time > move_max_time:

            print("loop iterating in: " +
                  str(int(elapsed_time)) + " seconds")
            break

        foodDirections = game.calcFoodDirection()

        direction = np.argmax(foodDirections)

        gameOver, length = game.makeMove(direction)
        elapsed_time = current_time - start_time

        if elapsed_time > move_max_time:

            print("loop iterating in: " +
                  str(int(elapsed_time)) + " seconds")
            break

        win = game.win()
        if win:
            V_train = 1000
        else:
            V_train = successors[direction]

    for i in range(len(w)):
        w[i] = w[i] + alpha*(V_train-V_h)*x[i]

    print('weights:')
    print(w)
    w.tofile('weights.dat')

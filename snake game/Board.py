import numpy as np


import random


class BodyNode():

    def __init__(self, parent, x, y):

        self.parent = parent

        self.x = x

        self.y = y

    def setX(self, x):

        self.x = x

    def setY(self, y):

        self.y = y

    def setParent(self, parent):

        self.parent = parent

    def getPosition(self):

        return (self.x, self.y)

    def getIndex(self):

        return (self.y, self.x)


class Snake():

    def __init__(self, x, y):

        self.head = BodyNode(None, x, y)

        self.tail = self.head

    def moveBodyForwards(self):

        currentNode = self.tail

        while currentNode.parent != None:

            parentPosition = currentNode.parent.getPosition()

            currentNode.setX(parentPosition[0])

            currentNode.setY(parentPosition[1])

            currentNode = currentNode.parent

    def move(self, direction):

        (oldTailX, oldTailY) = self.tail.getPosition()

        self.moveBodyForwards()

        headPosition = self.head.getPosition()

        if direction == 0:

            self.head.setY(headPosition[1] - 1)

        elif direction == 1:

            self.head.setX(headPosition[0] + 1)

        elif direction == 2:

            self.head.setY(headPosition[1] + 1)

        elif direction == 3:

            self.head.setX(headPosition[0] - 1)

        return (oldTailX, oldTailY, *self.head.getPosition())

    def newHead(self, newX, newY):

        newHead = BodyNode(None, newX, newY)

        self.head.setParent(newHead)

        self.head = newHead

    def getHead(self):

        return self.head

    def getTail(self):

        return self.tail


class SnakeGame():

    def __init__(self, width, height):

        self.headVal = 2

        self.bodyVal = 1

        self.foodVal = 7

        self.obstacleVal = -1

        self.width = width

        self.height = height

        self.board = np.zeros([height, width], dtype=int)

        self.length = 1

        startX = random.randint(0, width-1)

        startY = random.randint(0, height-1)

        self.board[startX, startY] = self.headVal

        self.snake = Snake(startY, startX)

        self.insertObstacle()

        self.spawnFood()

    def insertObstacle(self, count=2):

        i = 0

        while i < count:

            x = random.randint(0, self.width-1)

            y = random.randint(0, self.height-1)

            if x < 1 | y < 1:

                continue

            if self.board[x, y] == 0 & self.board[x-1, y] == 0 & self.board[x, y-1] == 0 & self.board[x-1, y-1] == 0:

                self.board[x, y] = self.obstacleVal

                self.board[x-1, y] = self.obstacleVal

                self.board[x, y-1] = self.obstacleVal

                self.board[x-1, y-1] = self.obstacleVal

                i += 1

    def spawnFood(self):

        emptyCells = []

        for index, value in np.ndenumerate(self.board):

            if value != self.bodyVal and value != self.headVal:

                emptyCells.append(index)

        self.foodIndex = random.choice(emptyCells)

        self.board[self.foodIndex] = self.foodVal

    def checkValid(self, direction):

        newX, newY = self.potentialPosition(direction)

        if newX == -1 or newX == self.width:

            return False

        if newY == -1 or newY == self.height:

            return False

        if self.board[newY, newX] == self.bodyVal:

            return False

        if self.board[newY, newX] == self.obstacleVal:

            return False

        return True

    def potentialPosition(self, direction):

        (newX, newY) = self.snake.getHead().getPosition()

        if direction == 0:

            newY -= 1

        elif direction == 1:

            newX += 1

        elif direction == 2:

            newY += 1

        elif direction == 3:

            newX -= 1

        return (newX, newY)

    def calcAllFeatures(self):

        (x, y) = self.snake.getHead().getPosition()

        return self.getFeatures(x, y), self.getFeatures(x, y-1), self.getFeatures(x+1, y), self.getFeatures(x, y+1), self.getFeatures(x-1, y)

    def getFeatures(self, x, y):

        F = []

        if x == -1 or x == self.width or y == -1 or y == self.height or self.board[y, x] == self.bodyVal or self.board[y, x] == self.obstacleVal:

            F.append(1000)

        else:

            F.append(1)

        F.append(self.boardDistance(x, y))

        F.append(self.foodDistance(x, y))

        F.append(self.obstacleDistance(x, y))

        return F

        pass

    def boardDistance(self, x, y):

        (x, y) = self.snake.getHead().getPosition()

        xBorad = self.width

        yBorad = self.height

        dx = abs(xBorad-x)

        dx = min(x, dx)

        dy = abs(yBorad-y)

        dy = min(y, dy)

        return dx+dy

    def getFoodPos(self):

        x = 0

        y = 0

        for i in range(self.width):

            for j in range(self.height):

                if self.board[i, j] == self.foodVal:

                    x = i

                    y = j

        return(x, y)

    def foodDistance(self, x, y):

        (x, y) = self.snake.getHead().getPosition()

        i, j = self.getFoodPos()

        dx = abs(x-i)

        dy = abs(y-j)

        return dx+dy

    def obstacleDistance(self, x, y):

        (x, y) = self.snake.getHead().getPosition()

        res = []

        matrix = self.board

        for i in range(len(matrix)):

            for j in range(len(matrix[i])):

                if matrix[i][j] == self.obstacleVal:

                    res.append((i, j))

                    pass

                pass

            pass

        dist = []

        for item in res:

            dx = abs(x-item[0])

            dy = abs(y-item[1])

            dist.append(dx+dy)

        return np.min(dist)

    def calcFoodDirection(self):

        foodDirections = np.zeros(4, dtype=int)

        dist = np.array(self.foodIndex) - \
            np.array(self.snake.getHead().getIndex())

        if dist[0] < 0:

            foodDirections[0] = 1

        elif dist[0] > 0:

            foodDirections[2] = 1

        if dist[1] > 0:

            foodDirections[1] = 1

        elif dist[1] < 0:

            foodDirections[3] = 1

        return foodDirections

    def plottableBoard(self):

        board = np.zeros([self.width, self.height])

        currentNode = self.snake.tail

        count = 0

        while True:

            count += 1

            board[currentNode.getIndex()] = 0.2 + 0.8*count/self.length

            currentNode = currentNode.parent

            if currentNode == None:

                break

        board[self.foodIndex] = -1

        return board

    def display(self):

        for i in range(self.width+2):

            print('-', end='')

        for i in range(self.height):

            print('\n|', end='')

            for j in range(self.width):

                if self.board[i, j] == 0:

                    print(' ', end='')

                elif self.board[i, j] == self.headVal:

                    print('O', end='')

                elif self.board[i, j] == self.bodyVal:

                    print('X', end='')

                elif self.board[i, j] == self.foodVal:

                    print('*', end='')

                elif self.board[i, j] == self.obstacleVal:

                    print('&', end='')

            print('|', end='')

        print()

        for i in range(self.width+2):

            print('-', end='')

        print()

    def makeMove(self, direction):

        gameOver = False

        if self.checkValid(direction):

            (headX, headY) = self.snake.getHead().getPosition()

            self.board[headY, headX] = self.bodyVal

            potX, potY = self.potentialPosition(direction)

            if self.board[potY, potX] == self.foodVal:

                self.snake.newHead(potX, potY)

                self.board[potY, potX] = self.headVal

                self.spawnFood()

                self.length += 1

            else:

                (oldTailX, oldTailY, newHeadX, newHeadY) = self.snake.move(direction)

                self.board[oldTailY, oldTailX] = 0

                self.board[newHeadY, newHeadX] = self.headVal

        else:

            gameOver = True

        return (gameOver, self.length)

    def win(self):

        win = False

        for i in range(self.width):

            for j in range(self.height):

                if self.board[i, j] != 0:

                    win = True

                else:

                    win = False

        return (win)

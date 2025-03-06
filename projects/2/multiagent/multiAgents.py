# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        # Obtain successor state information
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()  
        newFood = successorGameState.getFood()  
        newGhostStates = successorGameState.getGhostStates()  
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]  

        # Initialize a basic score using the current state's score
        score = successorGameState.getScore()

        # Compute distance between Pacman and the closest food
        foodList = newFood.asList()
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            score += 1.0 / min(foodDistances)  # The closer the distance, the higher the score, so use the inverse of the distance

        # Compute distance between Pacman and ghosts
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if newScaredTimes[i] > 0:  # If the ghost is in a scared state, encourage Pacman to approach the ghost
                score += 200.0 / ghostDistance  
            else:
                if ghostDistance <= 1:  # Punish the score if ghost is too near
                    score -= 1000.0

        return score

    



















def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(agentIndex, depth, gameState):
            # If the game is over or the maximum depth is reached, return the evaluation score of the current state
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's round (max layer)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghost's round (min layer)
            else:
                return minValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            # Pacman's round
            v = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)
            bestAction = None
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # Recursively call minimax
                value = minimax(1, depth, successor)
                if value > v:
                    v = value
                    bestAction = action
            if depth == 0:
                return bestAction  
            return v

        def minValue(agentIndex, depth, gameState):
            # Ghost's round
            v = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1  # Switch to next ghost
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  # Return to Pacman
                depth += 1  
            
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, depth, successor))
            return v

        return minimax(0, 0, gameState)  # Start the recursive search from Pacman's perspective


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            v = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)
            bestAction = None
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = alphaBeta(1, depth, successor, alpha, beta)
                if value > v:
                    v = value
                    bestAction = action
                # Alpha-Beta Pruning
                alpha = max(alpha, v)
                if v > beta:
                    return v
            if depth == 0:
                return bestAction  
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            v = float('inf')
            legalActions = gameState.getLegalActions(agentIndex)
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():  
                nextAgent = 0
                depth += 1  

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphaBeta(nextAgent, depth, successor, alpha, beta))
                # Alpha-Beta Pruning
                beta = min(beta, v)
                if v < alpha:
                    return v
            return v

        # Initially, the call starts with Pacman, and the initial values of alpha and beta are negative infinity and positive infinity, respectively
        return alphaBeta(0, 0, gameState, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            else:
                return expectedValue(agentIndex, depth, gameState)

        def maxValue(agentIndex, depth, gameState):
            v = float('-inf')
            legalActions = gameState.getLegalActions(agentIndex)
            bestAction = None
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = expectimax(1, depth, successor)  
                if value > v:
                    v = value
                    bestAction = action
            if depth == 0:
                return bestAction  
            return v

        def expectedValue(agentIndex, depth, gameState):
            legalActions = gameState.getLegalActions(agentIndex)
            v = 0
            numActions = len(legalActions)
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():  
                nextAgent = 0
                depth += 1  

            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v += expectimax(nextAgent, depth, successor) / numActions  
            return v

        return expectimax(0, 0, gameState)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    
    DESCRIPTION:
    This evaluation function considers several key factors:
    - Distance to the closest food
    - Number of remaining food pellets
    - Distance to ghosts (penalizing proximity to non-scared ghosts)
    - Distance to scared ghosts (rewarding proximity to scared ghosts)
    - Distance to power capsules
    - Current game score
    
    The function aims to balance aggressive food collection while avoiding dangerous ghosts.
    """
    pacmanPos = currentGameState.getPacmanPosition()
    
    food = currentGameState.getFood().asList()
    
    ghostStates = currentGameState.getGhostStates()
    
    capsules = currentGameState.getCapsules()
    
    score = currentGameState.getScore()
    
    foodDistance = 0
    if food:
        foodDistance = min([manhattanDistance(pacmanPos, foodPos) for foodPos in food])
    
    remainingFood = len(food)
    
    ghostPenalty = 0
    scaredGhostBonus = 0
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)
        if ghostState.scaredTimer > 0:
            scaredGhostBonus += 200.0 / (ghostDistance + 1)
        else:
            if ghostDistance > 0:
                ghostPenalty += 10.0 / ghostDistance
    
    capsuleBonus = 0
    if capsules:
        capsuleDistance = min([manhattanDistance(pacmanPos, capPos) for capPos in capsules])
        capsuleBonus = 100.0 / (capsuleDistance + 1)
    
    evaluation = score
    evaluation += 10.0 / (foodDistance + 1)  
    evaluation -= 4 * remainingFood          
    evaluation -= ghostPenalty             
    evaluation += scaredGhostBonus           
    evaluation += capsuleBonus              
    
    return evaluation


# Abbreviation
better = betterEvaluationFunction


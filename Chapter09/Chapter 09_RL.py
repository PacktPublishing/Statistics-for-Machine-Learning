

# Solving MDP with Dynamic Programming - Value & Policy Iterations

import random,operator


def argmax(seq, fn):
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score > best_score:
            best, best_score = x, x_score
    return best

def vector_add(a, b):
    return tuple(map(operator.add, a, b))
    
 
orientations = [(1,0), (0, 1), (-1, 0), (0, -1)]

def turn_right(orientation):
    return orientations[orientations.index(orientation)-1]

def turn_left(orientation):
    return orientations[(orientations.index(orientation)+1) % len(orientations)]

def isnumber(x):
    return hasattr(x, '__int__')


   
"""A Markov Decision Process, defined by an init_pos_posial state, transition model,
and reward function. """

class MDP:

    def __init__(self, init_pos, actlist, terminals, transitions={}, states=None, gamma=0.99):
        if not (0 < gamma <= 1):
            raise ValueError("MDP should have 0 < gamma <= 1 values")

        if states:
            self.states = states
        else:
            self.states = set()
        self.init_pos = init_pos
        self.actlist = actlist
        self.terminals = terminals
        self.transitions = transitions
        self.gamma = gamma
        self.reward = {}

    """Returns a numeric reward for the state."""
    def R(self, state):
        return self.reward[state]

    """Transition model. From a state and an action, return a list of (probability, result-state) pairs"""
    def T(self, state, action):
        if(self.transitions == {}):
            raise ValueError("Transition model is missing")
        else:
            return self.transitions[state][action]

    """Set of actions that can be performed for a particular state"""
    def actions(self, state):
        if state in self.terminals:
            return [None]
        else:
            return self.actlist



"""A two-dimensional grid MDP"""
class GridMDP(MDP):

    def __init__(self, grid, terminals, init_pos=(0, 0), gamma=0.99):
        
        """ because we want row 0 on bottom, not on top """
        grid.reverse()  
        
        MDP.__init__(self, init_pos, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]

    """Return the state that results from going in this direction."""
    def go(self, state, direction):
        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
    def to_grid(self, mapping):
        return list(reversed([[mapping.get((x, y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))
                
    """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
    def to_arrows(self, policy):
        chars = {
            (1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})

"""Solving an MDP by value iteration and returns the optimum state values """
def value_iteration(mdp, epsilon=0.001):
    STSN = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        STS = STSN.copy()
        delta = 0
        for s in mdp.states:
            STSN[s] = R(s) + gamma * max([sum([p * STS[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(STSN[s] - STS[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return STS

"""Given an MDP and a utility function STS, determine the best policy,
as a mapping from state to action """
def best_policy(mdp, STS):
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a: expected_utility(a, s, STS, mdp))
    return pi

"""The expected utility of doing a in state s, according to the MDP and STS."""
def expected_utility(a, s, STS, mdp):
    return sum([p * STS[s1] for (p, s1) in mdp.T(s, a)])

"""Solve an MDP by policy iteration"""
def policy_iteration(mdp):
    STS = {s: 0 for s in mdp.states}
    pi = {s: random.choice(mdp.actions(s)) for s in mdp.states}
    while True:
        STS = policy_evaluation(pi, STS, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s),lambda a: expected_utility(a, s, STS, mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

"""Return an updated utility mapping U from each state in the MDP to its
utility, using an approximation (modified policy iteration)"""
def policy_evaluation(pi, STS, mdp, k=20):
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            STS[s] = R(s) + gamma * sum([p * STS[s1] for (p, s1) in T(s, pi[s])])
    return STS


def print_table(table, header=None, sep='   ', numfmt='{}'):
    justs = ['rjust' if isnumber(x) else 'ljust' for x in table[0]]
    if header:
        table.insert(0, header)
    table = [[numfmt.format(x) if isnumber(x) else x for x in row]
             for row in table]
    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in table]))))
    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))



""" A 4x3 grid environment that presents the agent with a sequential decision problem"""
sequential_decision_environment = GridMDP([[-0.02, -0.02, -0.02, +1],
                                           [-0.02, None, -0.02, -1],
                                           [-0.02, -0.02, -0.02, -0.02]],
                                          terminals=[(3, 2), (3, 1)])

# Value Iteration
value_iter = best_policy(sequential_decision_environment, value_iteration(sequential_decision_environment, .01))
print("\n Optimal Policy based on Value Iteration\n")
print_table(sequential_decision_environment.to_arrows(value_iter))


#Policy Iteration
policy_iter = policy_iteration(sequential_decision_environment)
print("\n Optimal Policy based on Policy Iteration & Evaluation\n")
print_table(sequential_decision_environment.to_arrows(policy_iter))




# Monte Carlo Methods

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  
actions = [ACTION_HIT, ACTION_STAND]



"policy for player"
policyPlayer = np.zeros(22)

for i in range(12, 20):
    policyPlayer[i] = ACTION_HIT

policyPlayer[20] = ACTION_STAND
policyPlayer[21] = ACTION_STAND

"function form of target policy of player"
def targetPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    return policyPlayer[playerSum]

"function form of behavior policy of player"
def behaviorPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

"policy for dealer"
policyDealer = np.zeros(22)
for i in range(12, 17):
    policyDealer[i] = ACTION_HIT
for i in range(17, 22):
    policyDealer[i] = ACTION_STAND

"get a new card"
def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# play a game

def play(policyPlayerFn, initialState=None, initialAction=None):
    # player status

    # sum of player
    playerSum = 0

    # trajectory of player
    playerTrajectory = []

    # whether player uses Ace as 11
    usableAcePlayer = False

    # dealer status
    dealerCard1 = 0
    dealerCard2 = 0
    usableAceDealer = False

    if initialState is None:
        # generate a random initial state

        numOfAce = 0

        # initialize cards of player
        while playerSum < 12:
            # if sum of player is less than 12, always hit
            card = getCard()

            # if get an Ace, use it as 11
            if card == 1:
                numOfAce += 1
                card = 11
                usableAcePlayer = True
            playerSum += card

        # if player's sum is larger than 21, he must hold at least one Ace, two Aces are possible
        if playerSum > 21:
            # use the Ace as 1 rather than 11
            playerSum -= 10

            # if the player only has one Ace, then he doesn't have usable Ace any more
            if numOfAce == 1:
                usableAcePlayer = False

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealerCard1 = getCard()
        dealerCard2 = getCard()

    else:
        # use specified initial state
        usableAcePlayer = initialState[0]
        playerSum = initialState[1]
        dealerCard1 = initialState[2]
        dealerCard2 = getCard()

    # initial state of the game
    state = [usableAcePlayer, playerSum, dealerCard1]

    # initialize dealer's sum
    dealerSum = 0
    if dealerCard1 == 1 and dealerCard2 != 1:
        dealerSum += 11 + dealerCard2
        usableAceDealer = True
    elif dealerCard1 != 1 and dealerCard2 == 1:
        dealerSum += dealerCard1 + 11
        usableAceDealer = True
    elif dealerCard1 == 1 and dealerCard2 == 1:
        dealerSum += 1 + 11
        usableAceDealer = True
    else:
        dealerSum += dealerCard1 + dealerCard2

    # game starts!

    # player's turn
    while True:
        if initialAction is not None:
            action = initialAction
            initialAction = None
        else:
            # get action based on current sum
            action = policyPlayerFn(usableAcePlayer, playerSum, dealerCard1)

        # track player's trajectory for importance sampling
        playerTrajectory.append([action, (usableAcePlayer, playerSum, dealerCard1)])

        if action == ACTION_STAND:
            break
        # if hit, get new card
        playerSum += getCard()

        # player busts
        if playerSum > 21:
            # if player has a usable Ace, use it as 1 to avoid busting and continue
            if usableAcePlayer == True:
                playerSum -= 10
                usableAcePlayer = False
            else:
                # otherwise player loses
                return state, -1, playerTrajectory

    # dealer's turn
    while True:
        # get action based on current sum
        action = policyDealer[dealerSum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        dealerSum += getCard()
        # dealer busts
        if dealerSum > 21:
            if usableAceDealer == True:
            # if dealer has a usable Ace, use it as 1 to avoid busting and continue
                dealerSum -= 10
                usableAceDealer = False
            else:
            # otherwise dealer loses
                return state, 1, playerTrajectory

    # compare the sum between player and dealer
    if playerSum > dealerSum:
        return state, 1, playerTrajectory
    elif playerSum == dealerSum:
        return state, 0, playerTrajectory
    else:
        return state, -1, playerTrajectory

# Monte Carlo Sample with On-Policy
def monteCarloOnPolicy(nEpisodes):
    statesUsableAce = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    statesUsableAceCount = np.ones((10, 10))
    statesNoUsableAce = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    statesNoUsableAceCount = np.ones((10, 10))
    for i in range(0, nEpisodes):
        state, reward, _ = play(targetPolicyPlayer)
        state[1] -= 12
        state[2] -= 1
        if state[0]:
            statesUsableAceCount[state[1], state[2]] += 1
            statesUsableAce[state[1], state[2]] += reward
        else:
            statesNoUsableAceCount[state[1], state[2]] += 1
            statesNoUsableAce[state[1], state[2]] += reward
    return statesUsableAce / statesUsableAceCount, statesNoUsableAce / statesNoUsableAceCount

# Monte Carlo with Exploring Starts
def monteCarloES(nEpisodes):
    # (playerSum, dealerCard, usableAce, action)
    stateActionValues = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    stateActionPairCount = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    def behaviorPolicy(usableAce, playerSum, dealerCard):
        usableAce = int(usableAce)
        playerSum -= 12
        dealerCard -= 1
        # get argmax of the average returns(s, a)
        return np.argmax(stateActionValues[playerSum, dealerCard, usableAce, :]
                      / stateActionPairCount[playerSum, dealerCard, usableAce, :])

    # play for several episodes
    for episode in range(nEpisodes):
        if episode % 1000 == 0:
            print('episode:', episode)
        # for each episode, use a randomly initialized state and action
        initialState = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initialAction = np.random.choice(actions)
        _, reward, trajectory = play(behaviorPolicy, initialState, initialAction)
        for action, (usableAce, playerSum, dealerCard) in trajectory:
            usableAce = int(usableAce)
            playerSum -= 12
            dealerCard -= 1
            # update values of state-action pairs
            stateActionValues[playerSum, dealerCard, usableAce, action] += reward
            stateActionPairCount[playerSum, dealerCard, usableAce, action] += 1

    return stateActionValues / stateActionPairCount


# print the state value
figureIndex = 0
def prettyPrint(data, tile, zlabel='reward'):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(tile)
    ax = fig.add_subplot(111, projection='3d')
    x_axis = []
    y_axis = []
    z_axis = []
    for i in range(12, 22):
        for j in range(1, 11):
            x_axis.append(i)
            y_axis.append(j)
            z_axis.append(data[i - 12, j - 1])
    ax.scatter(x_axis, y_axis, z_axis,c='red')
    ax.set_xlabel('player sum')
    ax.set_ylabel('dealer showing')
    ax.set_zlabel(zlabel)



# On-Policy results
def onPolicy():
    statesUsableAce1, statesNoUsableAce1 = monteCarloOnPolicy(10000)
    statesUsableAce2, statesNoUsableAce2 = monteCarloOnPolicy(500000)
    prettyPrint(statesUsableAce1, 'Usable Ace & 10000 Episodes')
    prettyPrint(statesNoUsableAce1, 'No Usable Ace & 10000 Episodes')
    prettyPrint(statesUsableAce2, 'Usable Ace & 500000 Episodes')
    prettyPrint(statesNoUsableAce2, 'No Usable Ace & 500000 Episodes')
    plt.show()
    
    
# Optimized or Monte Calro Control 
def MC_ES_optimalPolicy():
    stateActionValues = monteCarloES(500000)
    stateValueUsableAce = np.zeros((10, 10))
    stateValueNoUsableAce = np.zeros((10, 10))
    # get the optimal policy
    actionUsableAce = np.zeros((10, 10), dtype='int')
    actionNoUsableAce = np.zeros((10, 10), dtype='int')
    for i in range(10):
        for j in range(10):
            stateValueNoUsableAce[i, j] = np.max(stateActionValues[i, j, 0, :])
            stateValueUsableAce[i, j] = np.max(stateActionValues[i, j, 1, :])
            actionNoUsableAce[i, j] = np.argmax(stateActionValues[i, j, 0, :])
            actionUsableAce[i, j] = np.argmax(stateActionValues[i, j, 1, :])
    prettyPrint(stateValueUsableAce, 'Optimal state value with usable Ace')
    prettyPrint(stateValueNoUsableAce, 'Optimal state value with no usable Ace')
    prettyPrint(actionUsableAce, 'Optimal policy with usable Ace', 'Action (0 Hit, 1 Stick)')
    prettyPrint(actionNoUsableAce, 'Optimal policy with no usable Ace', 'Action (0 Hit, 1 Stick)')
    plt.show()



# Run on policy function
onPolicy()

# Run Monte Carlo Control or Explored starts
MC_ES_optimalPolicy()











# Cliff-Walking - TD Learning - SARSA & Q-Learning
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions
GRID_HEIGHT = 4
GRID_WIDTH = 12

# probability for exploration, step size,gamma 
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1

# all possible actions
ACTION_UP = 0; ACTION_DOWN = 1;ACTION_LEFT = 2;ACTION_RIGHT = 3
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
stateActionValues = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
startState = [3, 0]
goalState = [3, 11]

# reward for each action in each state
actionRewards = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
actionRewards[:, :, :] = -1.0
actionRewards[2, 1:11, ACTION_DOWN] = -100.0
actionRewards[3, 0, ACTION_RIGHT] = -100.0

# set up destinations for each action in each state
actionDestination = []
for i in range(0, GRID_HEIGHT):
    actionDestination.append([])
    for j in range(0, GRID_WIDTH):
        destinaion = dict()
        destinaion[ACTION_UP] = [max(i - 1, 0), j]
        destinaion[ACTION_LEFT] = [i, max(j - 1, 0)]
        destinaion[ACTION_RIGHT] = [i, min(j + 1, GRID_WIDTH - 1)]
        if i == 2 and 1 <= j <= 10:
            destinaion[ACTION_DOWN] = startState
        else:
            destinaion[ACTION_DOWN] = [min(i + 1, GRID_HEIGHT - 1), j]
        actionDestination[-1].append(destinaion)
actionDestination[3][0][ACTION_RIGHT] = startState

# choose an action based on epsilon greedy algorithm
def chooseAction(state, stateActionValues):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(actions)
    else:
        return np.argmax(stateActionValues[state[0], state[1], :])


# SARSA update

def sarsa(stateActionValues, expected=False, stepSize=ALPHA):
    currentState = startState
    currentAction = chooseAction(currentState, stateActionValues)
    rewards = 0.0
    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        newAction = chooseAction(newState, stateActionValues)
        reward = actionRewards[currentState[0], currentState[1], currentAction]
        rewards += reward
        if not expected:
            valueTarget = stateActionValues[newState[0], newState[1], newAction]
        else:
            valueTarget = 0.0
            actionValues = stateActionValues[newState[0], newState[1], :]
            bestActions = np.argwhere(actionValues == np.max(actionValues))
            for action in actions:
                if action in bestActions:
                    valueTarget += ((1.0 - EPSILON) / len(bestActions) + EPSILON / len(actions)) * stateActionValues[newState[0], newState[1], action]
                else:
                    valueTarget += EPSILON / len(actions) * stateActionValues[newState[0], newState[1], action]
        valueTarget *= GAMMA
        stateActionValues[currentState[0], currentState[1], currentAction] += stepSize * (reward +
            valueTarget - stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
    return rewards

# Q-Learning update
def qLearning(stateActionValues, stepSize=ALPHA):
    currentState = startState
    rewards = 0.0
    while currentState != goalState:
        currentAction = chooseAction(currentState, stateActionValues)
        reward = actionRewards[currentState[0], currentState[1], currentAction]
        rewards += reward
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        stateActionValues[currentState[0], currentState[1], currentAction] += stepSize * (
            reward + GAMMA * np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
    return rewards

# print optimal policy
def printOptimalPolicy(stateActionValues):
    optimalPolicy = []
    for i in range(0, GRID_HEIGHT):
        optimalPolicy.append([])
        for j in range(0, GRID_WIDTH):
            if [i, j] == goalState:
                optimalPolicy[-1].append('G')
                continue
            bestAction = np.argmax(stateActionValues[i, j, :])
            if bestAction == ACTION_UP:
                optimalPolicy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimalPolicy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimalPolicy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimalPolicy[-1].append('R')
    for row in optimalPolicy:
        print(row)

def SARSAnQLPlot():
    # averaging the reward sums from 10 successive episodes
    averageRange = 10

    # episodes of each run
    nEpisodes = 500

    # perform 20 independent runs
    runs = 20

    rewardsSarsa = np.zeros(nEpisodes)
    rewardsQLearning = np.zeros(nEpisodes)
    for run in range(0, runs):
        stateActionValuesSarsa = np.copy(stateActionValues)
        stateActionValuesQLearning = np.copy(stateActionValues)
        for i in range(0, nEpisodes):
            # cut off the value by -100 to draw the figure more elegantly
            rewardsSarsa[i] += max(sarsa(stateActionValuesSarsa), -100)
            rewardsQLearning[i] += max(qLearning(stateActionValuesQLearning), -100)

    # averaging over independt runs
    rewardsSarsa /= runs
    rewardsQLearning /= runs

    # averaging over successive episodes
    smoothedRewardsSarsa = np.copy(rewardsSarsa)
    smoothedRewardsQLearning = np.copy(rewardsQLearning)
    for i in range(averageRange, nEpisodes):
        smoothedRewardsSarsa[i] = np.mean(rewardsSarsa[i - averageRange: i + 1])
        smoothedRewardsQLearning[i] = np.mean(rewardsQLearning[i - averageRange: i + 1])

    # display optimal policy
    print('Sarsa Optimal Policy:')
    printOptimalPolicy(stateActionValuesSarsa)
    print('Q-Learning Optimal Policy:')
    printOptimalPolicy(stateActionValuesQLearning)

    # draw reward curves
    plt.figure(1)
    plt.plot(smoothedRewardsSarsa, label='Sarsa')
    plt.plot(smoothedRewardsQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()


# Sum of Rewards for SARSA vs. QLearning
SARSAnQLPlot()






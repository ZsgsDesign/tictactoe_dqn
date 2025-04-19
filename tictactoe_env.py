import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TicTacToeEnv(gym.Env):
    """
    0: vacant, 1: agent, -1: opponent
    agent always plays "1" and is the second player.
    NOTE: opponent moves first, to train the agent.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, opponent='random'):
        super().__init__()
        # 3x3 board，flat 9 dimension
        # 0: empty, 1: agent, -1: opponent
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self.opponent = opponent
        self.reset()

    def reset(self, seed=None, options=None):
        self.board = np.zeros(9, dtype=np.int8)
        self.done = False
        # opponent move first
        self._opponent_move()
        return self.board.copy(), {'board': self.board.copy()}

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")
        reward = 0
        # agent move
        if self.board[action] != 0:
            # illegal move
            reward = -10
            self.done = True
            return self.board.copy(), reward, self.done, False, {}
        self.board[action] = 1
        # check winner
        winner = self._check_winner(self.board)
        if winner != 0:
            self.done = True
            reward = 1 if winner == 1 else -1
            return self.board.copy(), reward, self.done, False, {}

        # opponent moves
        self._opponent_move()
        winner = self._check_winner(self.board)
        if winner != 0:
            self.done = True
            reward = 1 if winner == 1 else -1
            return self.board.copy(), reward, self.done, False, {}
        # draw
        if np.all(self.board != 0):
            self.done = True
            reward = 0
        return self.board.copy(), reward, self.done, False, {}

    def render(self, mode='human'):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        bf = self.board.reshape(3,3)
        print("\n".join(["|".join([symbols[x] for x in row]) for row in bf]))
        print("-"*6)

    def _check_winner(self, board: np.ndarray) -> int:
        b = board.reshape(3,3)
        lines = []
        lines += list(b)  # rows
        lines += list(b.T)  # cols
        lines.append(b.diagonal())
        lines.append(np.fliplr(b).diagonal())
        for line in lines:
            s = np.sum(line)
            if s == 3: return 1
            if s == -3: return -1
        return 0

    def _opponent_move(self):
        empties = np.where(self.board == 0)[0]
        if len(empties)==0: return
        if self.opponent == 'random':
            a = self.np_random.choice(empties)
        else:
            # extendable
            # e.g. simple strategy, Minimax, etc.
            a = self.np_random.choice(empties)
        self.board[a] = -1
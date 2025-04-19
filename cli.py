from os import system
import os
import time
from typing import Dict, Tuple
from colorama import Fore
import numpy as np
from stable_baselines3 import DQN

def load_model(path="saved_model/tictactoe_dqn.zip"):
    return DQN.load(path)

model = load_model()

obs = [
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
]

def predict(obs: np.ndarray) -> int:
    action, _ = model.predict(obs, deterministic=True)
    return action

def clear():
    system('cls' if os.name == 'nt' else 'clear')

def draw_board(board: np.ndarray) -> Tuple[Dict[int, str], np.ndarray]:
    clear()
    print("🔴 AI vs 🟢 Player")
    print("-"*12)
    symbols = {1: f'🔴', -1: f'{Fore.RED}🟢{Fore.RESET}', 0: '⚪'} # 🔴🟠🟡🟢🔵🟣⚫⚪🟤 
    bf = board.reshape(3,3)
    print("\n".join(["".join([symbols[x] for x in row]) for row in bf]))
    print("-"*12)
    return symbols, bf


def check_winner(board: np.ndarray) -> int:
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

def welcome_and_init():
    config = {
        'first_mode': False,
    }
    clear()
    print(f"Welcome to {Fore.MAGENTA}Tic Tac Toe!{Fore.RESET}")
    should_i_play_first = input("Do you want to play first? (y/n): ").strip().lower()
    if should_i_play_first == 'y':
        config['first_mode'] = False
    else:
        config['first_mode'] = True
    clear()
    print(f"Welcome to {Fore.MAGENTA}Tic Tac Toe!{Fore.RESET}")
    first_role = "AI" if config['first_mode'] else "Player"
    print(f"You are 🟢 and AI is 🔴\n{Fore.YELLOW}{first_role}{Fore.RESET} will play first")
    input("Press Enter to start...")
    clear()
    return config

if __name__ == "__main__":
    config = welcome_and_init()
    board = np.zeros(9, dtype=np.int8)
    done = False
    while not done:
        symbols, bf = draw_board(board)
        winner = check_winner(board)
        if winner == 1:
            print(f"{Fore.RED}AI Wins!{Fore.RESET}")
            break
        elif winner == -1:
            print(f"{Fore.GREEN}You win!{Fore.RESET}")
            break
        elif np.all(board != 0):
            print(f"{Fore.YELLOW}Draw!{Fore.RESET}")
            break
        if not config['first_mode']:
            # print only the empty spaces
            for i in range(3):
                print('|', end="")
                for j in range(3):
                    if board[i*3+j] == 0:
                        print(f"{Fore.YELLOW}{i*3+j}{Fore.RESET}", end="|")
                    else:
                        print(f" ", end="|")
                print()
            while True:
                print("Please choose a position (0-8): ", end="")
                action = int(input())
                if action < 0 or action > 8 or board[action] != 0:
                    # Invalid move
                    print(f"{Fore.RED}Invalid move! Please try again.{Fore.RESET}")
                    continue
                break
            board[action] = -1
        else:
            config["first_mode"] = False
        symbols, bf = draw_board(board)
        time.sleep(2)
        winner = check_winner(board)
        if winner == 1:
            print(f"{Fore.RED}AI Wins!{Fore.RESET}")
            break
        elif winner == -1:
            print(f"{Fore.GREEN}You win!{Fore.RESET}")
            break
        elif np.all(board != 0):
            print(f"{Fore.YELLOW}Draw!{Fore.RESET}")
            break
        obs = board.copy().astype(np.float32)
        action = predict(obs)
        board[action] = 1
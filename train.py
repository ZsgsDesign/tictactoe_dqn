import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from tictactoe_env import TicTacToeEnv

def main():
    # create environment
    train_env = TicTacToeEnv(opponent='random')
    eval_env  = TicTacToeEnv(opponent='random')

    # DQN
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        verbose=1,
    )

    # evaluation at a fixed interval
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # start training
    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
        progress_bar=True
    )

    # save the model
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/tictactoe_dqn")

if __name__=="__main__":
    main()
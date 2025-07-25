# DRL TVDT from Stable Baselines 3

import time

import numpy as np
import pandas as pd
from MySAC import config
from MySAC.SAC.MAE_SAC import SAC as SAC_MAE
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results




MODELS = {"maesac": SAC_MAE}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class CheckCallback(BaseCallback):
    def __init__(self, check_freq:int, verbose: int=1):
        super(CheckCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(self.n_calls)

class oursTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq:int, log_dir: str, verbose: int=1):
        super(oursTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 10 episodes
              mean_reward = np.mean(y[-50:])
              if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best TVDT, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best TVDT
                  if self.verbose > 0:
                    print(f"Saving new best TVDT to {self.save_path}")
                  self.model.save(self.save_path+'model.zip')

        return True      


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained TVDT
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
        seed=None,
        tensorboard_log=None,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, check_freq, ck_dir, log_dir, eval_env, total_timesteps=5000, verbose=1, deterministic=True):
        eval_callback = EvalCallback(eval_env, best_model_save_path=ck_dir, log_path=log_dir, eval_freq=check_freq, n_eval_episodes=1, deterministic=deterministic, render=False)
        tb_callback=TensorboardCallback(verbose=verbose)
        callback = CallbackList([eval_callback, tb_callback])

        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback = callback
        )
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]#, universal_results[0]

    @staticmethod
    # def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
    #     test_env, test_obs = environment.get_sb_env()
    #     if model_name not in MODELS:
    #         raise NotImplementedError("NotImplementedError")
    #     try:
    #         # load agent
    #         model = MODELS[model_name].load(cwd)
    #         print("Successfully load TVDT", cwd)
    #     except BaseException:
    #         raise ValueError("Fail to load agent!")
    #
    #     # test on the testing env
    #     state = environment.reset()
    #     episode_returns = list()  # the cumulative_return / initial_account
    #     episode_total_assets = list()
    #     episode_total_assets.append(environment.initial_amount)
    #     done = False
    #     while not done:
    #         action = model.predict(state, deterministic=deterministic)[0]
    #         state, reward, done, _ = environment.step(action)
    #
    #         total_asset = environment.end_total_asset
    #
    #         episode_total_assets.append(total_asset)
    #         episode_return = total_asset / environment.initial_amount
    #         episode_returns.append(episode_return)
    #
    #     print("episode_return", episode_return)
    #     print("Test Finished!")
    #
    #     account_memory = test_env.env_method(method_name="save_asset_memory")
    #     actions_memory = test_env.env_method(method_name="save_action_memory")
    #
    #     return episode_total_assets, account_memory[0], actions_memory[0]
    def DRL_prediction_load_from_file(model_name, environment, cwd, deterministic=True):
        test_env, test_obs = environment.get_sb_env()
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            # load agent
            model = MODELS[model_name].load(cwd)
            print("Successfully load TVDT", cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state = environment.reset()
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_amount)
        max_total_asset = environment.initial_amount
        drawdowns = []  # Store drawdown values
        accumulate_return = []  # Store drawdown values
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = environment.step(action)

            total_asset = environment.end_total_asset

            episode_total_assets.append(total_asset)
            episode_return = total_asset / environment.initial_amount
            episode_returns.append(episode_return)

            # Compute drawdown
            max_total_asset = max(max_total_asset, total_asset)
            drawdown = (total_asset - max_total_asset) / max_total_asset
            drawdowns.append(drawdown)
            accumulate_return.append(episode_return)

        print("episode_return", episode_return)
        print("Test Finished!")

        account_memory = test_env.env_method(method_name="save_asset_memory")

        dates = account_memory[0]["date"].tolist()  # Convert the 'date' column to a list

        accumulate_return_df = pd.DataFrame({
            "date": dates,
            "accumulate_returen": accumulate_return
        })
        # Combine the date and drawdown into a DataFrame
        drawdown_df = pd.DataFrame({
            "date": dates,
            "drawdown": drawdowns[:len(dates)]  # Ensure the length matches
        })
        episode_total_assets_series = pd.Series(episode_total_assets).iloc[:-1]
        return_day = episode_total_assets_series / episode_total_assets_series.shift(60) - 1  # 60-day return
        # Calculate 60-day maximum drawdown
        rolling_max = episode_total_assets_series.rolling(window=60, min_periods=1).max()
        drawdown_day = (rolling_max - episode_total_assets_series) / rolling_max
        drawdown_day.replace(0, np.nan, inplace=True)  # 将 0 变为 NaN，避免除以 0
        # Calculate 60-day Calmar ratio
        calmar = return_day / drawdown_day

        # Adding the Calmar ratio to a DataFrame
        calmar_df = pd.DataFrame({
            "date": dates,  # Aligning the dates for 60-day calculations
            "calmar_ratio": calmar  # Skipping the first 60 days
        })
        actions_memory = test_env.env_method(method_name="save_action_memory")

        return episode_total_assets, account_memory[0], actions_memory[0], drawdown_df, accumulate_return_df, calmar_df

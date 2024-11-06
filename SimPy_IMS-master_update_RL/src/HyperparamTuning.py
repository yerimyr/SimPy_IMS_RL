##### 최적의 하이퍼파라미터(learning rate, gamma)를 찾음 #####
import GymWrapper as gw
import optuna.visualization as vis
import optuna
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from GymWrapper import GymInterface


def tuning_hyperparam(trial):
    # Initialize the environment
    env = GymInterface()
    env.reset()
    # Define search space for hyperparameters  # optuna의 trial객체를 사용하여 하이퍼파라미터의 탐색 공간을 정의함
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1)  # 1e-5부터 1 사이의 값을 로그 스케일로 탐색하여 학습률을 선택함
    gamma = trial.suggest_float('gamma', 0.9, 0.9999, log=True)  # 0.9부터 0.9999사이의 값을 로그 스케일로 탐색하여 gamma를 선택함
    batch_size = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256])  # 16,32,64,128,256 중 하나의 배치 크기를 선택함
    # Define the RL model
    if RL_ALGORITHM == "DQN":
        model = DQN("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "DDPG":
        model = DDPG("MlpPolicy", env, learning_rate=learning_rate,
                     gamma=gamma, batch_size=batch_size, verbose=0)
    elif RL_ALGORITHM == "PPO":
        model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                    gamma=gamma, batch_size=batch_size, n_steps=SIM_TIME, verbose=0)
    # Train the model
    model.learn(total_timesteps=SIM_TIME*N_EPISODES)  # learn()메서드를 사용하여 모델을 학습함
    # Evaluate the model
    eval_env = GymInterface()  # GymInterface에서 모델의 성능 측정
    mean_reward, _ = gw.evaluate_model(model, eval_env, N_EVAL_EPISODES)  # 학습된 모델의 성능을 측정하고 평균 보상을 측정함

    return -mean_reward  # Minimize the negative of mean reward


def run_optuna():  # Optuna를 사용하여 강화학습 모델의 하이퍼파라미터 최적화를 수행
    # study = optuna.create_study( )
    study = optuna.create_study(direction='minimize')  # create_study()를 사용하여 하이퍼파라미터 최적화를 위한 study를 생성함, direction='minimize: 최적화의 목표를 최소화로 설정함
    study.optimize(tuning_hyperparam, n_trials=N_TRIALS)  # 하이퍼파라미터 최적화를 tuning_hyperparam을 호출하여 n_trials만큼 수행함

    # Print the result
    best_params = study.best_params  # 최적의 하이퍼파라미터 출력
    print("Best hyperparameters:", best_params)
    # Visualize hyperparameter optimization process
    vis.plot_optimization_history(study).show()  # 최적화 과정의 히스토리를 시각화하여 각 실험에서의 성능을 확인함(어떻게 개선되었는지 확인)
    vis.plot_parallel_coordinate(study).show()  # 최적의 하이퍼파라미터가 어떤 조합에서 나왔는지 확인할 수 있는 평행 좌표 플롯을 생성함
    vis.plot_slice(study).show()  # 각 하이퍼파라미터 값에 따른 성능 변화를 시각화
    vis.plot_contour(study, params=['learning_rate', 'gamma']).show()  # 두 개의 하이퍼파라미터(learning rate, gamma)의 조합에 따른 성능 변화를 등고선 플롯으로 시각화 

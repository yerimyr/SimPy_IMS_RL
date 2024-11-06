import os
import shutil
from config_SimPy import *

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
# BEST_PARAMS = {'learning_rate': 0.000171573369797847,
#                'gamma': 0.9380991034336233, 'batch_size': 16}

ACTION_SPACE = [0, 1, 2, 3, 4, 5] #####

'''
# State space
STATE_RANGES = []
for i in range(len(I)):
    # Inventory level
    STATE_RANGES.append((0, INVEN_LEVEL_MAX))
    # Daily change for the on-hand inventory
    STATE_RANGES.append((-INVEN_LEVEL_MAX, INVEN_LEVEL_MAX))
# Remaining demand: Demand quantity - Current product level
STATE_RANGES.append((0, max(DEMAND_QTY_MAX, INVEN_LEVEL_MAX)))
'''
# Find minimum Delta
PRODUCT_OUTGOING_CORRECTION = 0  # 이 변수는 최종적으로 제품의 생산량을 조정하는데 사용
for key in P:  # P라는 딕셔너리의 모든 항목을 순회
    PRODUCT_OUTGOING_CORRECTION = max(P[key]["PRODUCTION_RATE"] *
                                      max(P[key]['QNTY_FOR_INPUT_ITEM']), DEMAND_QTY_MAX)  # 제품 생산량과 최대 수요량 중 더 큰 값을 선택
# maximum production

# Episode
N_EPISODES = 3000  # 3000  # 시뮬레이션이나 학습에서 반복 횟수를 나타냄

'''
def DEFINE_FOLDER(folder_name):  # 주로 훈련 결과나 로그, 데이터, 모델 체크포인트 등을 저장하는 데 사용
    if os.path.exists(folder_name):
        file_list = os.listdir(folder_name)
        folder_name = os.path.join(folder_name, f"Train_{len(file_list)+1}")  # 폴더가 존재하는 경우, 해당 폴더에 "Train_X" 형식의 새로운 폴더 이름을 추가
    else:
        folder_name = os.path.join(folder_name, "Train_1")  # 폴더가 없으면 "Train_1"이라는 이름의 폴더를 새로 정의
    os.makedirs(folder_name, exist_ok=True)  
    return folder_name
'''

def DEFINE_FOLDER(folder_name):
    base_folder = folder_name
    counter = 1
    while os.path.exists(os.path.join(base_folder, f"Train_{counter}")):
        counter += 1
    folder_name = os.path.join(base_folder, f"Train_{counter}")
    os.makedirs(folder_name)
    return folder_name


def save_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)  # 만약 path가 존재하면, 해당 폴더와 그 안의 모든 파일 및 하위 폴더를 삭제
    # Create a new folder
    os.makedirs(path)  # 주어진 path에 새로운 폴더를 생성
    return path


# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 15  # 50  # 하이퍼파라미터 최적화를 위해 시도할 횟수를 설정

#RL_Options
USE_CORRECTION=False  # 강화 학습 과정에서 수정을 사용할지를 설정

# Evaluation
N_EVAL_EPISODES = 100  # 100  

# Export files
DAILY_REPORT_EXPORT = True
STATE_TRAIN_EXPORT = True
STATE_TEST_EXPORT = True

# Define parent dir's path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
# Define each dir's parent dir's path
tensorboard_folder = os.path.join(parent_dir, "tensorboard_log")
result_csv_folder = os.path.join(parent_dir, "result_CSV")
STATE_folder = os.path.join(result_csv_folder, "state")
daily_report_folder = os.path.join(result_csv_folder, "daily_report")

# Define dir's path
TENSORFLOW_LOGS = DEFINE_FOLDER(tensorboard_folder)
'''
STATE = DEFINE_FOLDER(STATE_folder)
REPORT_LOGS = DEFINE_FOLDER(daily_report_folder)
GRAPH_FOLDER = DEFINE_FOLDER(graph_folder)
'''
STATE = save_path(STATE_folder)
REPORT_LOGS = save_path(daily_report_folder)

# Makedir
'''
if os.path.exists(STATE):
    pass
else:
    os.makedirs(STATE)

if os.path.exists(REPORT_LOGS):
    pass
else:
    os.makedirs(REPORT_LOGS)
if os.path.exists(GRAPH_FOLDER):
    pass
else:
    os.makedirs(GRAPH_FOLDER)
'''
# Visualize_Graph
VIZ_INVEN_LINE = True
VIZ_INVEN_PIE = True
VIZ_COST_PIE = True
VIZ_COST_BOX = True

# Saved Model
SAVED_MODEL_PATH = os.path.join(parent_dir, "Saved_Model")
SAVE_MODEL = False
SAVED_MODEL_NAME = "DQN_MODEL_test_val"

# Load Model
LOAD_MODEL = False
LOAD_MODEL_NAME = "DQN_MODEL_SIM500"

# Non-stationary demand
mean_demand = 100
standard_deviation_demand = 20


# tensorboard --logdir="~\tensorboard_log"
# http://localhost:6006/

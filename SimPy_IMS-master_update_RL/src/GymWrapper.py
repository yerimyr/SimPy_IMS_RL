import gym
from gym import spaces
import numpy as np
from config_SimPy import *
from config_RL import *
import environment as env
from log_SimPy import *
from log_RL import *
import pandas as pd
import matplotlib.pyplot as plt
import Visualization
from torch.utils.tensorboard import SummaryWriter


class GymInterface(gym.Env):
    def __init__(self):
        self.writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)  # 강화학습 학습 과정 기록
        os = []
        super(GymInterface, self).__init__()
        # Action space, observation space
        if RL_ALGORITHM == "DQN":
            # Define action space
            self.action_space = spaces.Discrete(len(ACTION_SPACE))  # ACTION_SPACE = [0,1,2,3,4,5]
            # Define observation space:
            max_inventory_level = INVEN_LEVEL_MAX * 5
            num_inventory_items = len(I)
            if USE_CORRECTION:
                os = [max_inventory_level for _ in range(num_inventory_items * (1 + DAILY_CHANGE) + MAT_COUNT * INTRANSIT + 1)]
            else:
                os = [max_inventory_level for _ in range(num_inventory_items * (1 + DAILY_CHANGE) + MAT_COUNT * INTRANSIT + 1)]
                
            self.observation_space = spaces.MultiDiscrete(os)
        elif RL_ALGORITHM == 'PPO':
            self.action_space = spaces.Discrete(len(ACTION_SPACE))
            
            max_inventory_level = INVEN_LEVEL_MAX * 5
            num_inventory_items = len(I)
            if USE_CORRECTION:
                os = [max_inventory_level for _ in range(num_inventory_items * (1 + DAILY_CHANGE) + MAT_COUNT * INTRANSIT + 1)]
            else:
                os = [max_inventory_level for _ in range(num_inventory_items * (1 + DAILY_CHANGE) + MAT_COUNT * INTRANSIT + 1)]

            self.observation_space = spaces.MultiDiscrete(os)
            '''
            self.observation_space = spaces.Box(low=0, high=INVEN_LEVEL_MAX * 2, shape=(6,), dtype=np.int32)
            os = []
            os=[INVEN_LEVEL_MAX*2 for _ in range(len(I)*(1+DAILY_CHANGE)+MAT_COUNT*INTRANSIT+1)]
            self.observation_space = spaces.MultiDiscrete(os)
    
            self.observation_space = spaces.MultiDiscrete(os)  # os = [재고수준최대값, 수요량최대값+보정값, 수요량최대값]
            '''
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.num_episode = 1

        # For functions that only work when testing the model
        self.model_test = False
        # Record the cumulative value of each cost
        self.cost_ratio = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        
        scenario = {"DEMAND": DEMAND_SCENARIO, "LEADTIME": LEADTIME_SCENARIO}  # 강화학습이 진행되는 환경
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = env.create_env(
            I, P, LOG_DAILY_EVENTS)  # environment.py에서 create_env를 가져와서 진행
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.supplierList, self.daily_events, I, scenario)  # environment.py에서 simpy_event_processes를 가져와서 진행
        env.update_daily_report(self.inventoryList)  # environment.py에서 update_daily_report를 가져와서 진행
        
    def reset(self):
        self.cost_ratio = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        # Initialize the simulation environment
        print("\nEpisode: ", self.num_episode)
        scenario = {"DEMAND": DEMAND_SCENARIO, "LEADTIME": LEADTIME_SCENARIO}
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.supplierList, self.daily_events = env.create_env(
            I, P, LOG_DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                          self.productionList, self.sales, self.customer, self.supplierList, self.daily_events, I, scenario)

        env.update_daily_report(self.inventoryList)

        # print("==========Reset==========")
        self.shortages = 0
        state_real=self.get_current_state()  # 시뮬레이션 내에서 현재 상태를 정확하게 추적하고자 할 때 사용됨
        state_corr=self.correct_state_for_SB3()  # 강화학습 알고리즘이 요구하는 대로 조정 후 SB3 모델이 상태 정보를 올바르게 해석하고 학습할 수 있도록 함
        if USE_CORRECTION:
            state = state_corr
        else:
            state = state_real
    
        return np.array(state)  # 대부분의 강화학습 알고리즘은 numpy 배열 형태의 입력을 요구하므로 state를 numpy 배열로 변환하는 과정 필요함

    def step(self, action):

        # Update the action of the agent
        if RL_ALGORITHM == "DQN":
            I[1]["LOT_SIZE_ORDER"] = action  # LOT_SIZE_ORDER에 action값 할당
        elif RL_ALGORITHM == 'PPO':
            I[1]["LOT_SIZE_ORDER"] = action

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        # Action append
        STATE_ACTION_REPORT_CORRECTION[-1].append(action)  # STATE_ACTION_REPORT_CORRECTION(log.py)리스트에서 마지막 값을 action으로
        STATE_ACTION_REPORT_REAL[-1].append(action)
        
        self.simpy_env.run(until=self.simpy_env.now + 24)
        env.update_daily_report(self.inventoryList)

        # Capture the next state of the environment
        state_real=self.get_current_state()
        state_corr=self.correct_state_for_SB3()
        if USE_CORRECTION:
            next_state=state_corr
        else:
            next_state = state_real

        # Calculate the total cost of the day
        env.Cost.update_cost_log(self.inventoryList)
        if PRINT_SIM_EVENTS:
            cost = dict(DAILY_COST)  # DAILY_COST(log.py)를 cost에 딕셔너리 형태로 저장 후 이용할 것

        for key in DAILY_COST.keys():
            self.cost_ratio[key] += DAILY_COST[key]  # cost의 누적합계계산

        env.Cost.clear_cost()

        reward = -LOG_COST[-1]  # LOG_COST는 시뮬레이션 전체 기간 동안 발생한 비용을 누적하여 기록
        self.total_reward += reward
        #for sale in self.sales:
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0


        if PRINT_SIM_EVENTS:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "DQN":
                i = 0  # action[i]에서 인덱스로 사용됨
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Material":
                        if 0 <= action < len(ACTION_SPACE):
                            print(f"[Order Quantity for {I[_]['NAME']}] ", ACTION_SPACE[action])
                        else:
                            print(f"오류: action 값 {action}이 유효하지 않습니다.")
                        i += 1
            elif RL_ALGORITHM == "PPO":
                i = 0  # action[i]에서 인덱스로 사용됨
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Material":
                        if 0 <= action < len(ACTION_SPACE):
                            print(f"[Order Quantity for {I[_]['NAME']}] ", ACTION_SPACE[action])
                        else:
                            print(f"오류: action 값 {action}이 유효하지 않습니다.")
                        i += 1
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            for _ in cost.keys():
                print(_, cost[_])
            print("Total cost: ", -self.total_reward)

            if USE_CORRECTION:
                print("[CORRECTED_STATE for the next round] ", [item for item in next_state])
            else:
                print("[REAL_STATE for the next round] ",  [item-INVEN_LEVEL_MAX for item in next_state])   ##################

        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            self.writer.add_scalar(
                "reward", self.total_reward, global_step=self.num_episode)
            # Log each cost ratio at the end of the episode
            for cost_name, cost_value in self.cost_ratio.items():
                self.writer.add_scalar(
                    cost_name, cost_value, global_step=self.num_episode)  # 각 에피소드마다 cost_name과 cost_value를 tensorboard log에 기록

            print("Total reward: ", self.total_reward)
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0
            self.num_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)
        return np.array(next_state), reward, done, info
    '''
    def get_current_state(self):  # state = [on-hand inventory, daily_change in inventory, in-transition inventory, remainig demand, 여유공간 0으로 채우기]
        # Make State for RL
        temp = []
        # Update STATE_ACTION_REPORT_REAL
        for inven in self.inventoryList:
            # ID means Item_ID, 7 means to the length of the report for one item
            # append On_Hand_inventory
            temp.append(LOG_STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"]+INVEN_LEVEL_MAX)
            # append changes in inventory
            if DAILY_CHANGE==1:
                # append changes in inventory
                temp.append(LOG_STATE_DICT[-1][f"Daily_Change_{I[inven.item_id]['NAME']}"]+INVEN_LEVEL_MAX)
            if INTRANSIT==1:
                if I[id]["TYPE"]=="Material":
                    # append Intransition inventory
                    temp.append(LOG_STATE_DICT[-1][f"In_Transit_{I[inven.item_id]['NAME']}"]+INVEN_LEVEL_MAX)
        if len(self.inventoryList) > 0:
            temp.append(I[0]["DEMAND_QUANTITY"]-self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)  # append remaining demand
        else:
            print("Error: inventoryList is empty.")
            temp.append(INVEN_LEVEL_MAX)
        while len(temp) < 6:
            temp.append(0)
        STATE_ACTION_REPORT_REAL.append(temp)
        return STATE_ACTION_REPORT_REAL[-1]  # state 기록 및 반환
        '''
    def get_current_state(self):  #############################  # state = [product onhand, material onhand, intransit, remain demand]
        temp = []
        for inven in self.inventoryList:
            # On-hand inventory를 범위 내로 제한
            on_hand = LOG_STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"] + INVEN_LEVEL_MAX  # PRODUCT OnHand, MATERIAL OnHand
            temp.append(on_hand)

            # 수요량 변경 범위 확인
            if DAILY_CHANGE == 1:
                daily_change = LOG_STATE_DICT[-1][f"Daily_Change_{I[inven.item_id]['NAME']}"] + INVEN_LEVEL_MAX  # PRODUCT DailyChange, MATERIAL DailyChange
                temp.append(daily_change)

            # In-transit 범위 확인
            if INTRANSIT == 1 and I[inven.item_id]['TYPE'] == "Material":
                in_transit = LOG_STATE_DICT[-1][f"In_Transit_{I[inven.item_id]['NAME']}"]  # MATERIAL InTransit
                temp.append(in_transit)
            
        temp.append(I[0]["DEMAND_QUANTITY"]-self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)  # remain DEMAND
        STATE_ACTION_REPORT_REAL.append(temp)
        return STATE_ACTION_REPORT_REAL[-1]
    
        '''
        if len(self.inventoryList) > 0:
            remaining_demand = I[0]["DEMAND_QUANTITY"] - self.inventoryList[0].on_hand_inventory + INVEN_LEVEL_MAX  # Remain Demand
            temp.append(remaining_demand)  # append remaining demand
        else:
            print("Error: inventoryList is empty.")
            temp.append(INVEN_LEVEL_MAX)
    
        # 결과를 올바른 형태로 반환
        while len(temp) < 6:
            temp.append(0)
        print(temp)
        STATE_ACTION_REPORT_REAL.append(temp)
        return STATE_ACTION_REPORT_REAL[-1]
        '''

    # Min-Max Normalization
    def correct_state_for_SB3(self):
        # Find minimum Delta
        product_outgoing_correction = 0
        for key in P:
            # product_outgoing_correction = max(P[key]["PRODUCTION_RATE"] * max(P[key]['QNTY_FOR_INPUT_ITEM']), self.scenario["max"])
            product_outgoing_correction = max(
                P[key]["PRODUCTION_RATE"] * max(P[key]['QNTY_FOR_INPUT_ITEM']), INVEN_LEVEL_MAX)

        # Update STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        state_corrected = []
        for inven in self.inventoryList:
            # normalization Onhand inventory
            state_corrected.append(round((LOG_STATE_DICT[-1][f"On_Hand_{I[inven.item_id]['NAME']}"]/INVEN_LEVEL_MAX)*100))
            if DAILY_CHANGE==1:
                state_corrected.append(round(((LOG_STATE_DICT[-1][f"Daily_Change_{I[inven.item_id]['NAME']}"]-(-product_outgoing_correction))/(
                ACTION_SPACE[-1]-(-product_outgoing_correction)))*100))  # normalization changes in inventory
            if I[inven.item_id]['TYPE']=="Material":
                if INTRANSIT==1:
                    state_corrected.append(round((LOG_STATE_DICT[-1][f"In_Transit_{I[inven.item_id]['NAME']}"]-ACTION_SPACE[0])/(ACTION_SPACE[-1]-ACTION_SPACE[0])))
        
        state_corrected.append(round(
            ((I[0]["DEMAND_QUANTITY"]-self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)/(I[0]['DEMAND_QUANTITY']+INVEN_LEVEL_MAX))*100))
        STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        return STATE_ACTION_REPORT_CORRECTION[-1]
    
        '''
        # normalization remaining demand
        if len(self.inventoryList) > 0:
            state_corrected.append(round(
                ((I[0]["DEMAND_QUANTITY"]-self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)/(I[0]['DEMAND_QUANTITY']+INVEN_LEVEL_MAX))*100))
        else:
            print("Error: inventoryList is empty.")
            state_corrected.append(INVEN_LEVEL_MAX)
        while len(state_corrected) < 6:
            state_corrected.append(0)
        STATE_ACTION_REPORT_CORRECTION.append(state_corrected)
        return STATE_ACTION_REPORT_CORRECTION[-1]
        '''
    def render(self, mode='human'):
        pass

    def close(self):
        # 필요한 경우, 여기서 리소스를 정리
        pass


# Function to evaluate the trained model
def evaluate_model(model, env, num_episodes):
    all_rewards = []  # List to store total rewards for each episode  # 각 에피소드의 총 보상 저장
    # XAI = []  # List for storing data for explainable AI purposes
    
    STATE_ACTION_REPORT_REAL.clear()
    STATE_ACTION_REPORT_CORRECTION.clear()  # 평가해야하기때문에 이전 기록을 초기화
    ORDER_HISTORY = []  # 평가 중 기록할 데이터를 저장
    # For validation and visualization
    order_qty = []  # 평가 중 기록할 데이터를 저장
    demand_qty = []  # 평가 중 기록할 데이터를 저장
    onhand_inventory = []  # 평가 중 기록할 데이터를 저장
    test_order_mean = []  # List to store average orders per episode  # 평가 중 기록할 데이터를 저장 -> 에피소드 당 주문량의 평균을 저장
    for i in range(num_episodes):
        ORDER_HISTORY.clear()
        episode_inventory = [[] for _ in range(len(I))]
        LOG_DAILY_REPORTS.clear()  # Clear daily reports at the start of each episode
        obs = env.reset()  # Reset the environment to get initial observation 
        episode_reward = 0  # Initialize reward for the episode
        env.model_test = True
        done = False  # Flag to check if episode is finished
        day=1
        while not done:
            for x in range(len(env.inventoryList)):
                episode_inventory[x].append(  
                    env.inventoryList[x].on_hand_inventory)  # 재고 수준을 기록하여 episode_inventory에 저장함
            action, _ = model.predict(obs)  # Get action from mode: predict  # 모델이 예측한 행동을 model/presict(obs)로 가져옴
            # Execute action in environment => 현재 Material 1개에 대한 action만 코딩되어 있음. 추후 여러 Material에 대한 action을 코딩해야 함.
            #시뮬레이션 Validaition을 위한 코드 차후 지울것
            if VALIDATION:  # VALIDATION이 활성화된 경우
                action=validation_input(day)  # action을 조정할 수 있음
            obs, reward, done, _ = env.step(action)  # 환경에서 action을 실행하고 새로운 obs, reward, done(에피소드 종료 여부)를 반환함
            episode_reward += reward  # Accumulate rewards

            ORDER_HISTORY.append(action)  # Log order history  # 주문 기록 저장
            # ORDER_HISTORY.append(I[1]["LOT_SIZE_ORDER"])  # Log order history
            order_qty.append(action)  # 주문량 저장
            # order_qty.append(I[1]["LOT_SIZE_ORDER"])
            demand_qty.append(I[0]["DEMAND_QUANTITY"])  # 수요량 저장
            day+=1
        onhand_inventory.append(episode_inventory)  #onhand_inventory에 episode_inventory를 저장함
        all_rewards.append(episode_reward)  # Store total reward for episode

        # Function to visualize the environment

        # Calculate mean order for the episode
        test_order_mean.append(sum(ORDER_HISTORY) / len(ORDER_HISTORY))  # test_order_mean에는 에피소드 중 평균 주문량을 계산하여 저장함
        COST_RATIO_HISTORY.append(env.cost_ratio)  # 비용 비율을 저장함
    if VISUALIAZTION.count(1) > 0:  # VISUALIZATION이 설정된 경우
        Visualization.visualization(LOG_DAILY_REPORTS)  # 시각화 수행
    Visualize_invens(onhand_inventory, demand_qty, order_qty, all_rewards)  # 시각화할 요소들
    cal_cost_avg()  # 평균비용 계산
    # print("Order_Average:", test_order_mean)
    '''
    if XAI_TRAIN_EXPORT:
        df = pd.DataFrame(XAI)  # Create a DataFrame from XAI data
        df.to_csv(f"{XAI_TRAIN}/XAI_DATA.csv")  # Save XAI data to CSV file
    '''
    if STATE_TEST_EXPORT:
        export_state("TEST")
    # Calculate mean reward across all episodes
    mean_reward = np.mean(all_rewards)  # 평가된 모든 에피소드의 평균 보상을 계산
    std_reward = np.std(all_rewards)  # Calculate standard deviation of rewards  # 평가된 모든 에피소드의 평균 보상의 표준편차를 계산함

    return mean_reward, std_reward  # Return mean and std of rewards


def cal_cost_avg():
    # Temp_Dict
    cost_avg = {
        'Holding cost': 0,
        'Process cost': 0,
        'Delivery cost': 0,
        'Order cost': 0,
        'Shortage cost': 0
    }
    # Temp_List
    total_avg = []

    # Cal_cost_AVG
    for x in range(N_EVAL_EPISODES):
        for key in COST_RATIO_HISTORY[x].keys():
            cost_avg[key] += COST_RATIO_HISTORY[x][key]
        total_avg.append(sum(COST_RATIO_HISTORY[x].values()))
    for key in cost_avg.keys():
        cost_avg[key] = cost_avg[key]/N_EVAL_EPISODES
    # Visualize
    if VIZ_COST_PIE:  # cost_avg의 데이터를 시각화하여 각 비용 요소가 전체에서 차지하는 비율을 한눈에 파악 가능
        fig, ax = plt.subplots()
        plt.pie(cost_avg.values(), explode=[  
                0.2, 0.2, 0.2, 0.2, 0.2], labels=cost_avg.keys(), autopct='%1.1f%%')  # explode=[0.2, 0.2, 0.2, 0.2, 0.2]: 원형 차트에서 각 조각이 중심에서 떨어져 나오는 정도를 설정
        plt.show()
    if VIZ_COST_BOX:  # total_avg 데이터를 시각화하여 비용의 분포와 변동성을 확인 가능
        plt.boxplot(total_avg)
        plt.show()


def Visualize_invens(inventory, demand_qty, order_qty, all_rewards):
    best_reward = -99999999999999
    best_index = 0
    for x in range(N_EVAL_EPISODES):
        if all_rewards[x] > best_reward:
            best_reward = all_rewards[x]
            best_index = x

    avg_inven = [[0 for _ in range(SIM_TIME)] for _ in range(len(I))]
    lable=[]
    for id in I.keys():
        lable.append(I[id]["NAME"])
    
    if VIZ_INVEN_PIE:  # 평가 에피소드 전체의 평균 재고 수준을 시각화하여 각 아이템이 전체 재고에서 차지하는 비율을 파악 가능
        for x in range(N_EVAL_EPISODES):
            for y in range(len(I)):
                for z in range(SIM_TIME):
                    avg_inven[y][z] += inventory[x][y][z]  # 모든 평가 에피소드에 대해 반복, 각 아이템에 대해 반복, 시뮬레이션 시간 동안 각 시간 단계에 대해 바녹하여 각 에피소드의 재고 수준을 누적

        plt.pie([sum(avg_inven[x])/N_EVAL_EPISODES for x in range(len(I))],
                explode=[0.2 for _ in range(len(I))], labels=lable, autopct='%1.1f%%')
        plt.legend()
        plt.show() 

    if VIZ_INVEN_LINE:  # 특정 에피소드(best_index)에서 각 아이템의 재고 수준과 수요 및 주문량을 시각화하여 시간에 따른 재고의 변화를 선 그래프로 나타냄
        for id in I.keys():
            # Visualize the inventory levels of the best episode
            plt.plot(inventory[best_index][id],label=lable[id])
        plt.plot(demand_qty[-SIM_TIME:], "y--", label="Demand_QTY")  # 수요량을 점선으로 표시
        plt.plot(order_qty[-SIM_TIME:], "r--", label="ORDER")  # 주문량을 점선으로 표시
        plt.legend()
        plt.show()


def export_state(Record_Type):
    state_real = pd.DataFrame(STATE_ACTION_REPORT_REAL)
    state_corr = pd.DataFrame(STATE_ACTION_REPORT_CORRECTION)
    
    if Record_Type == 'TEST':
        state_corr.dropna(axis=0, inplace=True)
        state_real.dropna(axis=0, inplace=True)
        
    columns_list = []
    for id in I.keys():
        if I[id]["TYPE"]=='Material':
            columns_list.append(f"{I[id]['NAME']}.InvenLevel")
            if DAILY_CHANGE:
                columns_list.append(f"{I[id]['NAME']}.DailyChange")
            if INTRANSIT:
                columns_list.append(f"{I[id]['NAME']}.Intransit")
        else:
            columns_list.append(f"{I[id]['NAME']}.InvenLevel")
            if DAILY_CHANGE:
                columns_list.append(f"{I[id]['NAME']}.DailyChange")
    columns_list.append("Remaining_Demand")
    columns_list.append("Action")
    '''
    for keys in I:
        columns_list.append(f"{I[keys]['NAME']}'s inventory")
        columns_list.append(f"{I[keys]['NAME']}'s Change")
    
    columns_list.append("Remaining Demand")
    columns_list.append("Action")
    '''

    state_real.columns = columns_list
    state_corr.columns = columns_list
    
    state_real.to_csv(f'{STATE}/STATE_ACTION_REPORT_REAL_{Record_Type}.csv')
    state_corr.to_csv(
        f'{STATE}/STATE_ACTION_REPORT_CORRECTION_{Record_Type}.csv')
    

##### 이 그래프(summary plot)는 모델이 특정 action을 선택하는 과정에서 어떤 특징들이 중요한 역할을 했고, 각 특징들이 얼마나 큰 기여도를 가지고 있는지를 보여준다. #####
##### 그래프를 통해 어떤 특징들이 모델의 action선택에 있어 중요하게 작용했는지를 파악할 수 있고, 이를 기반으로 모델의 해석 및 성능 최적화를 위한 추가적인 조치 고려 가능 #####
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
import os
from call_shap import cal_shap  # Call cal_shap function

# auto, bar, violin, dot, layered_violin, heatmap, waterfall, image
SHAP_PLOT_TYPE = 'bar'
'''
'auto': Automatically selects the appropriate plot type based on the Shap value.
'bar': Creates a bar graph that displays the average Shap value for each feature and the absolute average effect of that feature value.
'violin': Visualize the distribution of Shap values for each feature as a violin plot.
'dot': Creates a dot plot that displays the distribution of Shap values for each feature as dots.
'layered_violin': Visualizes the violin plot divided into multiple layers.
'heatmap': Creates a heatmap of Shap values to visualize interactions between features.
'waterfall': Visualize the contribution of Shap values for each feature as a stacked bar graph.
'image': Supports visualization of Shap values for image data
'''


def read_path():
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    result_csv_folder = os.path.join(parent_dir, "result_CSV")
    STATE_folder = os.path.join(result_csv_folder, "state")
    # Data_place=os.path.join(STATE_folder,f"Train_{len(os.listdir(STATE_folder))}")
    return STATE_folder


def simplify_tree(tree, node_id=0):
    if tree.children_left[node_id] == tree.children_right[node_id] == -1:  # 현재 노드는 리프 노드
        return tree.value[node_id].argmax(), True, int(tree.value[node_id].sum())  # 리프 노드일 경우, 해당 노드의 클래스와 샘플 수를 반환하고, True를 반환하여 리프 노드임을 나타냄

    left_class, left_is_leaf, left_samples = simplify_tree(
        tree, tree.children_left[node_id])  # 리프 노드가 아닌 경우 왼쪽 자식과 오른쪽 자식에 대해 simplify_tree를 재귀적으로 호출함
    right_class, right_is_leaf, right_samples = simplify_tree(
        tree, tree.children_right[node_id])

    if left_is_leaf and right_is_leaf and left_class == right_class:  # 왼쪽과 오른쪽 자식 노드가 모두 리프 노드이며 반환하는 클래스가 동일한 경우, 두 자식 노드를 병합함

        tree.children_left[node_id] = -1
        tree.children_right[node_id] = -1  # 병합된 노드는 현재 노드가 리프 노드가 되도록 -1로 설정한다.

        return left_class, True, left_samples + right_samples  
    return tree.value[node_id].argmax(), False, int(tree.value[node_id].sum())  


# Read newest Test Dataset
file_path = os.path.join(
    f"{read_path()}", "STATE_ACTION_REPORT_REAL_TEST.csv")
print("Import a data file: ", file_path)
df = pd.read_csv(file_path)

# Extract features (X) and prediction values (y)
X = df.iloc[:, 1:-1]  # 데이터프레임의 첫 번째 열을 제외하고 마지막 열 전까지의 모든 열로 구성
y = df.iloc[:, -1:]  # 마지막 열로 구성

# Split the dataset into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=False)  # train set은 90%, test set은 10%로 분할하고, shuffle=False: 데이터 순서를 유지한 채 분할한다는 뜻


print(f"Number of samples in the train set: {len(X_train)}")
print(f"Number of samples in the test set: {len(X_test)}")

'''
# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
'''

# Decision tree learning
model = DecisionTreeClassifier(
    criterion='gini',          # 'gini' 또는 'entropy'
    max_depth=6,               # 트리의 최대 깊이
    min_samples_split=20,      # 노드를 분할하기 위한 최소 샘플 수
    min_samples_leaf=10        # 잎 노드가 가지고 있어야 할 최소 샘플 수
)
print('Start fit')
clf = model.fit(X_train, y_train)

# Predict on the testing set
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
 
acc_train = accuracy_score(y_train, y_train_pred)  # accuracy_score()함수를 이용하여 train 및 test set의 정확도를 계산하고 출력한다.
acc_test = accuracy_score(y_test, y_test_pred)

print('Training Accuracy: {:.3f}'.format(acc_train))
print('Testing Accuracy: {:.3f}'.format(acc_test))


# Visualize DOT format data to create graphs
# Generate data in DOT format to visualize decision trees
simplify_tree(clf.tree_)  # 트리 구조를 시각화하고 이해하기 쉽게 단순화함

# simplified_tree = simplify_tree(clf.tree_)
FEATURE_NAME = df.columns[1:-1]
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=FEATURE_NAME,
                           class_names=['[0]', '[1]',
                                        '[2]', '[3]', '[4]', '[5]'],
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)

# Save the decision tree graph as a PNG file
graph.render('DT',  format='png', view=False)
print(df.columns)

####### SHAP ########

# Extract Unique Actions of dataset
actions = df['Action'].unique()  # 데이터셋에서 action을 추출하고, cal_shap함수를 이용하여 shap값을 계산하여 모델의 예측을 해석함.dot


# Extract Unique Actions of test dataset
actions = df['Action'].unique()
# cal_shap(model,X of dataset, Plot type what you want, unique actions)
cal_shap(clf, X_train, SHAP_PLOT_TYPE, actions)
# model: Model of distilated policy
# X_test: Test dataset
# SHAP_PLOT_TYPE: Decision_PLOT_TYPE
# actions: actions of test_dataset

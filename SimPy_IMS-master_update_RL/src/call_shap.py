import shap
import matplotlib.pyplot as plt

# model: Model of distilated policy
# X_test: Test dataset
# SHAP_PLOT_TYPE: Decision_PLOT_TYPE
# actions: actions of test_dataset


def cal_shap(model, X, SHAP_PLOT_TYPE, actions):
    path = save_path()  # save_path() 함수는 SHAP 그래프 파일을 저장할 폴더의 경로를 반환하여 이 경로를 path 변수에 저장
    explainer = shap.TreeExplainer(model, X)  # shap.TreeExplainer는 모델에 대한 SHAP 값을 계산하는 객체를 생성

    # Cal Shap Values
    shap_values = explainer(X)  # explainer(X)는 입력 데이터 X에 대해 SHAP 값을 계산

    # Makes Shap plots
    for x in range(len(actions)):  # actions 리스트에 있는 각 행동에 대해 반복문을 실행
        # shap_values[:, :, x] means xth class's shap value
        shap.summary_plot(shap_values[:, :, x], X,
                          plot_type=SHAP_PLOT_TYPE, show=False)  # shap.summary_plot() 함수는 SHAP 값을 시각화해주는 함수 / show=False는 즉시 그래프를 화면에 보여주지 않고, 저장하기 위해 plt.savefig()를 사용할 수 있도록 함
        p = f"{path}/Action's{actions[x]}_shap.png"  # 그래프는 Action's 행동명_shap.png 형식으로 저장
        plt.savefig(p)
        print(p)
        plt.close()


def save_path():
    import os  # os: 파일 및 디렉토리 경로와 관련된 작업을 처리하는 모듈
    import shutil  # shutil: 파일과 디렉토리를 복사, 이동 또는 삭제하는 데 사용하는 모듈

    # Current working path
    current_path = os.path.dirname(__file__)  # __file__은 실행 중인 스크립트 파일의 경로를 나타내고, os.path.dirname()은 그 파일이 속한 디렉토리 경로를 반환

    # Parent folder path
    parent_path = os.path.dirname(current_path)  # current_path의 상위 디렉토리 경로를 얻음

    # New folder path
    summary_plot_path = os.path.join(parent_path, "summary_plot")  # 상위 디렉토리(parent_path)에 summary_plot이라는 새로운 폴더의 경로를 생성

    # Remove new folder if it already exists
    if os.path.exists(summary_plot_path):
        shutil.rmtree(summary_plot_path)  # summary_plot 폴더가 이미 존재하는 경우, shutil.rmtree() 함수를 사용하여 해당 폴더와 그 안의 모든 파일을 삭제

    # Create a new folder
    os.mkdir(summary_plot_path)  # os.mkdir() 함수를 사용하여 summary_plot_path에 해당하는 새로운 폴더를 생성

    return summary_plot_path  # 새롭게 생성된 summary_plot 폴더의 경로를 반환

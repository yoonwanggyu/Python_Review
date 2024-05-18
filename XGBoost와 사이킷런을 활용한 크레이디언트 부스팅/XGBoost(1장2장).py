# 1장) ----------------------------------##-----------------------------------

import pandas as pd
df_bikes = pd.read_csv("C:/Users/Administrator/Downloads/bike_rentals.csv")
df_bikes.head()
df_bikes.describe()     # 평균/중앙값을 비교하면 왜곡의 정도를 가늠 가능
df_bikes.info()
df_bikes.isna().sum().sum()

df_bikes[df_bikes.isna().any(axis=1)]
df_bikes['windspeed'].fillna((df_bikes['windspeed'].median()),inplace=True)        # inplace=True : 데이터프레임 자체를 수정
df_bikes.iloc[[56,81]]

df_bikes.groupby(['season']).median()
df_bikes['hum'] = df_bikes['hum'].fillna(df_bikes.groupby('season')['hum'].transform('median'))
df_bikes.iloc[[129,213,388]]

df_bikes[df_bikes['temp'].isna()]
mean_temp = (df_bikes.iloc[700]['temp'] + df_bikes.iloc[702]['temp'])/2
mean_atemp = (df_bikes.iloc[700]['atemp'] + df_bikes.iloc[702]['atemp'])/2
df_bikes['temp'].fillna((mean_temp),inplace=True)
df_bikes['atemp'].fillna((mean_atemp),inplace=True)
df_bikes.iloc[701]

df_bikes['dteday']
df_bikes['dteday'] = pd.to_datetime(df_bikes['dteday'])
import datetime as dt
df_bikes['mnth'] = df_bikes['dteday'].dt.month
df_bikes.tail()
df_bikes.loc[730,'yr'] = 1.0
df_bikes = df_bikes.drop('dteday',axis=1)
df_bikes.info()

        # <회귀 모델>---------------------------------------
    # casual + registered = cnt 임으로 삭제
df_bikes = df_bikes.drop(['casual','registered'],axis=1)
df_bikes.to_csv('bike_rentals_cleaned.csv',index=False)
x = df_bikes.iloc[:,:-1]
y = df_bikes.iloc[:,-1]
            # 1) 선형 회귀
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=2)
import warnings
warnings.filterwarnings('ignore')
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
import numpy as np
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("RMSE: %0.2f" % (rmse))
            # 2) XGBRegressor
from xgboost import XGBRegressor
xg_reg = XGBRegressor()
xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
print("RMSE: %0.2f" % (rmse))       # 더 낮음!!
            # 3) 교차검증(K-fold cross-validation) + 선형회귀
from sklearn.model_selection import cross_val_score
model = LinearRegression()
scores = cross_val_score(model,x,y,scoring='neg_mean_squared_error',cv=10)
rmse = np.sqrt(-scores)
print('회귀 rmse:',np.round(rmse,2))
print("RNSE 평균: %0.2f" %(rmse.mean()))
            # 4) 교차검증 + XGBRegressor
model = XGBRegressor()
scores = cross_val_score(model,x,y,scoring='neg_mean_squared_error',cv=10)
rmse = np.sqrt(-scores)
print("회귀 rmse:",np.round(rmse,2))
print('RMSE 평균: %0.2f' % (rmse.mean()))


        # <분류 모델>--------------------------------------------------
import pandas as pd
df_census = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',header=None)   # 첫번 째 행이 열 이름으로 들어가 있음
df_census.head()
df_census.info()
df_census.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss',
                     'hours-per-week','native-country','income']
df_census.head()
df_census = df_census.drop(['education'],axis=1)
df_census = pd.get_dummies(df_census)   # get_dummies = 수치열이 아닌 열을 수치형으로 교체
df_census.info()
df_census.iloc[:,6:] = df_census.iloc[:,6:].astype(int)
        # 타켓 = income <= 50 / income > 50
df_census = df_census.drop('income_ <=50K',axis=1)
X = df_census.iloc[:,:-1]
y = df_census.iloc[:,-1]
        # 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
def cross_val(classifier,num_splits=10):
    model = classifier
    scores = cross_val_score(model,X,y,cv=num_splits)
    print("정확도 : ", np.round(scores,2))
    print("평균 정확도 : %0.2f" %(scores.mean()))

cross_val(LogisticRegression())
        # XGBClassifier
from xgboost import XGBClassifier
cross_val(XGBClassifier(n_estimators=5))


# 2장) 결정트리----------------------------------##--------------------------------------------------------------------------
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
X = df_census.iloc[:,:-1]
y = df_census.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(random_state=2)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_pred,y_test)
model.score(X_test,y_test)
            # 시각화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(13,8))
plot_tree(model,max_depth=2,feature_names=list(X.columns),class_names=['0','1'],filled=True,rounded=True,fontsize=10)
plt.show()
        # 하이퍼파라미터 조정
df_bikes = pd.read_csv("C:/Users/Administrator/bike_rentals_cleaned.csv")
X_bikes = df_bikes.iloc[:,:-1]
y_bikes = df_bikes.iloc[:,-1]
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score
model = DecisionTreeRegressor()
scores = cross_val_score(model,X_bikes,y_bikes,scoring='neg_mean_squared_error',cv=5)
rmse = np.sqrt(-scores)
rmse.mean()             # 과적합됨

X_train,X_test,y_train,y_test = train_test_split(X_bikes,y_bikes,test_size=0.2,random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_train)
model_mse = mean_squared_error(y_train,y_pred)
model_rmse = np.sqrt(model_mse)
model_rmse      # 훈련 세트의 성능을 확인한 결과 0.0 결과가 나옴 -> 과대적합!!
        # 모델의 노드 수 확인
X_train.shape
leaf_node_count = 0
tree = model.tree_
for i in range(tree.node_count):
    if (tree.children_left[i] == -1) and (tree.children_right[i] == -1):
        leaf_node_count += 1
        if tree.n_node_samples[i] > 1:
            print("노드 인덱스: ",i, '샘플 개수: ',tree.n_node_samples[i])
print('전체 리프 노드 개수: ', leaf_node_count)         # 훈련 세트 크기가 584 / 전체 리프 노드가 585 -> 차이 없음
        # GridSearchCV
from sklearn.model_selection import GridSearchCV
model = DecisionTreeRegressor(random_state=2)
params = {'max_depth' : [None,2,3,4,6,8,10,20]}
grid = GridSearchCV(model,
                    param_grid=params,
                    scoring='neg_mean_squared_error',
                    cv=5,
                    return_train_score=True,
                    n_jobs=1)
grid.fit(X_train,y_train)
best_params = grid.best_params_
print("최상의 매개변수: ",best_params)
best_score = np.sqrt(-grid.best_score_)
print("훈련 점수: {:.3f}".format(best_score))

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = mean_squared_error(y_test,y_pred)**0.5
print("테스트 점수: {:.3f}".format(rmse_test))

        # GridSearchCV 함수
df_bikes = pd.read_csv("C:/Users/Administrator/bike_rentals_cleaned.csv")
X_bike = df_bikes.iloc[:,:-1]
y_bike = df_bikes.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X_bike,y_bike,random_state=42,test_size=0.3)
def grid_search(params, model=DecisionTreeRegressor(random_state=2)):
    grid_reg = GridSearchCV(model, params,scoring='neg_mean_squared_error',cv=5,n_jobs=1)
    grid_reg.fit(X_train,y_train)
    best_params = grid_reg.best_params_
    print("최상의 매개변수: ",best_params)
    best_score = np.sqrt(-grid_reg.best_score_)     # 교차 검증에서 계산된 검증 폴드의 평균 점수가 저장되어 있음
    print("훈련 점수: {:.3f}".format(best_score))
    best_model = grid_reg.best_estimator_           # best_estimator_를 쓰나 그냥 grid_reg.predict하나 똑같음 = 이미 grid_reg에 베스트 모델이 할당되어 있음
    y_pred = best_model.predict(X_test)
    rmse_test = mean_squared_error(y_test,y_pred)**0.5
    print("테스트 점수:{:.3f}".format(rmse_test))
X_train.shape
grid_search(params={"min_samples_leaf":[1,2,4,6,8,10,20,30]})
grid_search(params={'max_depth':[None,2,3,4,6,8,10,20],
                    'min_samples_leaf':[1,2,4,6,8,10,20,30]})
grid_search(params={'max_depth':[6,7,8,9],
                    'min_samples_leaf':[3,5,7,9]})


        # 실습) 심장 질환 예측하기
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

df_heart = pd.read_csv("C:/Users/Administrator/Downloads/heart_disease.csv")
df_heart.head()
df_heart.info()
df_heart.isnull().sum()

X = df_heart.iloc[:,:-1]
y = df_heart.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                # 기본 모델(튜닝 전)
model = DecisionTreeClassifier(random_state=2)
scores = cross_val_score(model,X,y,cv=5)
print("정확도:",np.round(scores,2))
print("정확도 평균: %0.2f" % (scores.mean()))
                # RandomizedSearchCV 하이퍼파라미터 탐색
from sklearn.model_selection import RandomizedSearchCV
def randomized_Search_cv(params,runs=20,model=DecisionTreeClassifier(random_state=2)):
    rand_clf = RandomizedSearchCV(model,params,n_iter=runs,cv=5,n_jobs=-1,scoring='accuracy')
    rand_clf.fit(X_train,y_train)
    best_model = rand_clf.best_estimator_
    best_score = rand_clf.best_score_
    print("훈련 점수 : {:.3f}".format(best_score))
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_pred,y_test)
    print("테스트 점수 : {:.3f}".format(accuracy))
    return best_model

randomized_Search_cv(params={'max_depth':[None,2,3,4,6,8],
                             'max_features':['sqrt',0.95,0.90,0.85,0.80,0.75,0.7],
                             'min_samples_leaf':[1,0.01,0.02,0.03,0.04],
                             'max_leaf_nodes':[10,15,20,25,30,35,40,None],
                             'min_impurity_decrease':[0.0,0.0005,0.005,0.05,0.10,0.15,0.20],
                             'min_samples_split':[2,3,4,6,8,10]})
                # 베스트 모델(튜닝 후)
best_model = randomized_Search_cv(params={'max_depth':[None,6,7],
                                        'max_features':['sqrt',0.95,0.85,0.78],
                                        'max_leaf_nodes':[45,None],
                                        'min_impurity_decrease':[0.005,0.05],
                                        'min_samples_leaf':[1,0.035,0.04,0.045,0.05],
                                        'min_samples_split':[2,9,10],
                                        'min_weight_fraction_leaf':[0.0,0.05,0.06,0.07]},runs=100)
scores = cross_val_score(best_model,X,y,cv=5)
print("정확도:",np.round(scores,2))
print("정확도 평균: %0.2f"%(scores.mean()))
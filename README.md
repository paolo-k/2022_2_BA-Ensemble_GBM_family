# 2022_2_BA-Ensemble_GBM_family
Tutorial Homework 3(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)


# Overview of Ensemble
## What is "Ensemble"
Ensemble의 사전적 정의는 합주단, 총체, 맞춰 입게 지은 옷 등이 있다.(출처 : 네이버 사전) 모든 의미는 개별 개체가 아닌 여러 주체가 모여서 하나의 집합을 이룬다는 점이 공통점이 있다. Ensemble의 직관적 예시로, 흔한 락밴드구성(일렉, 베이스, 드럼)을 예로 들겠다. 일렉의 경우 고음역대에서 가장 튀지만, 저음역대가 약하다. 베이스는 저음역대를 채우고 화음과 박자를 이어주므로 없으면 정말 허전하지만, 있으면 존재감이 부족하다. 드럼은 박자는 확실하게 제공하지만 음역을 제공할 수 없다. 락밴드구성은 나사 하나씩 빠진 친구들이 합주를 통해 상호 보완하도록 만든다. 즉, Ensemble이란 머신러닝 모델들이 각자의 강점을 통해 다른 모델의 약점을 보완하면서 학습해나가는 총체적 과정이라고 볼 수 있다.

![앙상블 예시](https://user-images.githubusercontent.com/106015570/204057689-bcd2d7c1-950d-40de-a79b-2b102b304c3f.jpg)

![앙상블 사례](https://user-images.githubusercontent.com/106015570/204058357-1a0a0345-5363-4b93-ab19-b9bd886c35e2.png)

## Purpose of Ensemble

Ensemble의 키워드는 다양성(diversity)와 결합(combine)이다. 다양한 머신러닝 모델을 학습하고 그 결과물을 결합하여 오류를 감소시키는 것이 Ensemble의 가장 큰 목적이라고 할 수 있다. Ensemble은 크게 분산의 감소를 통한 오류 감소를 추구하는 Bagging과 편향의 감소를 통한 오류 감소를 추구하는 Boosting의 두 가지 방법이 있다. 본 튜토리얼을 두 가지 방법 중 Boosting의 일종인 Gradient Boost Machine(GBM)과, 그 효율성을 제고하는 Extreme Gradient Boost(XGB), Light GBM(LGBM)을 구현해보고 그 성능을 평가할 것이다.

# What is GBM, XGB, LGBM
## Gradient Descent + Boost
GBM은 Boost에 속하는 방법이다. 따라서 편향의 감소를 통해 오류를 줄이고자 한다. 편향을 감소시키는 방법으로 Gradient descent를 이용하기 때문에 GBM이라고 한다. GBM은 각각의 단계에서 base learner를 손실함수의 Gradient를 최소화하도록 새로 학습함으로써, 전 단계 base learner를 업그레이드한다. 편향을 줄인다는 점에서, Overfit이 발생할 가능성이 상당히 높은데, 이를 방지하기 위한 장치로 학습 데이터의 일부분만 비복원추출하여 사용하는 subsampling과 다음 모델 결합의 가중치를 줄이는 Shrinkage가 있다.

## Extremely efficient GBM
XGB는 Gradient Boost의 일종이다. 전체 가능한 분기점을 모두 확인하는 basic exact greedy algorithm의 단점인 메모리 효율성 및 병렬 연산 불가를 해결함으로써 Split point 찾기의 효율성을 제고한 방법으로, 데이터 전체를 분할하여 최적의 Split을 찾아낸다.

## Light GBM
LGBM 또한 효율성을 제고하는 Gradient Boost라는 점에서는 XGB와 비슷하다. 그러나 실제 방법론은 전혀 다르다. 전체 데이터를 분리함으로써 효율적으로 분기점을 찾아내는 XGB와 다르게, LGBM은 Gradient가 높은 instance의 sampling(Gradient-based One-Side Sampling), 그리고 conflict가 적은 feature들의 bundling(Exclusive Feature Bundling)을 통해 information loss를 최소화한 데이터셋 축소로 효율성을 제고한다.


# Tutorial of Ensemble
## 코드 및 데이터 개요
본 tutorial에 사용된 패키지는 아래와 같다.

---

import pandas as pd
import numpy as np
import timeit

from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from datetime import datetime

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

---

사용한 데이터는 이곳저곳에서 예시로 많이 사용되는 Iris dataset이다. 전체 instance은 150개이고, 3개 class(y) 각각의 instance 개수는 50개 이다. feature(x)는 4개의 연속변수로 구성되어 있다.

![iris](https://user-images.githubusercontent.com/106015570/204062293-f18defc5-21ad-46a9-a3f8-8698c204c0e8.PNG)

차원을 2차원으로 축소하여 데이터 분포를 시각적으로 확인하면 아래와 같다.

![dataset_distribution](https://user-images.githubusercontent.com/106015570/204062317-5a82b06b-1b94-41b1-9db3-3a5641ebe193.png)

아래 각각의 방법론 코드는 fitting - hyperparameter tuning(by grid search) - testing - evaluation의 4단계로 구성되어있으며, 단계별로 소요시간을 측정하였다.

## GBM
GBM의 코드는 아래와 같다.

---

start_time = datetime.now().replace(microsecond=0)


gbm = GradientBoostingClassifier()

gbm.fit(X_tr, y_tr)
gbm_params = {
    'n_estimators' : [100, 200, 300 ,400, 500], 
    'learning_rate' : [0.01, 0.05, 0.1, 0.15, 0.2], 
    'max_depth' : [1, 2, 3, 4, 5]
}

gbm_search = GridSearchCV(gbm, param_grid = gbm_params, scoring="accuracy", n_jobs= -1, verbose = 1)
gbm_search.fit(X_val, y_val)
print("hyperparameter : ", gbm_search.best_params_)

gbm_tune = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.01, max_depth = 1, random_state = 333)
gbm_tune.fit(X_tr, y_tr)

gbm_pred = gbm_tune.predict(X_te)
print("---Confusion Matrix---")
print(confusion_matrix(y_te, gbm_pred))
print(" ")
print("---Index for Evaluation---")
print(classification_report(y_te, gbm_pred))

end_time = datetime.now().replace(microsecond=0)
print(f"processing time {end_time - start_time}")

---

결과는 아래와 같다.

1) Evaluation
- confusion matrix

|  |real 0|real 1|real 2|
|------|---|---|---|
|pred 0| 10| 0| 0|
|pred 1| 0| 10| 0|
|pred 2| | 1| 9|

- indicators of perfomance

|class|precision|recall|F1|
|------|---|---|---|
|0|1.00|1.00|1.00|
|1|0.91|1.00|0.95|
|2|1.00|0.90|0.95|

accuracy : 0.97

2) 소요시간
- 0:00:31(31 sec)

## XGB
XGB의 코드는 아래와 같다.

---

start_time = datetime.now().replace(microsecond=0)


xgb = XGBClassifier()

xgb.fit(X_tr, y_tr)
xgb_params = {
    'n_estimators' : [100, 200, 300 ,400, 500], 
    'learning_rate' : [0.01, 0.05, 0.1, 0.15, 0.2], 
    'max_depth' : [1, 2, 3, 4, 5]
}

xgb_search = GridSearchCV(xgb, param_grid = xgb_params, scoring="accuracy", n_jobs= -1, verbose = 1)
xgb_search.fit(X_val, y_val)
print("hyperparameter : ", xgb_search.best_params_)

xgb_tune = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.01, max_depth = 1, random_state = 333)
xgb_tune.fit(X_tr, y_tr)

xgb_pred = xgb_tune.predict(X_te)
print("---Confusion Matrix---")
print(confusion_matrix(y_te, xgb_pred))
print(" ")
print("---Index for Evaluation---")
print(classification_report(y_te, xgb_pred))

end_time = datetime.now().replace(microsecond=0)
print(f"processing time {end_time - start_time}")

---

결과는 아래와 같다.

1) Evaluation
- confusion matrix

|  |real 0|real 1|real 2|
|------|---|---|---|
|pred 0| 10| 0| 0|
|pred 1| 0| 10| 0|
|pred 2| | 1| 9|

- indicators of perfomance

|class|precision|recall|F1|
|------|---|---|---|
|0|1.00|1.00|1.00|
|1|0.91|1.00|0.95|
|2|1.00|0.90|0.95|

accuracy : 0.97

2) 소요시간
- 0:00:06(6 sec)

## LGBM
LGBM의 코드는 아래와 같다.

---

start_time = datetime.now().replace(microsecond=0)


lgb = LGBMClassifier()

lgb.fit(X_tr, y_tr)
lgb_params = {
    'n_estimators' : [100, 200, 300 ,400, 500], 
    'learning_rate' : [0.01, 0.05, 0.1, 0.15, 0.2], 
    'max_depth' : [1, 2, 3, 4, 5]
}

lgb_search = GridSearchCV(lgb, param_grid = lgb_params, scoring="accuracy", n_jobs= -1, verbose = 1)
lgb_search.fit(X_val, y_val)
print("hyperparameter : ", lgb_search.best_params_)

lgb_tune = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.01, max_depth = 1, random_state = 333)
lgb_tune.fit(X_tr, y_tr)

lgb_pred = lgb_tune.predict(X_te)
print("---Confusion Matrix---")
print(confusion_matrix(y_te, lgb_pred))
print(" ")
print("---Index for Evaluation---")
print(classification_report(y_te, lgb_pred))

end_time = datetime.now().replace(microsecond=0)
print(f"processing time {end_time - start_time}")

---

결과는 아래와 같다.

1) Evaluation
- confusion matrix

|  |real 0|real 1|real 2|
|------|---|---|---|
|pred 0| 10| 0| 0|
|pred 1| 0| 10| 0|
|pred 2| | 1| 9|

- indicators of perfomance

|class|precision|recall|F1|
|------|---|---|---|
|0|1.00|1.00|1.00|
|1|0.91|1.00|0.95|
|2|1.00|0.90|0.95|

accuracy : 0.97

2) 소요시간
- 0:00:02(2 sec)  


# Conclusion
3개 모형 모두 분류 성능은 동일하게 나왔다. 반면, 연산시간 측면에서는 결과가 다르게 나타났다. GBM의 경우 31초가 소모된 반면, XGB는 6초, LGBM은 2초가 소모되었다.

모든 모형은 overfit 수준으로 매우 높은 예측력을 보였는데, 편차를 줄인다는 Boosting Idea 특성상 overfit의 위험성이 높은데다가, 분류가 쉬운 Iris dataset 특성으로 이와 같은 결과가 나타난 것으로 보인다. 연산시간 측면에서는 예상대로 GBM에 비해 XGB, LGBM이 훨씬 더 좋은 성능을 보였다. 일반 GBM 대비 XGB은 약 16.1%, LGBM은 6.4%의 시간이면 동일한 성능을 도출하기에 충분했다.

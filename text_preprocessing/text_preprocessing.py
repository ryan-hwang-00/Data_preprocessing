import pandas as pd
import numpy as np

# csv 불러오기
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submission = pd.read_csv('./sampleSubmission.csv')



# .을 기준으로 텍스트 데이터를 파싱한다. 예)'Braund, Mr. Owen Harris'
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)


#나이('Age') 필드를 그룹핑하여 'AgeGroup'필드 생성하여 할당하기
bins = [0 , 18, 25, 35, 60, 100]
group_names = ['Baby', 'Youth', 'YoungAdult', 'MiddleAged', 'Senior']
data['AgeGroup'] = pd.cut(data['Age'], bins, labels=group_names)
data['AgeGroup']

# 컬럼에 대한 드롭
data.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'Cabin', 'AgeGroup', 'Emabarked'], axis=1, inplace=True)


# 텍스트 데이터 숫자 변환 --의미-- 3개의 컬럼의 텍스트 데이터를 카테고리로 묶어서 하나의 컬럼에 넘버링 해줌
# 타이타닉 데이터에서는 6개의 카테고리로 의미를 나눌 수 있음

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

for col in ['Sex', 'Embarked', 'Title']:
    data[col] = label.fit_transform(data[col])


# 하나의 데이터를 트레이닝과 테스트 데이터로 나눈다. #성능 체크에 필요
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(data, y, test_size=0.2, random_state = 5, stratify = y)

# 텍스트 날짜 데이터를 날짜형식 데이터로 바꿔주기
train['Dates'] = pd.to_datetime(train['Dates'], format='%Y-%m-%d %H:%M:%S', errors='raise')
#train['Dates'] = train['Dtate'].astype('datetime64')


#날짜 데이터 속성 별로 넣어주기
train['year'] = train['Dates'].dt.year
train['month'] = train['Dates'].dt.month
train['day'] = train['Dates'].dt.day
train['dayofweek'] = train['Dates'].dt.dayofweek
train['hour'] = train['Dates'].dt.hour
train['minute'] = train['Dates'].dt.minute

#람다로 처음 있었던 날로부터 몇일째인지 계산해보기
train['n_days'] = (train['Dates'].dt.date - train['Dates'].dt.date.min()).apply(lambda x: x.days)
test['n_days'] = (test['Dates'].dt.date - test['Dates'].dt.date.min()).apply(lambda x: x.days)


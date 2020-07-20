import numpy as np
import pandas as pd

# 특정 경로의 파일 데이터를 컬럼으로 추가하기 (람다)
train['imgpath'] = train['id'].apply(lambda x: '../input/dog-breed-identification/train/'+ x +'.jpg')

# 선생님 버전
# filePath = '../input/dog-breed-identification/train/'
# f = lambda x: filePath + x + '.jpg'
# label['imgpath'] = label['id'].apply(f)

from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator

# 이미지를 불러와서 imp 테이블에 넣어준다.(람다)
train['img'] = train['imgpath'].apply(lambda x: load_img( x, target_size = (10, 10)))
# 이미지를 다시 array 형태로 바꿔준다. (람다)
train['imgArray'] = train['img'].apply(lambda x: img_to_array(x))

# 선생님버전
# ff = lambda x: img_to_array(load_img(x , target_size = (10, 10)))
# label['imgArray'] = label['imgpath'].apply(ff)
# label['imgArray'][0].shape

# X: 이미지 데이터(array) 정규화하여 할당
X_train= label['imgArray'].values / 255

# Y :정답레이블(breed)을 원핫인코딩 변환
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
a = le.fit_transform(label['breed'])
print(a)
Y_train = np_utils.to_categorical(a)
Y_train[0]
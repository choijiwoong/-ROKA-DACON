#[1단계: 기본적인 데이터의 특성 파악]
import pandas as pd
import tensorflow as tf

train_data=pd.read_csv("movies_train.csv")
test_data=pd.read_csv("movies_test.csv")

print("train의 상위 5개 데이터 샘플 출력: ", train_data.head())
print("test의 상위 5개 데이터 샘플 출력: ", test_data.head(), '\n')
print("train데이터의 information: ", train_data.info())
print("test데이터의 information: ", test_data.info(), '\n')

""""""
#[2-2단계: 공통vocab생성]
vocab_base=[]
vocab_base.extend([train_data['title'], train_data['distributor'], train_data['genre'], train_data['screening_rat'], train_data['director']])
vocab_base=pd.Series(vocab_base).explode()#flatten.
print(vocab_base.shape)#3000

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer=Tokenizer()
tokenizer.fit_on_texts(vocab_base)

word_to_index=tokenizer.word_index#vocab
vocab_size=len(word_to_index)+1
print(vocab_size)#1676

#[3단계: 모델의 입력값을 위한 전처리]
train_data['title']=tokenizer.texts_to_sequences(train_data['title'])#word embedding을 통한 score처리
print(train_data['title'].head())

train_data['genre']=tokenizer.texts_to_sequences(train_data['genre'])
print(train_data['genre'].head())

def parse_str_time(str_data):#2012-11-22 to 734722
    year=int(str_data[:4])
    month=int(str_data[5:7])
    day=int(str_data[8:])

    day_month=[31,28,31,39,31,30,31,31,30,31,30,31]#윤년아닐때

    def is_leap(year):
        return ((year%4==0) and (year%100!=0)) or (year%400==0)
    def days_month(month):
        if(is_leap(year)):
            day_month[1]=29
            return day_month[month-1]
        else:
            return day_month[month-1]

    return year*365 + month*days_month(month) + day
train_data['release_time']=train_data['release_time'].apply(parse_str_time)
print(train_data['release_time'].head())

print(train_data['time'].head())

train_data['screening_rat']=tokenizer.texts_to_sequences(train_data['screening_rat'])
print(train_data['screening_rat'].head())

train_data['director']=tokenizer.texts_to_sequences(train_data['director'])
print(train_data['director'].head())

print(train_data['dir_prev_bfnum'])

print(train_data['dir_prev_num'])

print(train_data['num_staff'])

print(train_data['num_actor'])

#[4-2단계: 모델 설계]_title, distributor, director, genre제외
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model=Sequential()
model.add(Dense(32, input_shape=(600,1), activation='relu'))#release_time, time, dir_prev_bfnum, dir_prev_num, num_staff, num_actor만을 사용예정
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.fit(train_data['release_time'], train_data['box_off_num'], epochs=500, verbose=2)#, train_data['time'], train_data['dir_prev_bfnum'], train_data['dir_prev_num'], train_data['num_staff'], train_data['num_actor']

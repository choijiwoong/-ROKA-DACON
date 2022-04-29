"""
초기 계획을 잡아보자.
데이터는 총 12개로
title(영화의 제목), distributor(배급사), genre(장르), release_time(개봉일), time(상영시간), screening_rat(상영등급), director(감독이름)
dir_prev_bfnum(감독이 만든 영화들의 평균 관객수), dir_prev_num(감독이 만든 영화개수), num_staff(스텝수), num_actor(주연배우수), box_off_num(관객수)
이며 관객수에 해당하는 box_off_num은 우리가 예측해야하는 데이터에 해당한다.
 time과 dif_prev_num, num_staff, num_actor, box_off_num은 int64타입, dir_prev_bfnum은 float54타입이며 나머지는 string object형식이다.

train데이터는 총 600개의 샘플로 이루어져 있으며 결측값은 없다. test데이터는 총 243개의 샘플로 이루어지며 결측값은 없다.

예측에 사용할 데이터는 box_off_num을 제외한 데이터값으로, 각 항목을 model의 인풋으로 어떻게 반영할지를 생각해보자.
 title의 경우 영화의 흥행에 미칠 영향이 있을까 없을까... title의 경우 Hmm..NLP로 크롤링해서 유사 제목의 흥행도를 반영하는 것 외에는 딱히 없을 것 같아서
word_embedding을 통해 단어 자체를 score에 반영시키는 것도 나쁘지 않을 것 같다. 굳이 새로운 정보를 찾는 것이 아닌, 제목 자체와 흥행성의 상관관계를 학습시킨다랄까..
근데 문장 자체를 임베딩하는 것이 단어 임베딩 보다 좋을 듯
 distributor는 임베딩...보다는 그냥 자체적으로 vocab같은 걸 생성해서(train데이터) 특정 배급사와의 흥행성을 추론하는게 좋을듯
 genre역시 vocab을 만들면 될듯
 release_time은...일단 날짜의 범위가 1년으로 제한되어 있으니 일자 기준으로 1~365처럼 날짜를 스코어로 변환해서 입력하면 좋을듯
 time의 경우는 상영시간이니까 그대로 score에 넣으면 될듯. int기도하고
 screening_rat은 몇세관람가 이런건데 이것도 vocab 정확히는 integer encoding해서 위와 마찬가지로 넣으면 될듯
 director도 vocab으로 만들자 웬만한 string object는 한정적이라 vocab으로 만들면 이상없을듯. 예외 데이터도 OOB같은거로 하면 되니
 dir_prev_bfnum은 처음에 감독이랑 연관시키려했는데 별개로 해도 상관은 없을듯. 관객수니까 그대로 input
 dir_prev_num도 마찬가지로 그대로 input
 num_staff 그대로
 num_actor그대로.
최종적인 예측값(출력)인 관객수는 int니까 모델의 출력은 int가 되게끔!
"""
#[1단계: 기본적인 데이터의 특성 파악]
import pandas as pd
import tensorflow as tf

train=pd.read_csv("movies_train.csv")
test=pd.read_csv("movies_test.csv")

print("train의 상위 5개 데이터 샘플 출력: ", train.head())
print("test의 상위 5개 데이터 샘플 출력: ", test.head(), '\n')
print("train데이터의 information: ", train.info())
print("test데이터의 information: ", test.info(), '\n')

"""[2-1단계: title을 word embedding하여 input으로 사용할건데 자기가 만든 코드만 사용하랬으니...CNN에서 feature 얻는 기법응용.. auto encoding]
#우선 train의 title데이터의 길이를 파악, 패딩 후에 해당 사이즈를 input으로 하는 linear 모델을 통해 축소된 임베딩 벡터를 얻고, 다시 원래 사이즈로 복구시킨 뒤에 기존의 값과의 손실을 계산
#선인코딩 후 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

train_title=train['title']
print("title_train상위 5개 데이터: ", train_title[:5])

title_tokenizer=Tokenizer()
title_tokenizer.fit_on_texts(train_title)
title_train_encoded=title_tokenizer.texts_to_sequences(train_title)#title train을 인코딩
print("\n인코딩된 train's title상위 5개 데이터: ", title_train_encoded[:5])

title_word_to_index=title_tokenizer.word_index
title_vocab_size=len(title_word_to_index)+1

plt.hist([len(each_title) for each_title in title_train_encoded], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of sampled')
plt.show()#대부분의 데이터가 15안쪽에 있고, 15이상의 데이터는 대략 50개 정도 되어보임. 20을 기준으로 보면 20보다 넘는것은 10개 미만으로 보이니 20으로 패딩예정.

max_len_title_encoded=6

title_train_padded=pad_sequences(title_train_encoded, maxlen=max_len_title_encoded)
print("title_train의 shape: ", title_train_padded.shape)#600, 20
#2-1단계를 통해 title_tokenizer와 ...나머지를 얻었는데 지금 생각해보니 전반적으로 사용할 전용 vocab을 만들어 모든 element에 공통적인 score에 사용하는게 좋을듯.
#따로 title별, distributor별로 vocab을 만드는 것은 의미가 없어보임. 다만 배급사와 같은 경우 기본적인 전처리를 통해 내용이 삭제될 수 있으므로, customizing을 통해 배급사만 전처리 하지 않게끔
하는것이 중요할 듯"""
#[2-2단계: 공통vocab생성]_2-1단계에서 느낀 피드백을 바탕으로 train_data에 사용되는 string object를 처리할 공통적인 vocab을 생성해보자
#영화의 제목, 배급사, 장르, 상영등급, 감독이 해당된다. 이때 감독과 배급사는 고유명사로서 dict에 직접 추가할 예정이다.


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

auto_encoding=Sequential()
#model.add

"""1차시도, 간단한 linear모델을 위한 int64와 float64항목만을 사용
x_train=x_train.drop('title', 'distributor', 'genre', 'release_time', 'screening_rat', 'director')#6개 항목 제거, 4개항목

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

model=Sequential()
model.add(Dense(32, input_shape=(4,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='binary_crossentropy', optimizer='SGD', matrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, verbose=2)
"""

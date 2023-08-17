import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D,BatchNormalization
from keras import layers
from keras.optimizers import Adam,RMSprop,legacy
from keras.optimizers.legacy import SGD

input_shape = (227, 227, 3)
model = Sequential()
#CONV1
model.add(Conv2D(96, (11, 11), strides=4,padding='valid', input_shape=input_shape))
#MAX POOL1
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
#NORM1  Local response normalization 
model.add(BatchNormalization())
#CONV2
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
#MAX POOL2
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
#NORM2
model.add(BatchNormalization())
#CONV3
model.add(Conv2D(384, (3, 3),strides=1, activation='relu', padding='same'))
#CONV4
model.add(Conv2D(384, (3, 3),strides=1, activation='relu', padding='same'))
#CONV5
model.add(Conv2D(256, (3, 3),strides=1, activation='relu', padding='same'))
#MAX POOL3
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
model.add(Flatten())
#FC6 예측 class가 적어 FC layer을 조정했습니다.
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
#FC7
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
#FC8 이진 분류이기 때문에 sigmoid
model.add(Dense(1, activation='sigmoid'))
# SGD Momentum 0.9, L2 weight decay 5e-4
optimizer =  SGD(learning_rate=0.01, decay=5e-4, momentum=0.9)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
model.summary()
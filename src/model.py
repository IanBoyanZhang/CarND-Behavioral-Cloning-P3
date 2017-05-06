from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Lambda, MaxPooling2D
from keras.layers.convolutional import Conv2D

from keras.optimizers import Adam
import keras.backend as K

from config import W, H

model = Sequential()
# Meah shift
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(H, W, 1)))
model.add(Conv2D(2, 3, 3, border_mode='valid', input_shape=(H, W, 1), activation='relu'))
# Debug intermediate layer
ConvOut1 = MaxPooling2D((4, 4), (4, 4), 'valid')
# model.add(MaxPooling2D((4, 4), (4, 4), 'valid'))
model.add(ConvOut1)
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer=Adam(lr=1e-3), loss='mse')

print('Main model')
model.summary()

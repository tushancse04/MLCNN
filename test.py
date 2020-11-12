from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine.input_layer import InputLayer

m = 3
img = [[[1,2,3],[1,2,3],[1,2,3]]]
iimg = np.array(img)
oimg = np.array(img)
iimg = iimg.reshape((len(iimg),m,m,1))
oimg = oimg.reshape((len(oimg),m,m,1))

y = np.array([[1]])
CVN = 2
model = Sequential()
input_shape = (m,m,1)
model.add(Conv2D(m, kernel_size=CVN, activation='relu', input_shape=(m,m,1),padding='same'))
model.add(Conv2D(m, kernel_size=CVN, activation='relu',padding='same'))
model.add(Conv2D(1, kernel_size=CVN, activation='relu',padding='same'))

model.compile(optimizer='adam', loss='mse')
model.fit(iimg, oimg, epochs=2)

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions

# Testing

layer_outs = [func([iimg]) for func in functors]
ll = layer_outs[2][0][0]

for i in range(len(ll)):
	tot = 0
	for j in range(len(ll[i])):
		tot += ll[i][j][0]
	ll[i] = tot
print(ll)

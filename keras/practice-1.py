import keras
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

num_class = 10
x_train = np.reshape(x_train, newshape=(np.shape(x_train)[0], np.shape(x_train)[1], np.shape(x_train)[2], 1))
x_test = np.reshape(x_test, newshape=(np.shape(x_test)[0], np.shape(x_test)[1], np.shape(x_test)[2], 1))
input_share = (np.shape(x_train)[1], np.shape(x_train)[2], 1)
print(input_share)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(np.shape(y_test))



model = keras.models.Sequential()
model.add(keras.layers.Conv2D(
    32,
    activation="relu",
    input_shape=input_share,
    kernel_size=3
))
model.add(keras.layers.Conv2D(
    64,
    activation="relu",
    kernel_size=3
))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_class, activation="softmax"))
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
# 令人兴奋的训练过程
model.fit(x_train, y_train, batch_size=64, epochs=6,
          verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
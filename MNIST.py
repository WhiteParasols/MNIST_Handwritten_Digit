from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Import MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
train_size = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
print('x_train shape:{0}, y_train shape:{1}'.format(x_train.shape, y_train.shape))
print('x_test shape:{0}, y_test shape:{1}'.format(x_test.shape, y_test.shape))

# 2. Data pre-processing
x_train = x_train.reshape(56000, 784) # 28x28 -> 1x784
x_test = x_test.reshape(14000, 784)
x_train = x_train.astype('float32') # integer type -> float type to divide by 255
x_test = x_test.astype('float32')
x_train /= 255 # black = 0, white = 255 -> 0-1 scale
x_test /= 255
print('x_train matrix shape:{0}, x_test matrix shape:{1}'.format(x_train.shape, x_test.shape))

# 3. Model Construction
model = Sequential() # sequential modeling
model.add(Dense(512, input_shape=(784,))) # Dense - fully connected layer
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# 4. Model Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Model training
hist = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test,y_test), verbose=1)

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()
plt.xticks(range(5), range(1,6))
loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val loss')
acc_ax.plot(hist.history['accuracy'],'b',label='train acc')
acc_ax.plot(hist.history['val_accuracy'],'g',label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()

# 6. Accuracy assessment
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:{0}%'.format(test_acc*100))

predicted_classes = np.argmax(model.predict(x_test), axis=1)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    correct = correct_indices[i]
    plt.imshow(x_test[correct].reshape(28, 28), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
plt.tight_layout()

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    incorrect = incorrect_indices[i]
    plt.imshow(x_test[incorrect].reshape(28, 28), cmap='gray')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
plt.tight_layout()
plt.show()

model.save('mnist_model.h5')
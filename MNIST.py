from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. MNIST dataset import
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))
train_size = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
print('x_train shape:{0}, y_train shape:{1}'.format(x_train.shape, y_train.shape))
print('x_test shape:{0}, y_test shape:{1}'.format(x_test.shape, y_test.shape))

# 2. Data pre-processing
x_train = x_train.reshape(56000, 784)
x_test = x_test.reshape(14000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:{0}, x_test shape:{1}'.format(x_train.shape, x_test.shape))

# 3. Model Construction
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()

# 4. Model Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Model training
loss, accuracy = [], []
for i in range(5):
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1)
    loss.append(model.evaluate(x_test, y_test)[0])
    accuracy.append(model.evaluate(x_test, y_test)[1])

fig, ax1 = plt.subplots()
ax1.set_xlabel('epoch')
ax1.set_ylabel('Accuracy')
ax1.plot(accuracy, color='green', label='Accuracy')

ax2 = ax1.twinx()
ax2.set_ylabel('Loss')
ax2.plot(loss, color='red', label='Loss')
ax1.legend(loc='center right')
ax2.legend(loc='center left')

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
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

predictions = model.predict(x_test)

plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()
print(f"Prediction: {tf.argmax(predictions[0])}")
print(f"Real label: {y_test[0]}")

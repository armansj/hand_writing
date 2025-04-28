import random

index = random.randint(0, 9999)
plt.imshow(x_test[index], cmap=plt.cm.binary)
plt.show()

prediction = model.predict(x_test[index].reshape(1, 28, 28))
predicted_label = tf.argmax(prediction[0])

print(f"Prediction: {predicted_label.numpy()}")
print(f"Real label: {y_test[index]}")

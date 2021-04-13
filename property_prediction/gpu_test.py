import tensorflow as tf
import time

start_time = time.time()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)

print("\nloss=", loss)
print("\naccuracy=", accuracy)
print("\nruntime", time.time() - start_time)

outF = open("myOutFile.txt", "a")
outF.write("\n")
outF.write("\nloss={}".format(loss))
outF.write("\naccuracy={}".format(accuracy))
outF.write("\nruntime={}".format(time.time() - start_time))
outF.close()
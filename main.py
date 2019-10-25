import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0

new_model = tf.keras.Sequential([
        keras.layers.Reshape(
            target_shape=[28, 28, 1],
            input_shape=(28, 28)),
        keras.layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
        keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        keras.layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
        keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu)])

opt = keras.optimizers.Adam(learning_rate=1e-3)
new_model.compile(optimizer=opt, loss=tfa.losses.triplet_semihard_loss)
new_model.fit(x_train, y_train, batch_size=64, epochs=20)
new_model.save("model.h5")

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

_, (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test / 255.0

trainedNet = tf.keras.models.load_model("model.h5")
untrainedNet = tf.keras.Sequential([
        keras.layers.Reshape(
            target_shape=[28, 28, 1],
            input_shape=(28, 28)),
        keras.layers.Conv2D(2, 5, padding='same', activation=tf.nn.relu),
        keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        keras.layers.Conv2D(4, 5, padding='same', activation=tf.nn.relu),
        keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation=tf.nn.relu)])


good_embeddings = trainedNet(x_test)
bad_embeddings = untrainedNet(x_test)

no_of_components = 2
pca = PCA(n_components=no_of_components)

trained_compoments = pca.fit_transform(good_embeddings)
untrained_components = pca.fit_transform(bad_embeddings)

test_class_labels = np.unique(np.array(y_test))

epochs = 20
step = 10
fig = plt.figure(figsize=(16, 8))
for label in test_class_labels:
    decomposed_embeddings_class = trained_compoments[y_test == label]
    decomposed_gray_class = untrained_components[y_test == label]

    plt.subplot(1, 2, 1)
    plt.scatter(decomposed_gray_class[::step, 1], decomposed_gray_class[::step, 0], label=str(label))
    plt.title('before training (embeddings)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0], label=str(label))
    plt.title('after @%d epochs' % epochs)
    plt.legend()

plt.show()

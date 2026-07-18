# train.py
import tensorflow as tf
from tensorflow.keras import layers, models, datasets


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


def build_model() -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    model.save("mnist_model.h5")


if __name__ == "__main__":
    main()
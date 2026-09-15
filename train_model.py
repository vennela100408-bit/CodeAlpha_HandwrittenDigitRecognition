import tensorflow as tf
from pathlib import Path

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "mnist_cnn.keras"


# ---------------------------------------------------------
# LOAD MNIST DATASET
# ---------------------------------------------------------
print("Loading MNIST dataset...")

(
    x_train,
    y_train
), (
    x_test,
    y_test
) = tf.keras.datasets.mnist.load_data()


# ---------------------------------------------------------
# NORMALIZE DATA
# ---------------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# ---------------------------------------------------------
# ADD CHANNEL DIMENSION
# ---------------------------------------------------------
x_train = x_train[..., None]
x_test = x_test[..., None]


# ---------------------------------------------------------
# CREATE CNN MODEL
# ---------------------------------------------------------
model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(28, 28, 1)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ---------------------------------------------------------
# COMPILE MODEL
# ---------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
print("Training CNN model...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------
print("Evaluating model...")

loss, accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------
model.save(MODEL_PATH)

print(
    f"Model saved successfully at: {MODEL_PATH}"
)
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import json
import os

# Configuración
NUM_CLASSES = 10
EPOCHS = 10
BATCH_SIZE = 64
os.makedirs("resultados", exist_ok=True)

# Carga y preprocesamiento de CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Definición de AlexNet
def build_alexnet(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential()

    # Conv 1
    model.add(layers.Conv2D(96, (3, 3), strides=1, padding="same",
                            activation="relu", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv 2
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Conv 3
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu"))

    # Conv 4
    model.add(layers.Conv2D(384, (3, 3), padding="same", activation="relu"))

    # Conv 5
    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Fully Connected
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

# Crear modelo
model = build_alexnet()
model.summary()

# Compilación
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Entrenamiento
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# Evaluación
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nPrecisión en test: {test_acc:.4f}")

# Guardar Métricas
metrics = {
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "train_accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
}
with open("resultados/metricas_alexnet.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("\n Métricas guardadas en resultados/metricas_alexnet.json")

# Guardar Modelo
model.save("resultados/alexnet_cifar10.h5")
print("Modelo guardado en resultados/alexnet_cifar10.h5")

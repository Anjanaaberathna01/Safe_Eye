import os
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns  # type: ignore
import tensorflow as tf # type: ignore
from sklearn.metrics import confusion_matrix, classification_report # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

DATA_DIR = "final_dataset"
RESULTS_DIR = "training_results_class"
os.makedirs(RESULTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SELECTED_CLASSES = ["drowsy", "non_drowsy", "no_yawn", "yawn"]
NUM_CLASSES = len(SELECTED_CLASSES)
EPOCHS = 8

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="categorical",
    class_names=SELECTED_CLASSES,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    labels="inferred",
    label_mode="categorical",
    class_names=SELECTED_CLASSES,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    labels="inferred",
    label_mode="categorical",
    class_names=SELECTED_CLASSES,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
print("Selected Classes (4):", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(RESULTS_DIR, "best_4class_drowsiness_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# 1. Accuracy & Loss Curves Save කිරීම
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history["val_accuracy"], label="Val Accuracy", marker="o")
plt.title("Model Accuracy (4-Class Full Face)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history["val_loss"], label="Val Loss", marker="o")
plt.title("Model Loss (4-Class Full Face)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=300)
plt.close()

# 2. Test Set Evaluation & Predictions
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ 4-Class Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# 3. Confusion Matrix Save කිරීම
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.title("Confusion Matrix (4-Class Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# 4. Classification Report Save කිරීම
report = classification_report(y_true, y_pred, target_names=class_names)
print("\n📊 Classification Report:\n", report)
with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\nTest Loss: {test_loss:.4f}\n\n")
    f.write(report)

print(f"\n🎉 All 4-class results saved to '{RESULTS_DIR}'!")
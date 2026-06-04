import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from src.config import (
    BALANCED_ROI_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    MODEL_OUTPUT_DIR,
    BALANCED_ROI_DIR,
    EPOCHS_BASE,
    EPOCHS_FINETUNE,
)

def build_dataset():
    """Loads the generated ROI images using Keras."""
    dataset = tf.keras.utils.image_dataset_from_directory(
        BALANCED_ROI_DIR,
        labels="inferred",
        label_mode="binary",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        seed=SEED,
        validation_split=0.2,
        subset="both",
    )
    return dataset[0], dataset[1]  # train_ds, val_ds


def train_model():
    train_ds, val_ds = build_dataset()

    # Data preprocessing built into the model
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    # Step 1: Base Model (Frozen)
    base_model = ResNet50(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    # Compile and train head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Training Classification Head...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_BASE)

    # Step 2: Fine-Tuning (Unfreeze top layers)
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Freeze bottom 100, fine-tune the rest
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower learning rate
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Fine-tuning top ResNet blocks...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)

    # Save the final model
    model.save(MODEL_OUTPUT_DIR / "final_stamp_classifier.keras")
    print(f"Model saved to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    train_model()

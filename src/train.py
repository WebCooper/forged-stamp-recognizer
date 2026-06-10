# =====================================================================
# DEEP LEARNING MODEL TRAINING (TRANSFER LEARNING)
# In this script, we train our classification model.
# Since we have a relatively small dataset, training a deep neural network 
# from scratch would overfit instantly. Instead, we use Transfer Learning:
# 1. Load a ResNet50 pre-trained on ImageNet (frozen).
# 2. Train a new binary classification head.
# 3. Unfreeze the top layers and fine-tune with a very small learning rate.
# =====================================================================

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
    """
    Loads the generated ROI images using Keras' directory loader.
    Splits the data 80% for training and 20% for validation.
    """
    # This utility is amazing! It handles image resizing, batching, shuffling, 
    # and splits the directories ('forged' and 'genuine') into labels for us.
    dataset = tf.keras.utils.image_dataset_from_directory(
        BALANCED_ROI_DIR,
        labels="inferred",
        label_mode="binary",      # Binary labels (0 or 1) since we have 2 classes
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        seed=SEED,                # Constant seed so we get the exact same split every run
        validation_split=0.2,     # 20% for validation
        subset="both",            # Returns both (train, validation) datasets
    )
    return dataset[0], dataset[1]  # train_ds, val_ds


def train_model():
    # Load training and validation datasets
    train_ds, val_ds = build_dataset()

    # Preprocessing function specific to ResNet50 (performs scaling/mean subtraction)
    # This matches the preprocessing used when ResNet50 was trained on ImageNet.
    preprocess_input = tf.keras.applications.resnet50.preprocess_input

    # STEP 1: BASE MODEL (FROZEN)
    # Load ResNet50 pre-trained on ImageNet. 
    # include_top=False means we discard the final 1000-class dense layer of ResNet
    # because we only need to classify between 2 classes (genuine vs forged).
    base_model = ResNet50(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    # Freeze the base model weights so we don't destroy pre-trained features!
    base_model.trainable = False

    # Define the Keras Functional API model structure
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # Apply ResNet preprocessing first
    x = preprocess_input(inputs)
    # Run through the frozen ResNet50 feature extractor.
    # training=False is important here to keep batch normalization layers in inference mode!
    x = base_model(x, training=False)
    # Global average pooling turns the 7x7 spatial features into a flat 1D vector (size 2048)
    x = layers.GlobalAveragePooling2D()(x)
    # Dropout of 30% randomly turns off neurons to prevent overfitting during training
    x = layers.Dropout(0.3)(x)
    # Dense sigmoid output layer (maps output to probability [0.0, 1.0])
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)

    # Compile the model with the frozen base. 
    # Use binary cross-entropy loss, which is standard for binary classification.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Training Classification Head...")
    # Train the new classification head for 15 epochs
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_BASE)

    # STEP 2: FINE-TUNING (UNFREEZE TOP LAYERS)
    # Now that the head is trained, we unfreeze the base model to fine-tune the high-level
    # feature extractors to recognize specific stamp texture details.
    base_model.trainable = True
    
    # We freeze the bottom 100 layers (which detect basic textures/edges)
    # and only unfreeze the remaining top layers (which detect complex stamp shapes)
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # RE-COMPILE! If we don't compile again, the unfreezing changes won't apply.
    # Note the learning rate is 100x smaller (1e-5). If we use a large rate (like 1e-3),
    # the backpropagation gradients would completely overwrite the pre-trained weights!
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("Fine-tuning top ResNet blocks...")
    # Fine-tune for 10 epochs
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)

    # Save the final model weights to a file so Streamlit can use it!
    model.save(MODEL_OUTPUT_DIR / "final_stamp_classifier.keras")
    print(f"Model saved to {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    train_model()


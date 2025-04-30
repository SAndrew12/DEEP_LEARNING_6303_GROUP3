import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNetV2, ResNet101V2, ResNet152V2, MobileNet,
    MobileNetV3Small, MobileNetV3Large
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda, Input, Dropout # Added Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.optimizers import Adam # Explicitly import Adam to set learning rate
from tensorflow.keras.callbacks import EarlyStopping # Import EarlyStopping
import pandas as pd
import numpy as np
import os
import sys

# Get the directory this script is in
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level (from code/ to Trashnet/)
trashnet_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Now build the path to the dataset
dataset_path = os.path.join(trashnet_dir, "Data", "dataset-resized")

# Print for confirmation
print(f"Current dir: {current_dir}")
print(f"Trashnet dir: {trashnet_dir}")
print(f"Dataset path: {dataset_path}")

# Check if path exists
if not os.path.isdir(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found at {dataset_path}")

#%%

img_size = (224, 224)
batch_size = 32
test_split = 0.2



# --- Training Parameters --- ### NEW
initial_epochs = 10
fine_tune_epochs = 10
initial_lr = 1e-3
fine_tune_lr = 1e-5
fine_tune_layers = 20
dropout_rate = 0.5
early_stopping_patience = 5

# --- Data Generators Setup ---
# Test set generator (using validation split from full dataset)
full_datagen_test = ImageDataGenerator(rescale=1./255, validation_split=test_split)
test_generator = full_datagen_test.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation', # Use the 'validation' subset as the TEST set
    class_mode='categorical',
    shuffle=False
)

# Train/Validation generator setup (using the remaining data)
# Note: validation_split here acts on the data *not* used by test_generator
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.25 # 25% of the (1 - test_split) data -> validation set
    # This means Train: 60%, Val: 20%, Test: 20% of total
)

train_generator = train_val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='training', # Use the 'training' subset from the split *within* this generator
    class_mode='categorical'
)

val_generator = train_val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation', # Use the 'validation' subset from the split *within* this generator
    class_mode='categorical'
)

print(f"Training images: {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")
print(f"Test images: {test_generator.samples}")

# --- Model Configurations ---
model_configs = [
    {
        'name': 'MobileNetV2',
        'class': MobileNetV2,
        'preprocess_fn': mobilenetv2_preprocess
    },
    {
        'name': 'ResNet101V2',
        'class': ResNet101V2,
        'preprocess_fn': resnet_preprocess
    },
    {
        'name': 'ResNet152V2',
        'class': ResNet152V2,
        'preprocess_fn': resnet_preprocess
    },
    {
        'name': 'MobileNet',
        'class': MobileNet,
        'preprocess_fn': mobilenet_preprocess
    },
    {
        'name': 'MobileNetV3Small',
        'class': MobileNetV3Small,
        'preprocess_fn': mobilenetv2_preprocess
    },
    {
        'name': 'MobileNetV3Large',
        'class': MobileNetV3Large,
        'preprocess_fn': mobilenetv2_preprocess
    },
]

# --- Results Collection ---
results = []

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    restore_best_weights=True, # Restore model weights from the epoch with the best val_loss
    verbose=1
)

# --- Training Loop ---
for config in model_configs:
    print(f"\n\033[1mProcessing {config['name']}\033[0m")

    # --- Build Model ---
    input_layer = Input(shape=(*img_size, 3), name="input_image")

    # Rescale back to 0-255 range *before* applying model-specific preprocessing
    x = Lambda(lambda img: img * 255.0, name="rescale_to_255")(input_layer)
    x = Lambda(config['preprocess_fn'], name="preprocessing")(x)

    base_model = config['class'](weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False

    # Add custom head
    x = GlobalAveragePooling2D(name="global_avg_pooling")(base_model.output)
    x = Dense(128, activation='relu', name="dense_128")(x)
    # Add dropout for regularization
    x = Dropout(dropout_rate, name="dropout")(x) # Added Dropout for regularization
    outputs = Dense(6, activation='softmax', name="output_softmax")(x)

    model = Model(input_layer, outputs)

    # --- Phase 1: Initial Training ---
    print(f"\n--- Initial Training ({config['name']}) ---")
    model.compile(optimizer=Adam(learning_rate=initial_lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_initial = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=initial_epochs,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        validation_steps=max(1, val_generator.samples // batch_size),
        # Use early stopping
        callbacks=[early_stopping]
    )

    initial_epochs_trained = len(history_initial.history['loss'])
    last_initial_val_acc = history_initial.history['val_accuracy'][-1]
    print(f"Initial training finished after {initial_epochs_trained} epochs. Last Val Acc: {last_initial_val_acc:.4f}")


    # --- Phase 2: Fine-tuning ---
    if fine_tune_epochs > 0 and fine_tune_layers > 0:
        print(f"\n--- Fine-tuning ({config['name']}) ---")
        # Unfreeze the base model and specific layers
        base_model.trainable = True

        # Freeze all layers except the top 'fine_tune_layers'
        print(f"Unfreezing the top {fine_tune_layers} layers of {config['name']}")
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False

        # Re-compile the model with a lower learning rate for fine-tuning
        model.compile(optimizer=Adam(learning_rate=fine_tune_lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Continue training
        history_fine_tune = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=initial_epochs_trained,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            validation_steps=max(1, val_generator.samples // batch_size),
            callbacks=[early_stopping]
        )
        history_final = history_fine_tune
    else:
        print("\n--- Skipping Fine-tuning ---")
        history_final = history_initial

    # --- Evaluate on Test Set ---
    print(f"\n--- Evaluating on Test Set ({config['name']}) ---")
    test_loss, test_acc = model.evaluate(test_generator, steps=max(1, test_generator.samples // batch_size))

    # --- Store Results ---
    # Use the history from the *last* training phase completed
    final_train_acc = history_final.history['accuracy'][-1]
    final_val_acc = history_final.history['val_accuracy'][-1]

    results.append({
        'Model': config['name'],
        'Test Accuracy': f"{test_acc:.4f}",
        'Test Loss': f"{test_loss:.4f}",
        'Final Train Accuracy': f"{final_train_acc:.4f}",
        'Final Val Accuracy': f"{final_val_acc:.4f}",
        'Initial Epochs': initial_epochs_trained,
        'FineTune Epochs': len(history_final.history['loss']) - initial_epochs_trained if fine_tune_epochs > 0 and fine_tune_layers > 0 else 0
    })

    # Save model and training history (optional - uncomment if needed)
    # model_save_path = f"./{config['name']}_trashnet_finetuned.h5"
    # history_save_path = f"./{config['name']}_finetuned_history.npy"
    # print(f"Saving model to {model_save_path}")
    # model.save(model_save_path)
    # print(f"Saving history to {history_save_path}")
    # np.save(history_save_path, history_final.history)


# --- Display Results ---
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Test Accuracy', ascending=False)

print("\n\033[1mModel Comparison Results (with Fine-tuning):\033[0m")
try:
    print(results_df.to_markdown(index=False))
except ImportError:
    print(results_df)

# Save results to CSV (optional - uncomment if needed)
# results_csv_path = './model_comparison_finetuned.csv'
# print(f"\nSaving results to {results_csv_path}")
# results_df.to_csv(results_csv_path, index=False)
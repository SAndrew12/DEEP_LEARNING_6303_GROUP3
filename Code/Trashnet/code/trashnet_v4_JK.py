import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNetV2, ResNet101V2, ResNet152V2, MobileNet,
    MobileNetV3Small, MobileNetV3Large, EfficientNetV2S)


from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess #new sd
try:
    from tensorflow.keras.optimizers import AdamW
except ImportError:
    from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt #new sd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay #new sd


# Get the directory this script is in
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level
trashnet_dir = os.path.abspath(os.path.join(current_dir, ".."))
# Now build the path to the dataset
dataset_path = os.path.join(trashnet_dir, "Data", "dataset-resized")

print(f"Current dir: {current_dir}")
print(f"Trashnet dir: {trashnet_dir}")
print(f"Dataset path: {dataset_path}")

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found at {dataset_path}")


#%%

img_size = (224, 224)
batch_size = 32
test_split = 0.2

# --- Training Parameters ---
initial_epochs = 15
fine_tune_epochs = 20
initial_lr = 1e-3
fine_tune_lr = 5e-6
fine_tune_layers = 50
dropout_rate = 0.5
early_stopping_patience = 7
lr_reduce_patience = 3
lr_reduce_factor = 0.2
weight_decay = 1e-4

# --- Data Generators Setup ---
# Test set generator
full_datagen_test = ImageDataGenerator(rescale=1./255, validation_split=test_split)
test_generator = full_datagen_test.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

# Train/Validation generator setup
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2, #new sd
    brightness_range=[0.7, 1.3], #new sd
    validation_split=0.25

)

train_generator = train_val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

val_generator = train_val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
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
    {
    'name': 'EfficientNetV2S', #new sd
    'class': EfficientNetV2S,
    'preprocess_fn': efficientnetv2_preprocess
}
]

# --- Results Collection ---
results = []

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    restore_best_weights=True,
    verbose=1
)

# Callback of reduced learning rate on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=lr_reduce_factor,
    patience=lr_reduce_patience,
    min_lr=1e-7,
    verbose=1
)

# Combine callbacks
callbacks_list = [early_stopping, reduce_lr]

# --- Training Loop ---
for config in model_configs:
    print(f"\n\033[1mProcessing {config['name']}\033[0m")

    # --- Build Model ---
    input_layer = Input(shape=(*img_size, 3), name="input_image")
    x = Lambda(lambda img: img * 255.0, name="rescale_to_255")(input_layer)
    x = Lambda(config['preprocess_fn'], name="preprocessing")(x)

    base_model = config['class'](weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False

    # Add new dense layer:
    x = GlobalAveragePooling2D(name="global_avg_pooling")(base_model.output)
    x = Dense(256, activation='relu', name="dense_256")(x)
    x = Dropout(dropout_rate, name="dropout_1")(x)
    x = Dense(128, activation='relu', name="dense_128")(x)
    x = Dropout(dropout_rate, name="dropout_2")(x)
    outputs = Dense(6, activation='softmax', name="output_softmax")(x)

    model = Model(input_layer, outputs)
    model.summary()

    # --- Phase 1: Initial Training ---
    print(f"\n--- Initial Training ({config['name']}) ---")
    # CHANGED: Use AdamW optimizer with weight decay
    optimizer_initial = AdamW(learning_rate=initial_lr, weight_decay=weight_decay)
    model.compile(optimizer=optimizer_initial,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_initial = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=initial_epochs, # Use increased initial epochs
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        validation_steps=max(1, val_generator.samples // batch_size),
        # Use combined callbacks list
        callbacks=callbacks_list
    )

    initial_epochs_trained = len(history_initial.history['loss'])
    # Get best validation accuracy achieved during this phase
    best_initial_val_acc = max(history_initial.history['val_accuracy']) if history_initial.history['val_accuracy'] else 0
    print(f"Initial training finished after {initial_epochs_trained} epochs. Best Val Acc: {best_initial_val_acc:.4f}")


    # --- Phase 2: Fine-tuning ---
    if fine_tune_epochs > 0 and fine_tune_layers > 0:
        print(f"\n--- Fine-tuning ({config['name']}) ---")
        base_model.trainable = True

        # CHANGED: Freeze all layers except the top 'fine_tune_layers' (now more layers)
        print(f"Unfreezing the top {fine_tune_layers} layers of {config['name']}")
        if len(base_model.layers) <= fine_tune_layers:
             print(f"Warning: fine_tune_layers ({fine_tune_layers}) >= number of layers in base model ({len(base_model.layers)}). Unfreezing all base layers.")
             for layer in base_model.layers:
                 layer.trainable = True
        else:
             for layer in base_model.layers[:-fine_tune_layers]:
                 layer.trainable = False
             # Optional: Make sure the unfrozen layers are actually trainable
             # for layer in base_model.layers[-fine_tune_layers:]:
             #    layer.trainable = True

        # Re-compile the model with a lower learning rate for fine-tuning
        # CHANGED: Use AdamW optimizer with fine-tune LR and weight decay
        optimizer_finetune = AdamW(learning_rate=fine_tune_lr, weight_decay=weight_decay)
        model.compile(optimizer=optimizer_finetune,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print(f"Starting fine-tuning with LR: {fine_tune_lr}")
        # Continue training
        history_fine_tune = model.fit(
            train_generator,
            validation_data=val_generator,
            # Use increased total epochs for fine-tuning phase
            epochs=initial_epochs_trained + fine_tune_epochs,
            initial_epoch=initial_epochs_trained,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            validation_steps=max(1, val_generator.samples // batch_size),
            callbacks=callbacks_list # Use combined callbacks list again
        )
        history_final = history_fine_tune
    else:
        print("\n--- Skipping Fine-tuning ---")
        history_final = history_initial

    # --- Evaluate on Test Set ---
    # Model weights should be from the best epoch due to restore_best_weights=True in EarlyStopping
    print(f"\n--- Evaluating on Test Set ({config['name']}) ---")
    test_loss, test_acc = model.evaluate(test_generator, steps=max(1, test_generator.samples // batch_size))

    # --- Save loss plots and conf. matrix --- sd
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, steps=int(np.ceil(test_generator.samples / batch_size)))
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax_cm, xticks_rotation=45)
    plt.title(f"Confusion Matrix - {config['name']}")
    confusion_matrix_path = os.path.join(current_dir, f"confusion_matrix_{config['name']}.png")
    fig_cm.savefig(confusion_matrix_path)
    plt.close(fig_cm)

    print(f"Saved Confusion Matrix for {config['name']} at {confusion_matrix_path}")


    # --- Training Curves ---
    # Combine histories safely
    total_acc = history_initial.history['accuracy'].copy()
    total_val_acc = history_initial.history['val_accuracy'].copy()
    total_loss = history_initial.history['loss'].copy()
    total_val_loss = history_initial.history['val_loss'].copy()

    if 'history_fine_tune' in locals():
        total_acc += history_fine_tune.history['accuracy']
        total_val_acc += history_fine_tune.history['val_accuracy']
        total_loss += history_fine_tune.history['loss']
        total_val_loss += history_fine_tune.history['val_loss']

    epochs_range = range(len(total_acc))

    fig_curve, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy Plot
    ax1.plot(epochs_range, total_acc, label='Training Accuracy')
    ax1.plot(epochs_range, total_val_acc, label='Validation Accuracy')  # <- FIXED
    ax1.set_title(f"Accuracy - {config['name']}")
    ax1.legend()

    # Loss Plot
    ax2.plot(epochs_range, total_loss, label='Training Loss')
    ax2.plot(epochs_range, total_val_loss, label='Validation Loss')
    ax2.set_title(f"Loss - {config['name']}")
    ax2.legend()

    plt.tight_layout()
    training_curve_path = os.path.join(current_dir, f"training_curves_{config['name']}.png")
    fig_curve.savefig(training_curve_path)
    plt.close(fig_curve)

    print(f"Saved Training Curves for {config['name']} at {training_curve_path}")
    # --- Save loss plots and conf. matrix --- sd

    # --- Store Results ---
    final_train_acc = history_final.history['accuracy'][-1] # Accuracy at the end of training run
    # Get the BEST validation accuracy achieved during the entire training process
    all_val_acc = history_initial.history['val_accuracy']
    if 'history_fine_tune' in locals(): # Check if fine-tuning happened
        all_val_acc.extend(history_fine_tune.history['val_accuracy'])
    best_overall_val_acc = max(all_val_acc) if all_val_acc else 0

    results.append({
        'Model': config['name'],
        'Test Accuracy': f"{test_acc:.4f}",
        'Test Loss': f"{test_loss:.4f}",
        'Best Val Accuracy': f"{best_overall_val_acc:.4f}", # Changed to report best overall val acc
        'Final Train Accuracy': f"{final_train_acc:.4f}",
        'Initial Epochs Run': initial_epochs_trained,
        'FineTune Epochs Run': len(history_final.history['loss']) - initial_epochs_trained if 'history_fine_tune' in locals() else 0
    })

    # Cleanup to avoid potential issues in loop if fine-tuning was skipped
    if 'history_fine_tune' in locals():
        del history_fine_tune

    # Optional: Save model (consider saving only the best model based on validation)
    # model_save_path = f"./{config['name']}_trashnet_improved.h5" ... etc


# --- Display Results ---
results_df = pd.DataFrame(results)
# CHANGED: Sort by best overall validation accuracy before looking at test accuracy
results_df = results_df.sort_values(by=['Best Val Accuracy', 'Test Accuracy'], ascending=[False, False])

print("\n\033[1mModel Comparison Results (Improved Tuning):\033[0m")
try:
    print(results_df.to_markdown(index=False))
except ImportError:
    print(results_df)

# Optional: Save results
# results_csv_path = './model_comparison_improved.csv' ... etc
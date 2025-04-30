#Based on Stephen's v5, with fine tuning of class weights, changing alpha level
import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNetV2, ResNet101V2, ResNet152V2, MobileNet,
    MobileNetV3Small, MobileNetV3Large, EfficientNetV2S
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
trashnet_dir = os.path.abspath(os.path.join(current_dir, ".."))
dataset_path = os.path.join(trashnet_dir, "Data", "dataset-resized")

print(f"Current dir: {current_dir}")
print(f"Trashnet dir: {trashnet_dir}")
print(f"Dataset path: {dataset_path}")

if not os.path.isdir(dataset_path):
    raise FileNotFoundError(f"Dataset directory not found at {dataset_path}")
# --- End Path Setup ---

#%%

img_size = (224, 224)
batch_size = 32
test_split = 0.2

# --- Training Parameters ---
initial_epochs = 10
fine_tune_epochs = 10
initial_lr = 1e-3
fine_tune_lr = 1e-5
fine_tune_layers = 20
dropout_rate = 0.5
early_stopping_patience = 5

# --- NEW JK ---
# 0.0 = No weighting (all weights 1.0)
# 1.0 = Full 'balanced' weighting
alpha_values_to_test = [0.0, 0.2, 0.5, 0.8, 1.0]

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
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    validation_split=0.25
)

train_generator = train_val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='categorical',
    shuffle=True # Shuffle training data
)

val_generator = train_val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical',
    shuffle=False # No need to shuffle validation
)

print(f"Training images: {train_generator.samples}")
print(f"Validation images: {val_generator.samples}")
print(f"Test images: {test_generator.samples}")
print(f"Class Indices: {train_generator.class_indices}")

# --- Calculate Class Weights ---
class_counts = {
    'cardboard': 403,
    'glass': 501,
    'metal': 410,
    'paper': 594,
    'plastic': 482,
    'trash': 137
}

class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

# Check if hardcoded labels match generator labels
if set(class_labels) != set(class_counts.keys()):
    print("WARNING: Hardcoded class counts labels do not match generator class labels!")
    print(f" Generator labels: {class_labels}")
    print(f" Hardcoded labels: {list(class_counts.keys())}")

# Generate a flat array of class labels repeated by count
# Map generator indices to the order of hardcoded counts if needed
label_to_index = train_generator.class_indices
index_to_label = {v: k for k, v in label_to_index.items()}

y_all = []
indices_for_weighting = []
for i in range(num_classes):
    label = index_to_label[i]
    if label in class_counts:
        count = class_counts[label]
        y_all.extend([i] * count)
        indices_for_weighting.append(i)
    else:
         print(f"Warning: Label '{label}' not found in hardcoded class_counts.")

y_all = np.array(y_all)
if len(y_all) == 0:
     raise ValueError("Could not create label array for class weight calculation based on hardcoded counts.")

# Compute base balanced weights
class_weight_balanced = compute_class_weight(
    class_weight='balanced',
    classes=np.array(indices_for_weighting),
    y=y_all
)
# Map balanced weights back to the full 0-5 range, setting weight to 1.0 for missing classes
base_weights = np.ones(num_classes)
for i, idx in enumerate(indices_for_weighting):
    base_weights[idx] = class_weight_balanced[i]


# --- Model Configurations ---
model_configs = [
    {
        'name': 'EfficientNetV2S',
        'class': EfficientNetV2S,
        'preprocess_fn': efficientnetv2_preprocess
    },
    {
        'name': 'MobileNetV2',
        'class': MobileNetV2,
        'preprocess_fn': mobilenetv2_preprocess
    },{
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
    }
]

# --- Overall Results Collection ---
all_results = [] # NEW JK

# ---  Outer loop for testing different alpha values --- NEW JK
for alpha in alpha_values_to_test:
    print(f"\n\n{'='*20} TESTING ALPHA = {alpha:.2f} {'='*20}\n")

    class_weight_damped = 1.0 + alpha * (base_weights - 1.0)
    class_weight_dict = dict(enumerate(class_weight_damped))
    print(f"Using Class Weights for alpha={alpha:.2f}: {class_weight_dict}")

    # --- Callbacks ---
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )

    # --- Training Loop ---
    for config in model_configs:
        print(f"\n--- Processing {config['name']} (Alpha: {alpha:.2f}) ---")

        # --- Build Model ---
        tf.keras.backend.clear_session()

        input_layer = Input(shape=(*img_size, 3), name="input_image")
        x = Lambda(lambda img: img * 255.0, name="rescale_to_255")(input_layer)
        x = Lambda(config['preprocess_fn'], name="preprocessing")(x)

        base_model = config['class'](weights='imagenet', include_top=False, input_tensor=x)
        base_model.trainable = False

        x = GlobalAveragePooling2D(name="global_avg_pooling")(base_model.output)
        x = Dense(128, activation='relu', name="dense_128")(x)
        x = Dropout(dropout_rate, name="dropout")(x)
        outputs = Dense(num_classes, activation='softmax', name="output_softmax")(x)

        model = Model(input_layer, outputs)

        # --- Phase 1: Initial Training ---
        print(f"\n--- Initial Training ({config['name']}, Alpha: {alpha:.2f}) ---")
        model.compile(optimizer=Adam(learning_rate=initial_lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        history_initial = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=initial_epochs,
            steps_per_epoch=max(1, train_generator.samples // batch_size),
            validation_steps=max(1, val_generator.samples // batch_size),
            callbacks=[early_stopping],
            class_weight=class_weight_dict # pass to current alpha
        )

        initial_epochs_trained = len(history_initial.history['loss'])
        last_initial_val_acc = history_initial.history['val_accuracy'][-1] if history_initial.history['val_accuracy'] else 0
        print(f"Initial training finished after {initial_epochs_trained} epochs. Last Val Acc: {last_initial_val_acc:.4f}")
        # --- End Phase 1 ---

        # --- Phase 2: Fine-tuning ---
        if fine_tune_epochs > 0 and fine_tune_layers > 0:
            print(f"\n--- Fine-tuning ({config['name']}, Alpha: {alpha:.2f}) ---")
            base_model.trainable = True
            print(f"Unfreezing the top {fine_tune_layers} layers of {config['name']}")
            if len(base_model.layers) <= fine_tune_layers:
                 print(f"Warning: fine_tune_layers ({fine_tune_layers}) >= number of layers in base model ({len(base_model.layers)}). Unfreezing all base layers.")
                 for layer in base_model.layers: layer.trainable = True
            else:
                 for layer in base_model.layers[:-fine_tune_layers]: layer.trainable = False

            model.compile(optimizer=Adam(learning_rate=fine_tune_lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            history_fine_tune = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=initial_epochs_trained + fine_tune_epochs,
                initial_epoch=initial_epochs_trained,
                steps_per_epoch=max(1, train_generator.samples // batch_size),
                validation_steps=max(1, val_generator.samples // batch_size),
                callbacks=[early_stopping],
                class_weight=class_weight_dict # pass to current alpha
            )
            history_final = history_fine_tune
        else:
            print("\n--- Skipping Fine-tuning ---")
            history_final = history_initial
        # --- End Phase 2 ---

        # --- Evaluate on Test Set ---
        print(f"\n--- Evaluating on Test Set ({config['name']}, Alpha: {alpha:.2f}) ---")
        test_generator.reset()
        test_loss, test_acc = model.evaluate(test_generator, steps=max(1, test_generator.samples // batch_size))

        # --- Save loss plots and conf. matrix ---
        # Reset again before prediction
        test_generator.reset()
        y_pred_probs = model.predict(test_generator, steps=int(np.ceil(test_generator.samples / batch_size)))
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true = test_generator.classes

        cm = confusion_matrix(y_true, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
        fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
        disp.plot(cmap='Blues', ax=ax_cm, xticks_rotation=45)
        plt.title(f"Confusion Matrix - {config['name']} (Alpha: {alpha:.2f})") #NEW JK, add alpha into title
        confusion_matrix_path = os.path.join(current_dir, f"confusion_matrix_{config['name']}_alpha{alpha:.2f}.png")
        try:
            fig_cm.savefig(confusion_matrix_path)
            print(f"Saved Confusion Matrix for {config['name']} (Alpha: {alpha:.2f}) at {confusion_matrix_path}")
        except Exception as e:
            print(f"Error saving confusion matrix: {e}")
        plt.close(fig_cm)

        total_acc = history_initial.history['accuracy'].copy()
        total_val_acc = history_initial.history['val_accuracy'].copy()
        total_loss = history_initial.history['loss'].copy()
        total_val_loss = history_initial.history['val_loss'].copy()
        if 'history_fine_tune' in locals():
            total_acc.extend(history_fine_tune.history['accuracy'])
            total_val_acc.extend(history_fine_tune.history['val_accuracy'])
            total_loss.extend(history_fine_tune.history['loss'])
            total_val_loss.extend(history_fine_tune.history['val_loss'])
        epochs_range = range(len(total_acc))

        fig_curve, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.plot(epochs_range, total_acc, label='Training Accuracy')
        ax1.plot(epochs_range, total_val_acc, label='Validation Accuracy')
        # CHANGED: Include alpha in plot title and filename
        ax1.set_title(f"Accuracy - {config['name']} (Alpha: {alpha:.2f})")
        ax1.legend()
        ax2.plot(epochs_range, total_loss, label='Training Loss')
        ax2.plot(epochs_range, total_val_loss, label='Validation Loss')
        ax2.set_title(f"Loss - {config['name']} (Alpha: {alpha:.2f})")
        ax2.legend()
        plt.tight_layout()
        training_curve_path = os.path.join(current_dir, f"training_curves_{config['name']}_alpha{alpha:.2f}.png")
        try:
            fig_curve.savefig(training_curve_path)
            print(f"Saved Training Curves for {config['name']} (Alpha: {alpha:.2f}) at {training_curve_path}")
        except Exception as e:
            print(f"Error saving training curves: {e}")
        plt.close(fig_curve)

        # --- Store Results ---
        final_train_acc = history_final.history['accuracy'][-1] if history_final.history['accuracy'] else 0
        final_val_acc = history_final.history['val_accuracy'][-1] if history_final.history['val_accuracy'] else 0

        all_results.append({
            'Alpha': alpha, #NEW JK
            'Model': config['name'],
            'Test Accuracy': f"{test_acc:.4f}",
            'Test Loss': f"{test_loss:.4f}",
            'Final Train Accuracy': f"{final_train_acc:.4f}",
            'Final Val Accuracy': f"{final_val_acc:.4f}",
            'Initial Epochs Run': initial_epochs_trained,
            'FineTune Epochs Run': len(history_final.history['loss']) - initial_epochs_trained if 'history_fine_tune' in locals() else 0
        })

        # Cleanup optional history variable
        if 'history_fine_tune' in locals():
            del history_fine_tune



# --- Display Final Results ---
results_df = pd.DataFrame(all_results)
# CHANGED: Sort by Alpha first, then Test Accuracy
results_df = results_df.sort_values(by=['Alpha', 'Test Accuracy'], ascending=[True, False])

print("\n\033[1mOverall Model Comparison Results (Across Alphas):\033[0m")
try:
    print(results_df.to_markdown(index=False))
except ImportError:
    print(results_df)

# Save final combined results (optional)
# results_csv_path = './model_comparison_alpha_tuning.csv'
# print(f"\nSaving final results to {results_csv_path}")
# results_df.to_csv(results_csv_path, index=False)
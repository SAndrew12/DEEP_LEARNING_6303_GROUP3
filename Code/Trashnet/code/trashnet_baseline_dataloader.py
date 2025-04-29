import tensorflow as tf
from tensorflow.keras.applications import (
    MobileNetV2, ResNet101V2, ResNet152V2, MobileNet,
    MobileNetV3Small, MobileNetV3Large
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
import pandas as pd
import numpy as np
import os
#%%

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
epochs = 10
test_split = 0.3

# Data generators setup
full_datagen = ImageDataGenerator(rescale=1./255, validation_split=test_split)

# Test set generator
test_generator = full_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical',
    shuffle=False
)

# Train/val generator with augmentation
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
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

# Model configurations with proper preprocessing
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

# Results collection
results = []

for config in model_configs:
    print(f"\n\033[1mTraining {config['name']}\033[0m")

    # Build model with preprocessing
    input_layer = Input(shape=(*img_size, 3))
    x = Lambda(lambda x: config['preprocess_fn'](x * 255.))(input_layer)
    base_model = config['class'](weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    base_model.trainable = False

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(6, activation='softmax')(x)

    model = Model(input_layer, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_steps=val_generator.samples // batch_size
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    results.append({
        'Model': config['name'],
        'Test Accuracy': f"{test_acc:.4f}",
        'Test Loss': f"{test_loss:.4f}",
        'Train Accuracy': f"{history.history['accuracy'][-1]:.4f}",
        'Val Accuracy': f"{history.history['val_accuracy'][-1]:.4f}"
    })





results_df = pd.DataFrame(results)
# CHANGED: Sort by best overall validation accuracy before looking at test accuracy
results_df




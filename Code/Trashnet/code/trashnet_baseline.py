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
dataset_path = '/DEEP_LEARNING_6303_GROUP3/Code/Trashnet/data/dataset-resized'
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

    # Save model and training history
    # model.save(f"/kaggle/working/{config['name']}_trashnet.h5")
    # np.save(f"/kaggle/working/{config['name']}_history.npy", history.history)

# # Save results to CSV
# results_df = pd.DataFrame(results)
# results_df = results_df.sort_values(by='Test Accuracy', ascending=False)
# results_df.to_csv('/kaggle/working/model_comparison.csv', index=False)
#
# print("\n\033[1mModel Comparison Results:\033[0m")
# print(results_df.to_markdown(index=False))

# from IPython.display import FileLink
# FileLink('/kaggle/working/ResNet152V2_history.npy')


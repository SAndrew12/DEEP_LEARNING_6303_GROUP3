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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight


class Dataset:
    """
    Class to manage data loading and preprocessing using ImageDataGenerator.
    """
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32, test_split=0.2, val_split=0.25):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.class_labels = None
        self.num_classes = None
        self.class_weight_dict = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def setup_generators(self):
        """
        Initialize train, validation, and test data generators.
        """
        # Test set generator
        full_datagen_test = ImageDataGenerator(rescale=1./255, validation_split=self.test_split)
        self.test_generator = full_datagen_test.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            subset='validation',
            class_mode='categorical',
            shuffle=False
        )

        # Train/Validation generator
        train_val_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            validation_split=self.val_split
        )

        self.train_generator = train_val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            subset='training',
            class_mode='categorical'
        )

        self.val_generator = train_val_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            subset='validation',
            class_mode='categorical'
        )

        # Setup class weights
        self.class_labels = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_labels)
        self._compute_class_weights()

    def _compute_class_weights(self):
        """
        Compute balanced class weights to handle class imbalance.
        """
        class_counts = {
            'cardboard': 403,
            'glass': 501,
            'metal': 410,
            'paper': 594,
            'plastic': 482,
            'trash': 137
        }
        y_all = np.concatenate([
            np.full(class_counts[label], idx) for idx, label in enumerate(self.class_labels)
        ])
        class_weight_values = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(self.num_classes),
            y=y_all

        )
        alpha = 0.2  # Reduce weighting strength
        class_weight_values = 1.0 + alpha * (class_weight_values - 1.0)
        self.class_weight_dict = dict(enumerate(class_weight_values))

    def get_generators(self):
        """
        Return the train, validation, and test generators.
        """
        return self.train_generator, self.val_generator, self.test_generator


class TrashNetModel:
    """
    Class to define and manage the CNN model.
    """
    def __init__(self, model_config, img_size=(224, 224), num_classes=6, dropout_rate=0.5):
        self.model_config = model_config
        self.img_size = img_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None

    def build_model(self):
        """
        Build the model using the specified pre-trained base model.
        """
        input_layer = Input(shape=(*self.img_size, 3), name="input_image")
        x = Lambda(lambda img: img * 255.0, name="rescale_to_255")(input_layer)
        x = Lambda(self.model_config['preprocess_fn'], name="preprocessing")(x)

        self.base_model = self.model_config['class'](
            weights='imagenet',
            include_top=False,
            input_tensor=x
        )
        self.base_model.trainable = False

        x = GlobalAveragePooling2D(name="global_avg_pooling")(self.base_model.output)
        x = Dense(128, activation='relu', name="dense_128")(x)
        x = Dropout(self.dropout_rate, name="dropout")(x)
        outputs = Dense(self.num_classes, activation='softmax', name="output_softmax")(x)

        self.model = Model(input_layer, outputs)

    def compile_model(self, learning_rate):
        """
        Compile the model with the specified learning rate.
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def unfreeze_layers(self, fine_tune_layers):
        """
        Unfreeze the top fine_tune_layers for fine-tuning.
        """
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-fine_tune_layers]:
            layer.trainable = False


def train_and_evaluate(dataset, model, initial_epochs, fine_tune_epochs, initial_lr, fine_tune_lr,
                       fine_tune_layers, early_stopping_patience, class_weight_dict, model_name, output_dir):
    """
    Train and evaluate the model with initial training and optional fine-tuning.
    """
    train_generator, val_generator, test_generator = dataset.get_generators()
    results = {'Model': model_name}

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )

    # Initial Training
    print(f"\n--- Initial Training ({model_name}) ---")
    model.compile_model(initial_lr)
    history_initial = model.model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=initial_epochs,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_steps=max(1, val_generator.samples // val_generator.batch_size),
        callbacks=[early_stopping],
        class_weight=class_weight_dict
    )

    initial_epochs_trained = len(history_initial.history['loss'])
    last_initial_val_acc = history_initial.history['val_accuracy'][-1]
    print(f"Initial training finished after {initial_epochs_trained} epochs. Last Val Acc: {last_initial_val_acc:.4f}")
    results['Initial Epochs'] = initial_epochs_trained
    results['Final Train Accuracy'] = f"{history_initial.history['accuracy'][-1]:.4f}"
    results['Final Val Accuracy'] = f"{last_initial_val_acc:.4f}"

    # Fine-tuning
    if fine_tune_epochs > 0 and fine_tune_layers > 0:
        print(f"\n--- Fine-tuning ({model_name}) ---")
        model.unfreeze_layers(fine_tune_layers)
        model.compile_model(fine_tune_lr)
        history_fine_tune = model.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=initial_epochs_trained,
            steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
            validation_steps=max(1, val_generator.samples // val_generator.batch_size),
            callbacks=[early_stopping],
            class_weight=class_weight_dict
        )
        history_final = history_fine_tune
        results['FineTune Epochs'] = len(history_final.history['loss']) - initial_epochs_trained
    else:
        print("\n--- Skipping Fine-tuning ---")
        history_final = history_initial
        results['FineTune Epochs'] = 0

    # Evaluate on Test Set
    print(f"\n--- Evaluating on Test Set ({model_name}) ---")
    test_loss, test_acc = model.model.evaluate(
        test_generator,
        steps=max(1, test_generator.samples // test_generator.batch_size)
    )
    results['Test Accuracy'] = f"{test_acc:.4f}"
    results['Test Loss'] = f"{test_loss:.4f}"

    # Generate Plots
    _generate_plots(history_initial, history_fine_tune if 'history_fine_tune' in locals() else None,
                    test_generator, model.model, model_name, output_dir, dataset.class_labels)

    # Save the model
    model_save_path = os.path.join(output_dir, f"{model_name}_trashnet_finetuned.h5")
    model.model.save(model_save_path)
    print(f"Saved model to {model_save_path}")

    # Save the training history
    history_save_path = os.path.join(output_dir, f"{model_name}_finetuned_history.npy")
    history_dict = history_final.history
    np.save(history_save_path, history_dict)
    print(f"Saved training history to {history_save_path}")

    return results


def _generate_plots(history_initial, history_fine_tune, test_generator, model, model_name, output_dir, class_labels):
    """
    Generate and save confusion matrix and training curves.
    """
    # Confusion Matrix
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, steps=int(np.ceil(test_generator.samples / test_generator.batch_size)))
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
    disp.plot(cmap='Blues', ax=ax_cm, xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name}")
    confusion_matrix_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    fig_cm.savefig(confusion_matrix_path)
    plt.close(fig_cm)
    print(f"Saved Confusion Matrix for {model_name} at {confusion_matrix_path}")

    # Training Curves
    total_acc = history_initial.history['accuracy'].copy()
    total_val_acc = history_initial.history['val_accuracy'].copy()
    total_loss = history_initial.history['loss'].copy()
    total_val_loss = history_initial.history['val_loss'].copy()

    if history_fine_tune:
        total_acc += history_fine_tune.history['accuracy']
        total_val_acc += history_fine_tune.history['val_accuracy']
        total_loss += history_fine_tune.history['loss']
        total_val_loss += history_fine_tune.history['val_loss']

    epochs_range = range(len(total_acc))

    fig_curve, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(epochs_range, total_acc, label='Training Accuracy')
    ax1.plot(epochs_range, total_val_acc, label='Validation Accuracy')
    ax1.set_title(f"Accuracy - {model_name}")
    ax1.legend()

    ax2.plot(epochs_range, total_loss, label='Training Loss')
    ax2.plot(epochs_range, total_val_loss, label='Validation Loss')
    ax2.set_title(f"Loss - {model_name}")
    ax2.legend()

    plt.tight_layout()
    training_curve_path = os.path.join(output_dir, f"training_curves_{model_name}.png")
    fig_curve.savefig(training_curve_path)
    plt.close(fig_curve)
    print(f"Saved Training Curves for {model_name} at {training_curve_path}")


def main():
    """
    Main function to orchestrate the training and evaluation process.
    """
    # Setup paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    trashnet_dir = os.path.abspath(os.path.join(current_dir, ".."))
    dataset_path = os.path.join(trashnet_dir, "data", "dataset-resized")

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_path}")

    print(f"Current dir: {current_dir}")
    print(f"Trashnet dir: {trashnet_dir}")
    print(f"Dataset path: {dataset_path}")

    # Training parameters
    initial_epochs = 10
    fine_tune_epochs = 10
    initial_lr = 1e-3
    fine_tune_lr = 1e-5
    fine_tune_layers = 20
    dropout_rate = 0.5
    early_stopping_patience = 5
    batch_size = 32
    img_size = (224, 224)
    test_split = 0.2
    val_split = 0.25

    # Model configurations
    model_configs = [
        {'name': 'EfficientNetV2S', 'class': EfficientNetV2S, 'preprocess_fn': efficientnetv2_preprocess},
        {'name': 'MobileNetV2', 'class': MobileNetV2, 'preprocess_fn': mobilenetv2_preprocess},
        {'name': 'ResNet101V2', 'class': ResNet101V2, 'preprocess_fn': resnet_preprocess},
        {'name': 'ResNet152V2', 'class': ResNet152V2, 'preprocess_fn': resnet_preprocess},
        {'name': 'MobileNet', 'class': MobileNet, 'preprocess_fn': mobilenet_preprocess},
        {'name': 'MobileNetV3Small', 'class': MobileNetV3Small, 'preprocess_fn': mobilenetv2_preprocess},
        {'name': 'MobileNetV3Large', 'class': MobileNetV3Large, 'preprocess_fn': mobilenetv2_preprocess}
    ]

    # Initialize dataset
    dataset = Dataset(dataset_path, img_size, batch_size, test_split, val_split)
    dataset.setup_generators()
    train_generator, val_generator, test_generator = dataset.get_generators()
    print(f"Training images: {train_generator.samples}")
    print(f"Validation images: {val_generator.samples}")
    print(f"Test images: {test_generator.samples}")
    print("Class Weights:", dataset.class_weight_dict)

    # Train and evaluate models
    results = []
    for config in model_configs:
        print(f"\n\033[1mProcessing {config['name']}\033[0m")
        model = TrashNetModel(config, img_size, dataset.num_classes, dropout_rate)
        model.build_model()
        result = train_and_evaluate(
            dataset, model, initial_epochs, fine_tune_epochs, initial_lr, fine_tune_lr,
            fine_tune_layers, early_stopping_patience, dataset.class_weight_dict,
            config['name'], current_dir
        )
        results.append(result)

    # Display results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Test Accuracy', ascending=False)
    print("\n\033[1mModel Comparison Results (with Fine-tuning):\033[0m")
    try:
        print(results_df.to_markdown(index=False))
    except ImportError:
        print(results_df)


if __name__ == '__main__':
    main()
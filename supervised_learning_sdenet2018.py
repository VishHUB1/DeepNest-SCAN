import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0, VGG16, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import uuid
import datetime

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define paths
base_data_dir = r'C:\Users\visha\Desktop\crack_project\SDNET2018'  # Update with actual path
results_dir = 'C:/Users/visha/Desktop/crack_project/supervised_SDENET2018'
os.makedirs(results_dir, exist_ok=True)

# Image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 10


# Prepare dataset for a specific category (D, P, or W)
def load_data(category_dir, category):
    positive_dir = os.path.join(category_dir, f'C{category}')
    negative_dir = os.path.join(category_dir, f'U{category}')

    positive_images = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if
                       f.endswith(('.jpg', '.jpeg', '.png'))]
    negative_images = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if
                       f.endswith(('.jpg', '.jpeg', '.png'))]

    positive_labels = [1] * len(positive_images)
    negative_labels = [0] * len(negative_images)

    all_images = positive_images + negative_images
    all_labels = positive_labels + negative_labels

    # Train-test split
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )

    return train_images, test_images, train_labels, test_labels


# Create data generators
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    return train_datagen, test_datagen


# Build model
def build_model(base_model, model_name):
    base_model.trainable = False  # Freeze base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# Train and evaluate model
def train_and_evaluate(model, model_name, train_generator, val_generator, test_images, test_labels,
                       category_results_dir):
    # Create model-specific results directory
    model_results_dir = os.path.join(category_results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )

    # Save model
    model.save(os.path.join(model_results_dir, f'{model_name}.h5'))

    # Evaluate on test set
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': test_images, 'class': [str(l) for l in test_labels]}),
        x_col='filename',
        y_col='class',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_labels

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred)
    }

    # Save metrics
    with open(os.path.join(model_results_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_results_dir, 'training_history.png'))
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(model_results_dir, 'confusion_matrix.png'))
    plt.close()

    return metrics, history.history


# Compare all models for a category
def compare_models(metrics_dict, category_results_dir):
    comparison_df = pd.DataFrame(metrics_dict).T
    comparison_df.to_csv(os.path.join(category_results_dir, 'model_comparison.csv'))

    # Plot comparison bar chart
    plt.figure(figsize=(12, 6))
    comparison_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(category_results_dir, 'model_comparison.png'))
    plt.close()


def main():
    # Define categories
    categories = ['D', 'P', 'W']  # Deck, Pavement, Walls

    # Define models
    models = {
        'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        'DenseNet121': DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        'EfficientNetB0': EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    }

    for category in categories:
        print(f'Processing category {category}...')
        category_dir = os.path.join(base_data_dir, category)
        category_results_dir = os.path.join(results_dir, category)
        os.makedirs(category_results_dir, exist_ok=True)

        # Load data for category
        train_images, test_images, train_labels, test_labels = load_data(category_dir, category)

        # Create data generators
        train_datagen, test_datagen = create_data_generators()

        # Create dataframes for generators
        train_df = pd.DataFrame({'filename': train_images, 'class': [str(l) for l in train_labels]})
        test_df = pd.DataFrame({'filename': test_images, 'class': [str(l) for l in test_labels]})

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col='filename',
            y_col='class',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        val_generator = test_datagen.flow_from_dataframe(
            dataframe=test_df,
            x_col='filename',
            y_col='class',
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        metrics_dict = {}

        # Train and evaluate each model
        for model_name, base_model in models.items():
            print(f'Training {model_name} for category {category}...')
            model = build_model(base_model, model_name)
            metrics, history = train_and_evaluate(model, model_name, train_generator, val_generator,
                                                  test_images, test_labels, category_results_dir)
            metrics_dict[model_name] = metrics

        # Compare models for this category
        compare_models(metrics_dict, category_results_dir)


if __name__ == '__main__':
    main()
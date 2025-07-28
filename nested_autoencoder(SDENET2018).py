import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Reshape
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import cv2
from datetime import datetime
import seaborn as sns
import pandas as pd
import time
import random
import uuid

# Clear previous TensorFlow session
tf.keras.backend.clear_session()

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class NestedAutoencoder:
    def __init__(self, input_shape=(128, 128, 3),
                 results_dir='./results',
                 cracked_dir='./data/cracked',
                 uncracked_dir='./data/uncracked'):
        """
        Initialize the nested autoencoder model for crack detection

        Args:
            input_shape: Shape of input images (height, width, channels)
            results_dir: Directory to save results and visualizations
            cracked_dir: Directory containing cracked images
            uncracked_dir: Directory containing uncracked images
        """
        self.input_shape = input_shape
        self.results_dir = results_dir
        self.cracked_dir = cracked_dir
        self.uncracked_dir = uncracked_dir

        # Create results directory structure
        os.makedirs(self.results_dir, exist_ok=True)
        self.summary_dir = os.path.join(self.results_dir, 'summary')
        os.makedirs(self.summary_dir, exist_ok=True)

        # Build the model on CPU
        with tf.device('/CPU:0'):
            self.model = self._build_model()

    def _build_model(self):
        """Build a nested autoencoder model"""
        input_img = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        shape_before_flatten = x.shape[1:]
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        encoded = Dense(128, activation='relu')(x)
        x = Dense(256, activation='relu')(encoded)
        x = Dense(512, activation='relu')(x)
        x = Dense(np.prod(shape_before_flatten), activation='relu')(x)
        x = Reshape(shape_before_flatten)(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def load_and_preprocess_images(self, filepaths, target_size=None):
        """
        Load and preprocess images from a list of file paths

        Args:
            filepaths: List of image file paths
            target_size: Target size for resizing (height, width)

        Returns:
            Preprocessed images as numpy array, corresponding filenames
        """
        if target_size is None:
            target_size = self.input_shape[:2]
        images = []
        filenames = []
        for img_path in filepaths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read {img_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (target_size[1], target_size[0]))
                img = img.astype('float32') / 255.0
                images.append(img)
                filenames.append(os.path.basename(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        return np.array(images), filenames

    def prepare_data(self):
        """
        Prepare training and testing data according to specifications

        Returns:
            Training images, testing images, testing labels, testing filenames
        """
        # Load all cracked and uncracked image paths
        cracked_files = [os.path.join(self.cracked_dir, f) for f in os.listdir(self.cracked_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        uncracked_files = [os.path.join(self.uncracked_dir, f) for f in os.listdir(self.uncracked_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Ensure sufficient images
        if len(cracked_files) == 0 or len(uncracked_files) == 0:
            raise ValueError("Empty dataset found")

        # Match uncracked set size to cracked set size for testing
        n_cracked = len(cracked_files)
        test_uncracked = random.sample(uncracked_files, min(n_cracked, len(uncracked_files)))
        train_uncracked = list(set(uncracked_files) - set(test_uncracked))

        # Use all cracked images for testing
        test_cracked = cracked_files

        # Use remaining uncracked images for training
        train_images, _ = self.load_and_preprocess_images(train_uncracked)

        # Load testing images
        test_images_cracked, test_filenames_cracked = self.load_and_preprocess_images(test_cracked)
        test_images_uncracked, test_filenames_uncracked = self.load_and_preprocess_images(test_uncracked)

        test_images = np.vstack([test_images_uncracked, test_images_cracked])
        test_filenames = test_filenames_uncracked + test_filenames_cracked
        test_labels = np.array([0] * len(test_images_uncracked) + [1] * len(test_images_cracked))

        return train_images, test_images, test_labels, test_filenames

    def train(self, train_images, iteration, batch_size=32, epochs=50, validation_split=0.2):
        """
        Train the autoencoder model on CPU

        Args:
            train_images: Training images
            iteration: Current iteration number
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data for validation

        Returns:
            Training history
        """
        iter_dir = os.path.join(self.results_dir, f'iteration_{iteration}')
        model_dir = os.path.join(iter_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
        with tf.device('/CPU:0'):
            history = self.model.fit(
                train_images, train_images,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=validation_split,
                shuffle=True,
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
        self.model.save(os.path.join(model_dir, 'final_model.keras'))
        self._save_training_history(history, iteration)
        return history

    def _save_training_history(self, history, iteration):
        """Save training history plots"""
        iter_dir = os.path.join(self.results_dir, f'iteration_{iteration}')
        visualization_dir = os.path.join(iter_dir, 'visualizations')
        metrics_dir = os.path.join(iter_dir, 'metrics')
        os.makedirs(visualization_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        plt.figure(figsize=(12, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'training_history.png'))
        plt.close()

        pd.DataFrame(history.history).to_csv(
            os.path.join(metrics_dir, 'training_history.csv'),
            index=False
        )

    def evaluate(self, test_images, test_labels, test_filenames, iteration, threshold=None):
        """
        Evaluate the model on test data on CPU

        Args:
            test_images: Test images
            test_labels: True labels
            test_filenames: Filenames of test images
            iteration: Current iteration number
            threshold: Reconstruction error threshold

        Returns:
            Evaluation metrics
        """
        iter_dir = os.path.join(self.results_dir, f'iteration_{iteration}')
        visualization_dir = os.path.join(iter_dir, 'visualizations')
        metrics_dir = os.path.join(iter_dir, 'metrics')
        os.makedirs(visualization_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        with tf.device('/CPU:0'):
            predictions = self.model.predict(test_images)
        mse = np.mean(np.square(test_images - predictions), axis=(1, 2, 3))

        if threshold is None:
            fpr, tpr, thresholds = roc_curve(test_labels, mse)
            optimal_idx = np.argmax(tpr - fpr)
            threshold = thresholds[optimal_idx]
            print(f"Iteration {iteration} - Automatically determined threshold: {threshold:.6f}")

        predicted_labels = (mse > threshold).astype(int)
        self._save_reconstruction_visualization(test_images, predictions, mse, threshold, test_filenames, iteration)
        metrics = self._calculate_metrics(test_labels, predicted_labels, mse, iteration)
        metrics['threshold'] = threshold
        return metrics

    def _save_reconstruction_visualization(self, original_images, reconstructed_images,
                                           reconstruction_errors, threshold, filenames, iteration,
                                           max_samples=10):
        """Save visualization of original vs reconstructed images"""
        visualization_dir = os.path.join(self.results_dir, f'iteration_{iteration}', 'visualizations')
        num_samples = min(max_samples, len(original_images))
        normal_indices = np.where(reconstruction_errors <= threshold)[0]
        anomaly_indices = np.where(reconstruction_errors > threshold)[0]
        selected_indices = []
        if len(normal_indices) > 0:
            selected_indices.extend(
                np.random.choice(normal_indices, min(num_samples // 2, len(normal_indices)), replace=False))
        if len(anomaly_indices) > 0:
            selected_indices.extend(
                np.random.choice(anomaly_indices, min(num_samples - len(selected_indices), len(anomaly_indices)),
                                 replace=False))
        if len(selected_indices) < num_samples:
            remaining = list(set(range(len(original_images))) - set(selected_indices))
            if remaining:
                selected_indices.extend(np.random.choice(remaining, num_samples - len(selected_indices), replace=False))

        plt.figure(figsize=(15, 3 * len(selected_indices)))
        for i, idx in enumerate(selected_indices):
            plt.subplot(len(selected_indices), 3, 3 * i + 1)
            plt.imshow(original_images[idx])
            plt.title(f"Original: {os.path.basename(filenames[idx])}")
            plt.axis('off')
            plt.subplot(len(selected_indices), 3, 3 * i + 2)
            plt.imshow(reconstructed_images[idx])
            plt.title('Reconstructed')
            plt.axis('off')
            plt.subplot(len(selected_indices), 3, 3 * i + 3)
            difference = np.abs(original_images[idx] - reconstructed_images[idx])
            plt.imshow(difference, cmap='hot')
            is_anomaly = "YES" if reconstruction_errors[idx] > threshold else "NO"
            plt.title(f"Error: {reconstruction_errors[idx]:.6f}\nAnomaly: {is_anomaly}")
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'reconstructions.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.histplot(reconstruction_errors, bins=50, kde=True)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
        plt.title('Distribution of Reconstruction Errors')
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(visualization_dir, 'error_distribution.png'))
        plt.close()

    def _calculate_metrics(self, true_labels, predicted_labels, reconstruction_errors, iteration):
        """Calculate and save evaluation metrics"""
        metrics_dir = os.path.join(self.results_dir, f'iteration_{iteration}', 'metrics')
        visualization_dir = os.path.join(self.results_dir, f'iteration_{iteration}', 'visualizations')

        cm = confusion_matrix(true_labels, predicted_labels)
        cr = classification_report(true_labels, predicted_labels, output_dict=True)
        fpr, tpr, _ = roc_curve(true_labels, reconstruction_errors)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(true_labels, reconstruction_errors)
        pr_auc = auc(recall, precision)

        cr_df = pd.DataFrame(cr).transpose()
        cr_df.to_csv(os.path.join(metrics_dir, 'classification_report.csv'))

        metrics = {
            'accuracy': cr['accuracy'],
            'precision': cr['1']['precision'],
            'recall': cr['1']['recall'],
            'f1-score': cr['1']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

        pd.DataFrame([metrics]).to_csv(os.path.join(metrics_dir, 'metrics_summary.csv'), index=False)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Uncracked', 'Cracked'],
                    yticklabels=['Uncracked', 'Cracked'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, 'confusion_matrix.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(visualization_dir, 'roc_curve.png'))
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(visualization_dir, 'pr_curve.png'))
        plt.close()

        print(f"\nIteration {iteration} Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1-score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"PR AUC: {metrics['pr_auc']:.4f}")

        return metrics

    def save_results_summary(self, training_time, evaluation_metrics, iteration):
        """Save a summary of results for an iteration"""
        iter_dir = os.path.join(self.results_dir, f'iteration_{iteration}')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = {
            'Date': timestamp,
            'Model': 'Nested Autoencoder',
            'Input Shape': str(self.input_shape),
            'Training Time': f"{training_time:.2f} seconds",
            'Iteration': iteration
        }
        summary.update(evaluation_metrics)
        pd.DataFrame([summary]).to_csv(os.path.join(iter_dir, 'summary.csv'), index=False)
        with open(os.path.join(iter_dir, 'summary.txt'), 'w') as f:
            f.write("CRACK DETECTION - ITERATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("\nResults and visualizations are saved in:\n")
            f.write(f"- Model: {os.path.join(iter_dir, 'model')}\n")
            f.write(f"- Visualizations: {os.path.join(iter_dir, 'visualizations')}\n")
            f.write(f"- Metrics: {os.path.join(iter_dir, 'metrics')}\n")

def process_surface_type(surface_type, base_data_dir, base_results_dir):
    """Process a single surface type (Deck, Pavement, Wall)"""
    cracked_dir = os.path.join(base_data_dir, surface_type, f'C{surface_type[0]}')
    uncracked_dir = os.path.join(base_data_dir, surface_type, f'U{surface_type[0]}')
    results_dir = os.path.join(base_results_dir, surface_type)

    print(f"\nProcessing {surface_type}...")
    with tf.device('/CPU:0'):
        model = NestedAutoencoder(
            input_shape=(128, 128, 3),
            results_dir=results_dir,
            cracked_dir=cracked_dir,
            uncracked_dir=uncracked_dir
        )
        train_images, test_images, test_labels, test_filenames = model.prepare_data()

    # Perform single iteration
    iteration = 1
    print(f"\nStarting {surface_type} Iteration {iteration}")
    start_time = time.time()

    # Train on all training images
    model.train(train_images, iteration, batch_size=32, epochs=50)

    # Evaluate on test set
    evaluation_metrics = model.evaluate(test_images, test_labels, test_filenames, iteration)

    # Save summary
    training_time = time.time() - start_time
    model.save_results_summary(training_time, evaluation_metrics, iteration)

    print(f"{surface_type} Iteration {iteration} completed in {training_time:.2f} seconds")
    print(f"\n{surface_type} processing completed. Results saved to {results_dir}")

def main():
    # Define directories
    base_data_dir = r"C:\Users\visha\Desktop\crack_project\SDNET2018"
    base_results_dir = "C:/Users/visha/Desktop/crack_project/Nested_SDENET2018"

    # Process each surface type
    surface_types = ['D', 'P', 'W']  # Deck, Pavement, Wall
    for surface_type in surface_types:
        process_surface_type(surface_type, base_data_dir, base_results_dir)

    print("\nAll surface types processed successfully.")

if __name__ == "__main__":
    main()
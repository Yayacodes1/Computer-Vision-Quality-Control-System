"""
Model Training Module for Quality Control System
Handles training, validation, and evaluation of CNN models
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import json
from datetime import datetime

from ..models.cnn_model import QualityControlCNN
from ..preprocessing.data_loader import DataLoader

class ModelTrainer:
    """
    Handles the complete training pipeline for the quality control system.
    
    This class provides methods for:
    - Training CNN models with proper validation
    - Monitoring training progress
    - Evaluating model performance
    - Saving training results and visualizations
    """
    
    def __init__(self, model: QualityControlCNN, data_loader: DataLoader, 
                 output_dir: str = "results"):
        """
        Initialize the model trainer.
        
        Args:
            model: CNN model instance
            data_loader: DataLoader instance
            output_dir: Directory to save training results
        """
        self.model = model
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.training_history = None
        self.best_model_path = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up TensorBoard logging
        self.tensorboard_dir = os.path.join(output_dir, "tensorboard")
        os.makedirs(self.tensorboard_dir, exist_ok=True)
    
    def train_model(self, 
                   train_data: Dict[str, np.ndarray],
                   val_data: Dict[str, np.ndarray],
                   epochs: int = 50,
                   batch_size: int = 32,
                   learning_rate: float = 0.001,
                   use_transfer_learning: bool = True,
                   fine_tune_epochs: int = 10) -> Dict[str, Any]:
        """
        Train the CNN model with comprehensive monitoring and evaluation.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            use_transfer_learning: Whether to use transfer learning approach
            fine_tune_epochs: Number of epochs for fine-tuning (if using transfer learning)
            
        Returns:
            Dictionary containing training results and metrics
        """
        print("Starting model training...")
        print(f"Training samples: {len(train_data['images'])}")
        print(f"Validation samples: {len(val_data['images'])}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Build model
        if use_transfer_learning:
            print("Using Transfer Learning approach with ResNet50...")
            model = self.model.build_transfer_learning_model('resnet50')
        else:
            print("Using Custom CNN architecture...")
            model = self.model.build_custom_cnn()
        
        # Create TensorBoard callback
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        # Create model checkpoint callback
        checkpoint_path = os.path.join(self.output_dir, "best_model.h5")
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        
        # Create early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Create learning rate reduction callback
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        callbacks = [tensorboard_callback, checkpoint_callback, early_stopping, lr_scheduler]
        
        # Train the model
        print("Starting training phase...")
        history = model.fit(
            train_data['images'], train_data['labels'],
            validation_data=(val_data['images'], val_data['labels']),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning phase (if using transfer learning)
        if use_transfer_learning and fine_tune_epochs > 0:
            print(f"\nStarting fine-tuning phase for {fine_tune_epochs} epochs...")
            
            # Unfreeze the base model
            base_model = model.layers[1]  # Assuming base model is the second layer
            base_model.trainable = True
            
            # Use lower learning rate for fine-tuning
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate/10),
                loss=model.loss,
                metrics=model.metrics_names
            )
            
            # Fine-tune the model
            fine_tune_history = model.fit(
                train_data['images'], train_data['labels'],
                validation_data=(val_data['images'], val_data['labels']),
                epochs=fine_tune_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Combine histories
            for key in history.history.keys():
                history.history[key].extend(fine_tune_history.history[key])
        
        self.training_history = history
        self.best_model_path = checkpoint_path
        
        # Save training results
        self._save_training_results(history)
        
        # Generate training visualizations
        self._plot_training_history(history)
        
        print("Training completed!")
        return {
            'history': history.history,
            'best_model_path': self.best_model_path,
            'final_accuracy': history.history['val_accuracy'][-1],
            'best_accuracy': max(history.history['val_accuracy'])
        }
    
    def evaluate_model(self, test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate the trained model on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.best_model_path is None:
            raise ValueError("No trained model found. Train the model first.")
        
        print("Evaluating model on test data...")
        
        # Load the best model
        model = keras.models.load_model(self.best_model_path)
        
        # Make predictions
        y_pred_proba = model.predict(test_data['images'])
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = test_data['labels']
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_true)
        precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        try:
            auc_score = roc_auc_score(y_true, y_pred_proba)
        except:
            auc_score = None
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=self.data_loader.class_names,
            output_dict=True
        )
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Save evaluation results
        evaluation_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'auc_score': float(auc_score) if auc_score else None,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        # Save to file
        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate evaluation visualizations
        self._plot_confusion_matrix(cm)
        self._plot_roc_curve(y_true, y_pred_proba)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1_score:.4f}")
        if auc_score:
            print(f"Test AUC: {auc_score:.4f}")
        
        return evaluation_results
    
    def _save_training_results(self, history: keras.callbacks.History) -> None:
        """Save training history and model information."""
        # Save training history
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2)
        
        # Save model information
        model_info = self.model.get_model_info()
        model_info_path = os.path.join(self.output_dir, "model_info.json")
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Training results saved to {self.output_dir}")
    
    def _plot_training_history(self, history: keras.callbacks.History) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot precision (if available)
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Plot recall (if available)
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confusion_matrix(self, cm: np.ndarray) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.data_loader.class_names,
                   yticklabels=self.data_loader.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.
        
        Returns:
            Dictionary containing training summary
        """
        if self.training_history is None:
            return {"error": "No training history available"}
        
        history = self.training_history.history
        
        return {
            "total_epochs": len(history['loss']),
            "final_training_accuracy": history['accuracy'][-1],
            "final_validation_accuracy": history['val_accuracy'][-1],
            "best_validation_accuracy": max(history['val_accuracy']),
            "final_training_loss": history['loss'][-1],
            "final_validation_loss": history['val_loss'][-1],
            "best_validation_loss": min(history['val_loss']),
            "model_path": self.best_model_path
        }

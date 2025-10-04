"""
CNN Model Architecture for Quality Control System
Implements custom CNN and transfer learning approaches using TensorFlow/Keras
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict, Any
import numpy as np

class QualityControlCNN:
    """
    Custom CNN architecture specifically designed for defect detection in quality control.
    
    This model uses a combination of:
    - Convolutional layers for feature extraction
    - Batch normalization for stable training
    - Dropout for regularization
    - Global average pooling to reduce overfitting
    - Dense layers for classification
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                 num_classes: int = 2, dropout_rate: float = 0.5):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_custom_cnn(self) -> Model:
        """
        Build a custom CNN architecture optimized for defect detection.
        
        Architecture Design Philosophy:
        - Start with small filters to detect fine details (defects)
        - Gradually increase filter size to capture larger patterns
        - Use batch normalization for stable training
        - Apply dropout to prevent overfitting
        - Use global average pooling instead of flatten to reduce parameters
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.Input(shape=self.input_shape, name='image_input')
        
        # First Convolutional Block - Detect fine details (scratches, cracks)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = layers.MaxPooling2D((2, 2), name='pool1')(x)
        x = layers.Dropout(0.25, name='dropout1')(x)
        
        # Second Convolutional Block - Detect textures and patterns
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4')(x)
        x = layers.MaxPooling2D((2, 2), name='pool2')(x)
        x = layers.Dropout(0.25, name='dropout2')(x)
        
        # Third Convolutional Block - Detect larger features and shapes
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6')(x)
        x = layers.MaxPooling2D((2, 2), name='pool3')(x)
        x = layers.Dropout(0.25, name='dropout3')(x)
        
        # Fourth Convolutional Block - High-level feature extraction
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7')(x)
        x = layers.BatchNormalization(name='bn4')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv8')(x)
        x = layers.MaxPooling2D((2, 2), name='pool4')(x)
        x = layers.Dropout(0.25, name='dropout4')(x)
        
        # Global Average Pooling instead of Flatten to reduce overfitting
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu', name='dense1')(x)
        x = layers.BatchNormalization(name='bn5')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout5')(x)
        
        x = layers.Dense(256, activation='relu', name='dense2')(x)
        x = layers.BatchNormalization(name='bn6')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout6')(x)
        
        # Output layer
        if self.num_classes == 2:
            # Binary classification
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            # Multi-class classification
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='QualityControlCNN')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name: str = 'resnet50') -> Model:
        """
        Build a model using transfer learning from pre-trained networks.
        
        Args:
            base_model_name: Name of pre-trained model ('resnet50', 'vgg16', 'inception_v3')
            
        Returns:
            Compiled Keras model with transfer learning
        """
        # Load pre-trained base model
        if base_model_name == 'resnet50':
            base_model = keras.applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'vgg16':
            base_model = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == 'inception_v3':
            base_model = keras.applications.InceptionV3(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=self.input_shape)
        
        # Preprocessing for the base model
        if base_model_name == 'inception_v3':
            # InceptionV3 expects inputs in [-1, 1] range
            x = keras.applications.inception_v3.preprocess_input(inputs)
        else:
            # ResNet50 and VGG16 expect inputs in [0, 255] range
            x = keras.applications.resnet50.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Custom classification head
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create model
        model = Model(inputs, outputs, name=f'TransferLearning_{base_model_name}')
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        self.model = model
        return model
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the model architecture.
        
        Returns:
            String representation of model summary
        """
        if self.model is None:
            return "Model not built yet. Call build_custom_cnn() or build_transfer_learning_model() first."
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "Model not built yet"}
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        return {
            "model_name": self.model.name,
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "optimizer": str(self.model.optimizer),
            "loss_function": self.model.loss,
            "metrics": [str(metric) for metric in self.model.metrics_names]
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Build model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, images: np.ndarray) -> np.ndarray:
        """
        Make predictions on new images.
        
        Args:
            images: Batch of preprocessed images
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not built yet. Build model first.")
        
        return self.model.predict(images)
    
    def predict_single(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Make prediction on a single image.
        
        Args:
            image: Single preprocessed image
            
        Returns:
            Tuple of (probability, class_label)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Build model first.")
        
        # Add batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch)[0]
        
        if self.num_classes == 2:
            # Binary classification
            probability = prediction[0] if isinstance(prediction, np.ndarray) else prediction
            class_label = "Defective" if probability > 0.5 else "Good"
            return probability, class_label
        else:
            # Multi-class classification
            class_idx = np.argmax(prediction)
            probability = prediction[class_idx]
            class_label = f"Class_{class_idx}"
            return probability, class_label

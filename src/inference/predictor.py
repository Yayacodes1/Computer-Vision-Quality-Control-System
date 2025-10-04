"""
Inference Pipeline for Quality Control System
Handles real-time and batch predictions using trained models
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.cnn_model import QualityControlCNN
from ..preprocessing.image_processor import ImageProcessor

class QualityPredictor:
    """
    Handles inference and predictions for the quality control system.
    
    This class provides methods for:
    - Loading trained models
    - Making predictions on single images
    - Batch processing of multiple images
    - Real-time quality assessment
    - Result visualization and reporting
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the quality predictor.
        
        Args:
            model_path: Path to the trained model
            confidence_threshold: Threshold for defect classification
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.image_processor = ImageProcessor()
        self.class_names = ['Good', 'Defective']
        
        # Load the trained model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from file."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def predict_single_image(self, image_path: str, 
                           return_confidence: bool = True) -> Dict[str, Union[str, float]]:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        image = self.image_processor.load_image(image_path)
        if image is None:
            return {"error": "Failed to load image"}
        
        # Apply preprocessing (no augmentation for inference)
        processed_image = self.image_processor.augment_image(image, is_training=False)
        
        # Add batch dimension
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch, verbose=0)[0]
        
        # Process results
        if isinstance(prediction, np.ndarray):
            defect_probability = float(prediction[0])
        else:
            defect_probability = float(prediction)
        
        is_defective = defect_probability > self.confidence_threshold
        quality_label = "Defective" if is_defective else "Good"
        pass_fail = "FAIL" if is_defective else "PASS"
        
        result = {
            "image_path": image_path,
            "quality_label": quality_label,
            "pass_fail": pass_fail,
            "defect_probability": defect_probability,
            "confidence": defect_probability if is_defective else (1 - defect_probability)
        }
        
        if not return_confidence:
            result.pop("defect_probability")
            result.pop("confidence")
        
        return result
    
    def predict_batch(self, image_paths: List[str], 
                     batch_size: int = 32) -> List[Dict[str, Union[str, float]]]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and preprocess batch
            for path in batch_paths:
                image = self.image_processor.load_image(path)
                if image is not None:
                    processed_image = self.image_processor.augment_image(image, is_training=False)
                    batch_images.append(processed_image)
                    valid_paths.append(path)
            
            if not batch_images:
                continue
            
            # Convert to batch
            batch_array = np.stack(batch_images)
            
            # Make predictions
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Process results
            for j, (path, prediction) in enumerate(zip(valid_paths, predictions)):
                if isinstance(prediction, np.ndarray):
                    defect_probability = float(prediction[0])
                else:
                    defect_probability = float(prediction)
                
                is_defective = defect_probability > self.confidence_threshold
                quality_label = "Defective" if is_defective else "Good"
                pass_fail = "FAIL" if is_defective else "PASS"
                
                result = {
                    "image_path": path,
                    "quality_label": quality_label,
                    "pass_fail": pass_fail,
                    "defect_probability": defect_probability,
                    "confidence": defect_probability if is_defective else (1 - defect_probability)
                }
                
                results.append(result)
        
        return results
    
    def predict_from_array(self, image_array: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Make prediction from numpy array (for real-time processing).
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess image
        processed_image = self.image_processor.augment_image(image_array, is_training=False)
        
        # Add batch dimension
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image_batch, verbose=0)[0]
        
        # Process results
        if isinstance(prediction, np.ndarray):
            defect_probability = float(prediction[0])
        else:
            defect_probability = float(prediction)
        
        is_defective = defect_probability > self.confidence_threshold
        quality_label = "Defective" if is_defective else "Good"
        pass_fail = "FAIL" if is_defective else "PASS"
        
        return {
            "quality_label": quality_label,
            "pass_fail": pass_fail,
            "defect_probability": defect_probability,
            "confidence": defect_probability if is_defective else (1 - defect_probability)
        }
    
    def visualize_prediction(self, image_path: str, 
                           save_path: Optional[str] = None) -> None:
        """
        Visualize prediction result on the image.
        
        Args:
            image_path: Path to the image
            save_path: Optional path to save the visualization
        """
        # Get prediction
        result = self.predict_single_image(image_path)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        # Load original image
        image = self.image_processor.load_image(image_path)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f'Original Image')
        axes[0].axis('off')
        
        # Image with prediction overlay
        axes[1].imshow(image)
        
        # Add prediction text
        prediction_text = f"{result['quality_label']}\n"
        prediction_text += f"Confidence: {result['confidence']:.3f}\n"
        prediction_text += f"Status: {result['pass_fail']}"
        
        # Color based on prediction
        color = 'red' if result['pass_fail'] == 'FAIL' else 'green'
        
        axes[1].text(10, 30, prediction_text, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                    fontsize=12, color='white', weight='bold')
        
        axes[1].set_title('Quality Assessment')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print prediction details
        print(f"\nPrediction Results:")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Quality: {result['quality_label']}")
        print(f"Status: {result['pass_fail']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Defect Probability: {result['defect_probability']:.3f}")
    
    def generate_quality_report(self, image_paths: List[str], 
                              output_path: str = "quality_report.txt") -> Dict[str, any]:
        """
        Generate a comprehensive quality report for multiple images.
        
        Args:
            image_paths: List of image paths to analyze
            output_path: Path to save the report
            
        Returns:
            Dictionary containing report summary
        """
        print(f"Generating quality report for {len(image_paths)} images...")
        
        # Get predictions
        results = self.predict_batch(image_paths)
        
        # Calculate statistics
        total_images = len(results)
        good_count = sum(1 for r in results if r['quality_label'] == 'Good')
        defective_count = total_images - good_count
        
        good_rate = good_count / total_images if total_images > 0 else 0
        defect_rate = defective_count / total_images if total_images > 0 else 0
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        # Generate report
        report_content = f"""
QUALITY CONTROL REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {os.path.basename(self.model_path)}

SUMMARY STATISTICS:
- Total Images Analyzed: {total_images}
- Good Products: {good_count} ({good_rate:.1%})
- Defective Products: {defective_count} ({defect_rate:.1%})
- Average Confidence: {avg_confidence:.3f}

DETAILED RESULTS:
"""
        
        for i, result in enumerate(results, 1):
            report_content += f"\n{i}. {os.path.basename(result['image_path'])}"
            report_content += f" - {result['quality_label']} ({result['pass_fail']})"
            report_content += f" - Confidence: {result['confidence']:.3f}"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        print(f"Quality report saved to {output_path}")
        
        # Return summary
        return {
            'total_images': total_images,
            'good_count': good_count,
            'defective_count': defective_count,
            'good_rate': good_rate,
            'defect_rate': defect_rate,
            'avg_confidence': avg_confidence,
            'report_path': output_path
        }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold for classification.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"Confidence threshold updated to {threshold}")
        else:
            print("Error: Threshold must be between 0.0 and 1.0")
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_path": self.model_path,
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "total_parameters": self.model.count_params(),
            "confidence_threshold": self.confidence_threshold
        }

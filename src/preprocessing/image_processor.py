"""
Image Preprocessing Module for Quality Control System
Handles image resizing, normalization, augmentation using OpenCV
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import albumentations as A
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    Handles all image preprocessing operations for the quality control system.
    
    This class provides methods for:
    - Image resizing and standardization
    - Normalization for CNN input
    - Data augmentation for training robustness
    - Converting between different image formats
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the image processor.
        
        Args:
            target_size: Target size for image resizing (height, width)
        """
        self.target_size = target_size
        
        # Define augmentation pipeline for training
        self.train_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomRotate90(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Define pipeline for validation/test (no augmentation)
        self.val_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image as numpy array
            size: Target size (height, width)
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (size[1], size[0]))  # cv2.resize takes (width, height)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    def augment_image(self, image: np.ndarray, is_training: bool = True) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image as numpy array
            is_training: Whether to apply training augmentations
            
        Returns:
            Augmented image
        """
        if is_training:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
        
        return transformed['image']
    
    def preprocess_batch(self, images: List[np.ndarray], is_training: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays
            is_training: Whether to apply training augmentations
            
        Returns:
            Preprocessed batch as numpy array
        """
        processed_images = []
        
        for image in images:
            processed = self.augment_image(image, is_training)
            processed_images.append(processed)
        
        return np.stack(processed_images)
    
    def visualize_preprocessing(self, original_image: np.ndarray, 
                              processed_image: np.ndarray, 
                              save_path: Optional[str] = None) -> None:
        """
        Visualize original vs processed image.
        
        Args:
            original_image: Original image
            processed_image: Processed image
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Denormalize for visualization
        if hasattr(processed_image, 'numpy'):
            processed_np = processed_image.numpy()
        else:
            processed_np = processed_image
            
        if processed_np.max() <= 1.0:
            processed_vis = processed_np * 255
        else:
            processed_vis = processed_np
            
        # Handle tensor format (C, H, W) -> (H, W, C)
        if len(processed_vis.shape) == 3 and processed_vis.shape[0] == 3:
            processed_vis = np.transpose(processed_vis, (1, 2, 0))
            
        processed_vis = np.clip(processed_vis, 0, 255).astype(np.uint8)
        
        axes[1].imshow(processed_vis)
        axes[1].set_title('Processed Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def create_augmentation_examples(self, image: np.ndarray, 
                                   num_examples: int = 4) -> List[np.ndarray]:
        """
        Create multiple augmented versions of an image for demonstration.
        
        Args:
            image: Input image
            num_examples: Number of augmented examples to create
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for _ in range(num_examples):
            augmented = self.augment_image(image, is_training=True)
            augmented_images.append(augmented)
        
        return augmented_images
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get basic statistics about an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        stats = {
            'shape': image.shape,
            'dtype': image.dtype,
            'min_value': image.min(),
            'max_value': image.max(),
            'mean_value': image.mean(),
            'std_value': image.std()
        }
        
        return stats

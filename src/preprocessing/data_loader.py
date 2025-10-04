"""
Data Loading Module for Quality Control System
Handles dataset creation, splitting, and batch generation
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from .image_processor import ImageProcessor

class DataLoader:
    """
    Handles dataset loading, preprocessing, and batch generation for the quality control system.
    
    This class provides methods for:
    - Loading images from directories
    - Creating train/validation/test splits
    - Generating batches for training
    - Dataset visualization and analysis
    """
    
    def __init__(self, data_dir: str, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the main data directory
            target_size: Target size for image resizing
        """
        self.data_dir = data_dir
        self.target_size = target_size
        self.image_processor = ImageProcessor(target_size)
        self.class_names = []
        self.class_to_idx = {}
        self.dataset_info = {}
        
    def load_dataset_from_directory(self, 
                                  good_dir: str, 
                                  defective_dir: str,
                                  test_size: float = 0.2,
                                  val_size: float = 0.2,
                                  random_state: int = 42) -> Dict[str, List]:
        """
        Load dataset from separate directories for good and defective products.
        
        Args:
            good_dir: Directory containing good product images
            defective_dir: Directory containing defective product images
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/val/test splits with images and labels
        """
        print("Loading dataset from directories...")
        
        # Load good product images
        good_images = self._load_images_from_directory(good_dir, label=0)
        print(f"Loaded {len(good_images)} good product images")
        
        # Load defective product images
        defective_images = self._load_images_from_directory(defective_dir, label=1)
        print(f"Loaded {len(defective_images)} defective product images")
        
        # Combine all data
        all_images = good_images + defective_images
        all_labels = [0] * len(good_images) + [1] * len(defective_images)
        
        # Set class information
        self.class_names = ['Good', 'Defective']
        self.class_to_idx = {'Good': 0, 'Defective': 1}
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            all_images, all_labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=all_labels
        )
        
        # Create train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train
        )
        
        # Store dataset information
        self.dataset_info = {
            'total_samples': len(all_images),
            'good_samples': len(good_images),
            'defective_samples': len(defective_images),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'class_distribution': {
                'Good': len(good_images),
                'Defective': len(defective_images)
            }
        }
        
        print(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return {
            'train': {'images': X_train, 'labels': y_train},
            'val': {'images': X_val, 'labels': y_val},
            'test': {'images': X_test, 'labels': y_test}
        }
    
    def _load_images_from_directory(self, directory: str, label: int) -> List[str]:
        """
        Load image paths from a directory.
        
        Args:
            directory: Directory containing images
            label: Label for images in this directory
            
        Returns:
            List of image file paths
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")
        
        for filename in os.listdir(directory):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(directory, filename))
        
        return image_paths
    
    def preprocess_dataset(self, dataset: Dict[str, Dict], 
                          is_training: bool = True) -> Dict[str, np.ndarray]:
        """
        Preprocess the entire dataset.
        
        Args:
            dataset: Dataset dictionary with images and labels
            is_training: Whether to apply training augmentations
            
        Returns:
            Dictionary with preprocessed images and labels
        """
        print(f"Preprocessing dataset (training={is_training})...")
        
        processed_dataset = {}
        
        for split_name, split_data in dataset.items():
            print(f"Processing {split_name} split...")
            
            images = split_data['images']
            labels = split_data['labels']
            
            # Process images
            processed_images = []
            for i, image_path in enumerate(images):
                if i % 100 == 0:
                    print(f"Processing image {i+1}/{len(images)}")
                
                # Load image
                image = self.image_processor.load_image(image_path)
                if image is not None:
                    # Apply preprocessing
                    processed_image = self.image_processor.augment_image(image, is_training)
                    processed_images.append(processed_image)
                else:
                    # Remove corresponding label if image failed to load
                    labels.pop(len(processed_images))
            
            # Convert to numpy arrays
            processed_dataset[split_name] = {
                'images': np.array(processed_images),
                'labels': np.array(labels)
            }
        
        print("Dataset preprocessing completed!")
        return processed_dataset
    
    def create_data_generators(self, dataset: Dict[str, Dict], 
                             batch_size: int = 32) -> Dict[str, 'tf.data.Dataset']:
        """
        Create TensorFlow data generators for efficient batch processing.
        
        Args:
            dataset: Preprocessed dataset
            batch_size: Size of each batch
            
        Returns:
            Dictionary of TensorFlow datasets for each split
        """
        import tensorflow as tf
        
        data_generators = {}
        
        for split_name, split_data in dataset.items():
            images = split_data['images']
            labels = split_data['labels']
            
            # Create TensorFlow dataset
            tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            
            # Configure dataset for performance
            tf_dataset = tf_dataset.batch(batch_size)
            tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
            
            data_generators[split_name] = tf_dataset
        
        return data_generators
    
    def visualize_dataset_distribution(self, dataset: Dict[str, Dict]) -> None:
        """
        Visualize the distribution of classes in the dataset.
        
        Args:
            dataset: Dataset dictionary
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (split_name, split_data) in enumerate(dataset.items()):
            labels = split_data['labels']
            
            # Count class distribution
            unique, counts = np.unique(labels, return_counts=True)
            class_counts = dict(zip([self.class_names[idx] for idx in unique], counts))
            
            # Create bar plot
            axes[i].bar(class_counts.keys(), class_counts.values())
            axes[i].set_title(f'{split_name.title()} Set Distribution')
            axes[i].set_ylabel('Number of Samples')
            axes[i].set_xlabel('Class')
            
            # Add count labels on bars
            for j, v in enumerate(class_counts.values()):
                axes[i].text(j, v + 0.1, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nDataset Summary:")
        print("-" * 50)
        for split_name, split_data in dataset.items():
            labels = split_data['labels']
            good_count = np.sum(labels == 0)
            defective_count = np.sum(labels == 1)
            total = len(labels)
            
            print(f"{split_name.title()} Set:")
            print(f"  Total samples: {total}")
            print(f"  Good products: {good_count} ({good_count/total*100:.1f}%)")
            print(f"  Defective products: {defective_count} ({defective_count/total*100:.1f}%)")
            print()
    
    def visualize_sample_images(self, dataset: Dict[str, Dict], 
                              num_samples: int = 8) -> None:
        """
        Visualize sample images from the dataset.
        
        Args:
            dataset: Dataset dictionary
            num_samples: Number of samples to visualize per class
        """
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
        
        for class_idx, class_name in enumerate(self.class_names):
            # Find images of this class
            class_images = []
            for split_data in dataset.values():
                labels = split_data['labels']
                images = split_data['images']
                class_mask = labels == class_idx
                class_images.extend(images[class_mask][:num_samples])
            
            # Display images
            for i in range(min(num_samples, len(class_images))):
                image = class_images[i]
                
                # Handle tensor format
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                
                # Denormalize if needed
                if image.max() <= 1.0:
                    image = image * 255
                
                # Convert to uint8
                image = np.clip(image, 0, 255).astype(np.uint8)
                
                # Handle channel dimension
                if len(image.shape) == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                
                axes[class_idx, i].imshow(image)
                axes[class_idx, i].set_title(f'{class_name}')
                axes[class_idx, i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'dataset_info': self.dataset_info,
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'target_size': self.target_size
        }

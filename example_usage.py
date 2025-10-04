#!/usr/bin/env python3
"""
Example Usage of Computer Vision Quality Control System
This script demonstrates how to use each component of the system
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append('src')

from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.data_loader import DataLoader
from src.models.cnn_model import QualityControlCNN
from src.inference.predictor import QualityPredictor

def example_image_preprocessing():
    """Demonstrate image preprocessing capabilities."""
    print("=" * 60)
    print("IMAGE PREPROCESSING EXAMPLE")
    print("=" * 60)
    
    # Initialize image processor
    processor = ImageProcessor(target_size=(224, 224))
    
    # Create a sample image (simulating a product image)
    print("Creating sample product image...")
    sample_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Get image statistics
    stats = processor.get_image_stats(sample_image)
    print("Original image statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate preprocessing
    print("\nApplying preprocessing...")
    processed_image = processor.augment_image(sample_image, is_training=False)
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Processed image type: {type(processed_image)}")
    
    # Create augmentation examples
    print("\nCreating augmentation examples...")
    augmented_images = processor.create_augmentation_examples(sample_image, num_examples=3)
    print(f"Created {len(augmented_images)} augmented examples")
    
    return processor

def example_cnn_model():
    """Demonstrate CNN model creation."""
    print("=" * 60)
    print("CNN MODEL EXAMPLE")
    print("=" * 60)
    
    # Initialize CNN model
    model = QualityControlCNN(input_shape=(224, 224, 3), num_classes=2)
    
    # Build custom CNN
    print("Building custom CNN...")
    custom_model = model.build_custom_cnn()
    print("Custom CNN built successfully!")
    
    # Get model information
    model_info = model.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Build transfer learning model
    print("\nBuilding transfer learning model...")
    transfer_model = model.build_transfer_learning_model('resnet50')
    print("Transfer learning model built successfully!")
    
    # Get model summary
    print("\nTransfer Learning Model Summary:")
    summary = model.get_model_summary()
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    
    return model

def example_data_loading():
    """Demonstrate data loading capabilities."""
    print("=" * 60)
    print("DATA LOADING EXAMPLE")
    print("=" * 60)
    
    # Create sample dataset structure
    os.makedirs("data/raw/good", exist_ok=True)
    os.makedirs("data/raw/defective", exist_ok=True)
    
    # Initialize data loader
    data_loader = DataLoader("data/raw")
    
    print("Data loader initialized!")
    print(f"Target size: {data_loader.target_size}")
    
    # Note: In real usage, you would have actual images
    print("\nNote: In a real implementation, you would:")
    print("1. Place good product images in data/raw/good/")
    print("2. Place defective product images in data/raw/defective/")
    print("3. Call data_loader.load_dataset_from_directory()")
    
    return data_loader

def example_inference():
    """Demonstrate inference capabilities (with mock model)."""
    print("=" * 60)
    print("INFERENCE EXAMPLE")
    print("=" * 60)
    
    print("Note: This example shows inference structure.")
    print("In real usage, you would:")
    print("1. Train a model first using the training pipeline")
    print("2. Save the trained model")
    print("3. Load it for inference")
    
    # Example of how inference would work
    print("\nExample inference workflow:")
    print("predictor = QualityPredictor('path/to/trained/model.h5')")
    print("result = predictor.predict_single_image('product.jpg')")
    print("print(f'Quality: {result[\"quality_label\"]}')")
    print("print(f'Confidence: {result[\"confidence\"]:.3f}')")

def main():
    """Run all examples."""
    print("Computer Vision Quality Control System - Examples")
    print("=" * 60)
    print("This script demonstrates the key components of the system.")
    print("Each component is designed to be modular and reusable.")
    print("=" * 60)
    
    try:
        # Run examples
        processor = example_image_preprocessing()
        model = example_cnn_model()
        data_loader = example_data_loading()
        example_inference()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\nNext steps:")
        print("1. Add your product images to data/raw/good/ and data/raw/defective/")
        print("2. Run: python main.py --mode train --epochs 50")
        print("3. Run: python main.py --mode inference --images path/to/test/image.jpg")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()

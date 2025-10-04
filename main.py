#!/usr/bin/env python3
"""
Computer Vision Quality Control System - Main Script
Demonstrates the complete pipeline from data loading to inference
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List

# Add src to path for imports
sys.path.append('src')

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.image_processor import ImageProcessor
from src.models.cnn_model import QualityControlCNN
from src.training.trainer import ModelTrainer
from src.inference.predictor import QualityPredictor

def create_sample_dataset():
    """
    Create a sample dataset structure for demonstration.
    This would typically be replaced with actual product images.
    """
    print("Creating sample dataset structure...")
    
    # Create sample directories
    os.makedirs("data/raw/good", exist_ok=True)
    os.makedirs("data/raw/defective", exist_ok=True)
    
    # Create placeholder files (in real scenario, these would be actual images)
    print("Note: In a real implementation, you would place actual product images in:")
    print("  - data/raw/good/ (for good quality products)")
    print("  - data/raw/defective/ (for defective products)")
    
    return True

def train_model(data_dir: str = "data/raw", epochs: int = 10):
    """
    Train the quality control model.
    
    Args:
        data_dir: Directory containing training data
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    
    # Initialize components
    data_loader = DataLoader(data_dir)
    model = QualityControlCNN()
    
    # Load dataset
    print("Loading dataset...")
    dataset = data_loader.load_dataset_from_directory(
        good_dir=os.path.join(data_dir, "good"),
        defective_dir=os.path.join(data_dir, "defective"),
        test_size=0.2,
        val_size=0.2
    )
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    processed_dataset = data_loader.preprocess_dataset(dataset, is_training=True)
    
    # Initialize trainer
    trainer = ModelTrainer(model, data_loader, output_dir="results")
    
    # Train model
    print("Starting model training...")
    training_results = trainer.train_model(
        train_data=processed_dataset['train'],
        val_data=processed_dataset['val'],
        epochs=epochs,
        batch_size=32,
        use_transfer_learning=True
    )
    
    print(f"Training completed! Best accuracy: {training_results['best_accuracy']:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    evaluation_results = trainer.evaluate_model(processed_dataset['test'])
    
    print(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    
    return training_results, evaluation_results

def run_inference(model_path: str, image_paths: List[str]):
    """
    Run inference on images using the trained model.
    
    Args:
        model_path: Path to the trained model
        image_paths: List of image paths to analyze
    """
    print("=" * 60)
    print("INFERENCE PHASE")
    print("=" * 60)
    
    # Initialize predictor
    predictor = QualityPredictor(model_path)
    
    # Make predictions
    print(f"Analyzing {len(image_paths)} images...")
    results = predictor.predict_batch(image_paths)
    
    # Display results
    print("\nPrediction Results:")
    print("-" * 50)
    for result in results:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"  Quality: {result['quality_label']}")
        print(f"  Status: {result['pass_fail']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print()
    
    # Generate quality report
    report = predictor.generate_quality_report(image_paths, "quality_report.txt")
    
    print(f"Quality Report Summary:")
    print(f"  Total Images: {report['total_images']}")
    print(f"  Good Rate: {report['good_rate']:.1%}")
    print(f"  Defect Rate: {report['defect_rate']:.1%}")
    print(f"  Average Confidence: {report['avg_confidence']:.3f}")
    
    return results

def demonstrate_preprocessing():
    """
    Demonstrate image preprocessing capabilities.
    """
    print("=" * 60)
    print("PREPROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize image processor
    processor = ImageProcessor()
    
    # Create a sample image (in real scenario, this would be a loaded image)
    sample_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    print("Original image shape:", sample_image.shape)
    print("Original image stats:")
    stats = processor.get_image_stats(sample_image)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Demonstrate augmentation
    print("\nCreating augmented examples...")
    augmented_images = processor.create_augmentation_examples(sample_image, num_examples=4)
    
    print(f"Created {len(augmented_images)} augmented examples")
    for i, aug_img in enumerate(augmented_images):
        print(f"  Example {i+1} shape: {aug_img.shape}")

def main():
    """Main function to run the quality control system."""
    parser = argparse.ArgumentParser(description="Computer Vision Quality Control System")
    parser.add_argument("--mode", choices=["train", "inference", "demo", "preprocess"], 
                       default="demo", help="Mode to run the system")
    parser.add_argument("--data-dir", default="data/raw", 
                       help="Directory containing training data")
    parser.add_argument("--model-path", default="results/best_model.h5",
                       help="Path to trained model for inference")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--images", nargs="+", 
                       help="Image paths for inference")
    
    args = parser.parse_args()
    
    print("Computer Vision Quality Control System")
    print("=" * 60)
    print("Technologies: Python, OpenCV, TensorFlow, NumPy")
    print("Target Accuracy: 85%+ for defect detection")
    print("=" * 60)
    
    if args.mode == "demo":
        # Run demonstration
        print("Running system demonstration...")
        
        # Create sample dataset structure
        create_sample_dataset()
        
        # Demonstrate preprocessing
        demonstrate_preprocessing()
        
        print("\nDemo completed! To train a real model:")
        print("1. Place your product images in data/raw/good/ and data/raw/defective/")
        print("2. Run: python main.py --mode train --epochs 50")
        print("3. Run: python main.py --mode inference --images path/to/image1.jpg path/to/image2.jpg")
        
    elif args.mode == "preprocess":
        # Demonstrate preprocessing only
        demonstrate_preprocessing()
        
    elif args.mode == "train":
        # Train the model
        if not os.path.exists(args.data_dir):
            print(f"Error: Data directory {args.data_dir} does not exist")
            print("Please create the directory structure and add your images:")
            print("  data/raw/good/ (good product images)")
            print("  data/raw/defective/ (defective product images)")
            return
        
        train_model(args.data_dir, args.epochs)
        
    elif args.mode == "inference":
        # Run inference
        if not os.path.exists(args.model_path):
            print(f"Error: Model file {args.model_path} does not exist")
            print("Please train a model first using --mode train")
            return
        
        if not args.images:
            print("Error: Please provide image paths using --images")
            return
        
        run_inference(args.model_path, args.images)

if __name__ == "__main__":
    main()

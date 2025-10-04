# Computer Vision Quality Control System
# Main package initialization

__version__ = "1.0.0"
__author__ = "Quality Control Team"

# Import main components for easy access
from .preprocessing.image_processor import ImageProcessor
from .models.cnn_model import QualityControlCNN

__all__ = [
    'ImageProcessor',
    'QualityControlCNN'
]

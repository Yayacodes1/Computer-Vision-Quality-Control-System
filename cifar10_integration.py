import pickle
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class CIFAR10Integrator:
    def __init__(self, cifar_dir="cifar-10-batches-py", output_dir="data/cifar10_animals"):
        self.cifar_dir = cifar_dir
        self.output_dir = output_dir
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "good"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "defective"), exist_ok=True)
    
    def load_cifar10_data(self):
        """Load all CIFAR-10 training data"""
        print("Loading CIFAR-10 dataset...")
        
        all_data = []
        all_labels = []
        
        for i in range(1, 6):  # 5 training batches
            filename = os.path.join(self.cifar_dir, f'data_batch_{i}')
            with open(filename, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                all_data.append(batch[b'data'])
                all_labels.extend(batch[b'labels'])
        
        data = np.vstack(all_data)
        labels = np.array(all_labels)
        
        print(f"✅ Loaded {len(data)} images")
        print(f"✅ Class distribution: {np.bincount(labels)}")
        return data, labels
    
    def extract_bird_images(self, data, labels):
        """Extract bird images (class 2) from CIFAR-10"""
        print("Extracting bird images...")
        
        bird_indices = np.where(labels == 2)[0]
        bird_data = data[bird_indices]
        bird_labels = labels[bird_indices]
        
        print(f"✅ Found {len(bird_data)} bird images")
        return bird_data, bird_labels
    
    def create_quality_dataset(self, bird_data):
        """Create good/defective quality control dataset"""
        print("Creating quality control dataset...")
        
        good_count = 0
        defect_count = 0
        
        for i, img_data in enumerate(bird_data):
            # Reshape and resize image
            img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
            img = cv2.resize(img, (224, 224))
            
            if i < len(bird_data) // 2:
                # Save as good quality
                filename = f"bird_good_{good_count+1:03d}.jpg"
                cv2.imwrite(os.path.join(self.output_dir, "good", filename), img)
                good_count += 1
            else:
                # Add defects and save as defective
                defective_img = self.add_defects(img)
                filename = f"bird_defect_{defect_count+1:03d}.jpg"
                cv2.imwrite(os.path.join(self.output_dir, "defective", filename), defective_img)
                defect_count += 1
        
        print(f"✅ Created {good_count} good images")
        print(f"✅ Created {defect_count} defective images")
        return good_count, defect_count
    
    def add_defects(self, img):
        """Add realistic defects to simulate quality issues"""
        defective_img = img.copy()
        
        # Add random noise
        noise = np.random.normal(0, 30, defective_img.shape).astype(np.uint8)
        defective_img = cv2.add(defective_img, noise)
        
        # Add slight blur
        defective_img = cv2.GaussianBlur(defective_img, (3, 3), 0)
        
        return defective_img
    
    def run_integration(self):
        """Run complete integration process"""
        print("🚀 CIFAR-10 INTEGRATION FOR QUALITY CONTROL SYSTEM")
        print("=" * 60)
        print("Technologies: Python, OpenCV, TensorFlow, NumPy")
        print("Dataset: Real CIFAR-10 Bird Images")
        print("=" * 60)
        
        # Load data
        data, labels = self.load_cifar10_data()
        
        # Extract birds
        bird_data, bird_labels = self.extract_bird_images(data, labels)
        
        # Create quality dataset
        good_count, defect_count = self.create_quality_dataset(bird_data)
        
        print(f"\n🎯 INTEGRATION COMPLETE!")
        print(f"📁 Dataset saved to: {self.output_dir}")
        print(f"📊 Total images: {good_count + defect_count}")
        print(f"🐔 Bird images processed: {len(bird_data)}")
        print(f"✅ Good quality: {good_count}")
        print(f"❌ Defective quality: {defect_count}")
        
        return good_count, defect_count

if __name__ == "__main__":
    integrator = CIFAR10Integrator()
    integrator.run_integration()

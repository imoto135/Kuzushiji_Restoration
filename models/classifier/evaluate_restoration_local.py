import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import argparse
from tqdm import tqdm
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageListDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('L').convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            logging.error(f"Error loading {img_path}: {e}")
            # Return a dummy tensor or handle better. For now, let's just crash or skip appropriately in collate if we were fancy.
            # Simpler: return a zero tensor and handle in loop? 
            # Or just raise and let the user know data issues.
            raise e

def evaluate(model_path, image_dir, device, batch_size=32):
    # Load model and classes
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    
    model = models.resnet18(weights=None) # Structure only
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logging.info(f"Loaded model from {model_path} with {num_classes} classes.")

    # Transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Find images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_paths:
        logging.warning(f"No images found in {image_dir}")
        return

    # Create DataLoader
    dataset = ImageListDataset(image_paths, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    correct = 0
    total = 0
    results = []

    logging.info(f"Evaluating {len(image_paths)} images...")
    
    with torch.no_grad():
        for inputs, paths in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(inputs.size(0)):
                pred_label = classes[preds[i].item()]
                img_path = paths[i]
                filename = os.path.basename(img_path)
                
                # Ground truth inference
                gt_label = None
                try:
                    gt_label = filename.split('_')[0]
                    if gt_label in classes:
                        total += 1
                        if gt_label == pred_label:
                            correct += 1
                except IndexError:
                    pass
                
                results.append({
                    'image': filename,
                    'prediction': pred_label,
                    'ground_truth': gt_label,
                    'correct': (gt_label == pred_label) if gt_label else None
                })
    
    if total > 0:
        accuracy = 100 * correct / total
        logging.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    else:
        logging.info("Could not infer ground truth for any images (or no images found).")

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Restored Images with Local Classifier')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images to evaluate')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Set warnings to ignore for cleaner output if desired, or handle the weights_only warning
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    evaluate(args.model_path, args.image_dir, device, args.batch_size)

if __name__ == "__main__":
    main()

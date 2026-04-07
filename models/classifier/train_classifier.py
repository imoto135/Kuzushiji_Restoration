import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
from tqdm import tqdm
import glob
import argparse
import copy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KuzushijiFlatDataset(Dataset):
    def __init__(self, root_dir, transform=None, classes=None):
        """
        Args:
            root_dir (string): Directory with all the images (flat structure).
            transform (callable, optional): Optional transform to be applied on a sample.
            classes (list, optional): List of class names. If None, inferred from files.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                           glob.glob(os.path.join(root_dir, "*.png"))
        
        # Parse classes from filenames
        # Assumes filename format: U+3042_xxxx.jpg
        self.samples = []
        found_classes = set()
        
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            # Split by underscore to get the class part (e.g., U+3042 from U+3042_200021743-00007_2_X0351_Y1327.jpg)
            try:
                class_name = filename.split('_')[0]
                found_classes.add(class_name)
                self.samples.append((img_path, class_name))
            except IndexError:
                logging.warning(f"Could not parse class from filename: {filename}. Skipping.")
        
        if classes is None:
            self.classes = sorted(list(found_classes))
        else:
            self.classes = sorted(list(classes))
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Filter samples to only include known classes (if classes was provided)
        self.samples = [s for s in self.samples if s[1] in self.class_to_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        try:
            image = Image.open(img_path).convert('L') # Convert to grayscale
            # Convert grayscale to RGB for ResNet
            image = image.convert('RGB')
            
            label = self.class_to_idx[class_name]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a dummy image or handle appropriately. 
            # For simplicity, returning the next valid item or raising error.
            # Here we just raise for now to be loud about data issues.
            raise e

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} epoch {epoch}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Step scheduler at end of epoch if it's the right type, or after validation metric
            if phase == 'train' and scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val':
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping triggered after {epochs_no_improve} epochs without improvement')
            break

        logging.info('')

    logging.info(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Kuzushiji Classifier on Local Dataset')
    parser.add_argument('--data_dir', type=str, default='./data/full_padded/gt', help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default upgraded to 50)')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--output_dir', type=str, default='./models/classifier/output_full_padded', help='Directory to save the model')
    parser.add_argument('--gpu', type=int, default=1, help='GPU index to use')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)), # ResNet expects 224x224
            # transforms.RandomHorizontalFlip(p=0.5), # Maybe not good for characters? Let's check. 
            # Hiragana characters are directional, so HorizontalFlip might be BAD. 
            # Let's use rotation and translation instead which are more realistic for handwriting.
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Locate datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    # If val doesn't exist, we might need to split train, but user said struct is train/val/test
    
    if not os.path.exists(train_dir):
        logging.error(f"Train directory not found: {train_dir}")
        return

    # Create datasets
    # We need to ensure classes are consistent across datasets
    logging.info("Scanning training data to determine classes...")
    train_dataset = KuzushijiFlatDataset(train_dir, transform=data_transforms['train'])
    classes = train_dataset.classes
    logging.info(f"Found {len(classes)} classes: {classes}")

    val_dataset = KuzushijiFlatDataset(val_dir, transform=data_transforms['val'], classes=classes)
    
    image_datasets = {'train': train_dataset, 'val': val_dataset}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    logging.info(f"Dataset sizes: {dataset_sizes}")

    # Model setup
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(classes))

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # Add weight decay for regularization
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Add Learning Rate Scheduler
    # Reduce LR by factor of 0.1 if val loss stops improving for 2 epochs
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=2, verbose=True)

    # Train
    model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=args.epochs, patience=args.patience)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'full_best_classifier_local.pth')
    
    # Save model and list of classes for inference
    torch.save({
        'model_state_dict': model_ft.state_dict(),
        'classes': classes
    }, save_path)
    
    logging.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()

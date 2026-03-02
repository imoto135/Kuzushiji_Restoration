import os
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    split_info = {
        "100241706": "train",
        "100249376": "train",
        "100249476": "train",
        "100249537": "train",
        "200003076": "train",
        "200003967": "train",
        "200004148": "train",
        "200006663": "train",
        "200006665": "train",
        "200008316": "train",
        "200014685": "train",
        "200015779": "train",
        "200020019": "train",
        "200021086": "train",
        "200021644": "train",
        "200021712": "train",
        "200021763": "train",
        "200021802": "train",
        "200021853": "train",
        "200021925": "train",
        "200025191": "train",
        "brsk00000": "train",
        "hnsd00000": "train",
        "umgy00000": "train",
        "100249371": "val",
        "100249416": "val",
        "200005598": "val",
        "200014740": "val",
        "200021637": "val",
        "200021660": "val",
        "200021851": "val",
        "200021869": "val",
        "200022050": "val",
        "200003803": "test",
        "200010454": "test",
        "200015843": "test",
        "200017458": "test",
        "200018243": "test",
        "200019865": "test",
        "200021063": "test",
        "200021071": "test",
        "200004107": "test",
        "200005798": "test",
        "200008003": "test",
    }
    
    source_dir = Path("/home/imoto/Kuzushiji_Restoration/data/full_dataset")
    target_dir = Path("/home/imoto/Kuzushiji_Restoration/data/full_characters")
    
    # Create target directories (train, val, test)
    for split in ["train", "val", "test"]:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
        
    for book_id, split in split_info.items():
        char_dir = source_dir / book_id / "characters"
        
        if not char_dir.exists():
            print(f"Directory not found, skipping: {char_dir}")
            continue
            
        print(f"\nProcessing {book_id} -> {split}...")
        
        # Find all images recursively under the `characters` directory
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            # Case insensitive path matching is possible but rglob is case-sensitive on linux
            # Using basic extensions since filenames generally follow these
            images.extend(list(char_dir.rglob(ext)))
            images.extend(list(char_dir.rglob(ext.upper())))
            
        target_split_dir = target_dir / split
        
        for img_path in tqdm(images, desc=f"Copying files for {book_id}", unit="file"):
            target_path = target_split_dir / img_path.name
            
            # Use copy2 to preserve metadata. 
            # If the file already exists, it will overwrite it. 
            # (Assuming filenames like U+843D_100241706_0... are globally unique).
            if not target_path.exists():
                shutil.copy2(img_path, target_path)

    print("\nDataset split completed successfully!")

if __name__ == "__main__":
    main()

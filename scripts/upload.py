import wandb
import os
from PIL import Image

# --- 設定 ---
PROJECT_NAME = "Kuzushiji_Restoration"
ENTITY = "imotoyuichi-ritsumeikan-university"
RUN_NAME = "Upload_All_Runs"
# 画像がたくさん入っている親ディレクトリ
RUNS_DIR = "Kuzushiji_Restoration/experiments"  # 例: experiments/ 以下に run1, run2... があると仮定
ARTIFACT_NAME = "experiment_results_archive"

def main():
    run = wandb.init(project=PROJECT_NAME, entity=ENTITY, name=RUN_NAME)
    
    # 1. Artifactを作成してフォルダごとアップロード
    print(f"Uploading directory: {RUNS_DIR} ...")
    artifact = wandb.Artifact(name=ARTIFACT_NAME, type="dataset")
    # フォルダを追加（再帰的に全て）
    artifact.add_dir(RUNS_DIR, name="runs_data") 
    run.log_artifact(artifact)
    print("Artifact upload scheduled. Now creating visualization table...")

    # 2. アップロードした画像をTableに登録して可視化
    # (ローカルのパスをスキャンしてTableを作る)
    table = wandb.Table(columns=["Run/Folder", "Filename", "Image"])
    
    image_count = 0
    # os.walkでサブディレクトリまで全探索
    for root, dirs, files in os.walk(RUNS_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                
                # 親フォルダ名（実験名など）を取得
                rel_path = os.path.relpath(img_path, RUNS_DIR)
                folder_name = os.path.dirname(rel_path) # 例: "segmentation/unet/resnet34"
                
                try:
                    img = Image.open(img_path)
                    # 画像サイズが大きすぎると重いのでリサイズしても良い
                    # img.thumbnail((512, 512)) 
                    
                    table.add_data(folder_name, file, wandb.Image(img))
                    image_count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

                if image_count % 100 == 0:
                    print(f"Processed {image_count} images...")

    print(f"Total {image_count} images found.")
    run.log({"All_Runs_Comparison": table})
    
    wandb.finish()

if __name__ == "__main__":
    main()
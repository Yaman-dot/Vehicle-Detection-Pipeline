import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil #copy files cuh
#config
CSV_PATH = r"E:\Documents\Codes\Python\vehicle detection pipeline\data\raw\train.csv"
IMG_DIR = r"E:\Documents\Codes\Python\vehicle detection pipeline\data\raw\train"
OUTPUT_DIR = r"E:\Documents\Codes\Python\vehicle detection pipeline\data"
CLASSES = ['Bus', 'Truck']
VAL_SPLIT = 0.2

def convert_to_yolo_bbox(xmin, xmax, ymin, ymax):
    x_center = (xmin+xmax) / 2
    y_center = (ymin+ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return x_center, y_center, width, height
def create_yolo_files():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["Unnamed: 0"])

    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)
    
    image_ids = df["ImageID"].unique()
    train_ids, val_ids = train_test_split(image_ids, test_size=VAL_SPLIT, random_state=42)
    
    for split, ids in [("train", train_ids), ("val", val_ids)]:
        subset = df[df["ImageID"].isin(ids)] #returns the specific subset that we are in rn
        for img_id, group in tqdm(subset.groupby("ImageID"), desc=f"Processing {split}"):
            label_path = f"{OUTPUT_DIR}/labels/{split}/{img_id}.txt"
            with open(label_path, "w") as f:
                for _, row in group.iterrows():
                    cls = CLASSES.index(row["LabelName"])
                    x_center, y_center, width, height = convert_to_yolo_bbox(row["XMin"], row["XMax"], row["YMin"], row["YMax"])
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            # Copy image safely using shutil
            src = os.path.join(IMG_DIR, f"{img_id}.jpg")
            dst = os.path.join(OUTPUT_DIR, "images", split, f"{img_id}.jpg")
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                print(f"[WARNING] Image not found: {src}")
    print(f"Finished Conversion, dataset created at {OUTPUT_DIR}")

create_yolo_files()


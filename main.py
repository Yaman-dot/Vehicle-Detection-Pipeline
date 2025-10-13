# main.py
from src.convert_to_yolo import create_yolo_files, create_yaml
from src.train import train_yolo_model
from src.evaluate import evaluate_model
from src.infer import run_inference

def main():
    # Step 1: Convert CSV -> YOLO format
    #create_yolo_files() #Already Done, uncomment if you want to preprocess the data.
    #create_yaml()

    # Step 2: Train YOLO model
    train_yolo_model(
        data_path="data/vehicles.yaml",
        model_path="yolov8s.pt",
        save_dir="runs/train",
        epochs=1,
        batch_size=16,
        img_size=640
    )

    # Step 3: Evaluate model
    evaluate_model(
        model_path="runs/train/train_results/weights/best.pt",
        data_yaml="data/vehicles.yaml"
    )

    # Step 4: Run inference on a sample image
    run_inference(
        model_path="runs/train/train_results/weights/best.pt",
        source="data/images/val/0000599864fd15b3.jpg"
    )

if __name__ == "__main__":
    main()

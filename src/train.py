import os
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Set non-interactive Matplotlib backend for compatibility in non-GUI environments
plt.switch_backend('Agg')

def plot_training_metrics(save_dir):
    # Path to the results CSV file
    results_csv_path = os.path.join(save_dir, "train_results", "results.csv")
    
    if not os.path.exists(results_csv_path):
        print(f"Results CSV not found at {results_csv_path}. Skipping plot.")
        return
    
    # Read metrics from CSV and strip any leading/trailing spaces from column names
    df = pd.read_csv(results_csv_path)
    df.columns = df.columns.str.strip()  # Remove spaces around column names
    
    # Extract epochs and metrics (adjust if your CSV has different columns)
    epochs = df['epoch']
    box_loss = df['train/box_loss']
    cls_loss = df['train/cls_loss']  # Assuming DFL loss is not plotted; add if needed
    precision = df['metrics/precision(B)']
    recall = df['metrics/recall(B)']
    map50 = df['metrics/mAP50(B)']
    map5095 = df['metrics/mAP50-95(B)']

    # Create the plot figure
    plt.figure(figsize=(12, 8))

    # Box/Class Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, box_loss, label="Box Loss", color="blue")
    plt.plot(epochs, cls_loss, label="Class Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(epochs, precision, label="Precision", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.title("Training Precision")
    plt.legend()

    # Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, recall, label="Recall", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.title("Training Recall")
    plt.legend()

    # mAP
    plt.subplot(2, 2, 4)
    plt.plot(epochs, map50, label="mAP@0.5", color="red")
    plt.plot(epochs, map5095, label="mAP@0.5:0.95", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.title("Mean Average Precision (mAP)")
    plt.legend()

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"training_metrics_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Training metrics plot saved to {plot_path}")

def train_yolo_model(data_path, model_path, save_dir, epochs=50, batch_size=16, img_size=640):
    
    os.makedirs(save_dir, exist_ok=True)  # Create save_dir if it doesn't exist
    model = YOLO(model_path)
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=save_dir,
        name="train_results",
        exist_ok=True  # Do not overwrite existing results
    )
    
    # Plot metrics from the CSV (no need to pass results, as per-epoch data is in CSV)
    plot_training_metrics(save_dir)
    
    # Path to the best model (YOLO saves it automatically if val=True, which is default)
    best_model_path = os.path.join(save_dir, "train_results", "weights", "best.pt")
    
    print(f"Training Complete. Results are saved in {save_dir}")
    
    return best_model_path
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime


def plot_training_metrics(results, save_dir):
    
    # Extract metrics safely
    metrics = results.results_dict
    epochs = range(1, len(metrics['train/box_loss']) + 1)

    # Prepare data
    box_loss = metrics['train/box_loss']
    cls_loss = metrics['train/cls_loss']
    precision = metrics['metrics/precision(B)']
    recall = metrics['metrics/recall(B)']
    map50 = metrics['metrics/mAP50(B)']
    map5095 = metrics['metrics/mAP50-95(B)']

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

def train_yolo_model(data_path, model_path, save_dir, epochs = 50, batch_size = 16, img_size = 640):
    
    os.makedirs(save_dir, exist_ok=True) #if savedirs is not available, create it, if not just dont overwrite it
    model = YOLO(model_path)
    
    results = model.train(
        data = data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=save_dir,
        name="train_results",
        exist_ok=True #to not overwrite the existing results.
    )
    best_model_path = os.path.join(save_dir, "train_results", "weights", "best.pt")
    plot_training_metrics(results, save_dir)
    print(f"Training Complete, Results are saved in {save_dir}")
    
    return best_model_path
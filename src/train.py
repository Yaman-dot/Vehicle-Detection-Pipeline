import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from datetime import datetime


def plot_training_metrics(results, save_dir):
    
    # Extract metrics from results
    epochs = range(len(results['metrics']['loss']))
    loss = results['metrics']['loss']
    precision = results['metrics']['precision']
    recall = results['metrics']['recall']
    map50 = results['metrics']['map50']  # mAP at IoU=0.5
    map95 = results['metrics']['map95'] 
    
    #Create the plot figure for the metrics
    plt.figure(figsize=(12,8))
    
    #plot the loss
    plt.subplot(2,2,1)
    plt.plot(epochs, loss, label="loss", color = "blue")
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot Precision
    plt.subplot(2, 2, 2)
    plt.plot(epochs, precision, label='Precision', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Training Precision')
    plt.legend()

    # Plot Recall
    plt.subplot(2, 2, 3)
    plt.plot(epochs, recall, label='Recall', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Training Recall')
    plt.legend()

    # Plot mAP
    plt.subplot(2, 2, 4)
    plt.plot(epochs, map50, label='mAP@0.5', color='red')
    plt.plot(epochs, map95, label='mAP@0.5:0.95', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision (mAP)')
    plt.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(save_dir, f'training_metrics_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"Training metrics plot saved to {plot_path}")

def train_yolo_model(data_path, model_path, save_dir, epochs = 50, batch_size = 16, img_size = 640):
    
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
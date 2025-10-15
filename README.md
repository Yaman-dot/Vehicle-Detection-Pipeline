# Vehicle-Detection-Pipeline
This project implements a custom pipeline for training, evaluating and running inference on a YoloV8 model **from scratch** for detecting Buses and Trucks. The pipeline includes scripts for data preparation (converting raw csv annotations to YOLO format), model training, evaluation and inference


## Requirements
* Python 3.8+
* PyTorch
* GPU with CUDA (recommended)
* Required Packages
    * Ultralytics
    * pandas
    * scikit-learn
    * PyYaml
    * tqdm
    * matplotlib
To install them:

```bash
pip install ultralytics pandas scikit-learn pyyaml tqdm matplotlib
```

## Data Setup

For datasets with the same format as the dataset I used, you can run the conversion script

```python
create_yolo_files()
create_yaml()
```

```create_yolo_files()```: Reads the CSV, splits image IDs into training (80%) and validation (20%), converts bounding box coordinates to the normalized YOLO format (x_center, y_center, width, height), and copies images/labels into data/images and data/labels subdirectories.

```create_yaml()```: Creates the data/vehicles.yaml configuration file required by the ultralytics library for training.

## Model Training
The ```main.py``` script is set up to run the training process

```python
# In main.py
train_yolo_model(
    data_path="data/vehicles.yaml",
    model_path="yolov8s.pt", # Starting from a pre-trained small YOLOv8 model
    save_dir="runs/train",
    epochs=2,              # Set to 2 for a quick test, increase for actual training
    batch_size=16,
    img_size=224           # Small size for a quick test, use 640 for standard training
)
```

**Output**: The trained model weights and logs will be saved under the ```runs/train/train_results``` directory. A plot of the training metrics (```training_metrics_*.png```) is also generated in the ```runs/train``` directory

## Evaluation and Inference

After training, you can evaluate the model on the validation set or run inference on new images.

1. Model Evaluation

    Uncomment the evaluation step in ```main.py``` and run the script. Specify your own ```.pt``` model or use Yolo pretrained models

    ```python
    # In main.py
    evaluate_model(
        model_path="path/to/your/best.pt",
        data_yaml="data/vehicles.yaml"
    )
    ```

    **Output**: Prints evaluation metrics like mAP50, mAP50-95, Precision, and Recall on the validation set.
2. Model Inference

    Uncomment the inference step in ```main.py``` and set a valid image path for the ```source``` argument

    ```python
    # In main.py
    run_inference(
        model_path="path/to/your/best.pt",
        source="path/to/your/image.jpg"
    )
    ```

    **Output**: The predicted image (with bounding boxes) will be saved by ```ultralytics``` to a new subdirectory within ```runs/detect``` (or the save_dirs you specify in ```infer.py```)

## Custom YOLOv8 Model Implementation

The ```src/model.py``` file contains custom implementations of core YOLOv8 architecture blocks, showing the underlying structure of the model's backbone:

* ```Conv Block```: Standard Convolution → Batch Normalization → SiLU activation.

* ```BottleNeck Block```: A residual unit with two Conv layers.

* ```C2F``` (C3-Faster): The main building block of the YOLOv8 backbone and neck, featuring a split-and-concatenate design with bottleneck modules to enhance feature extraction and reduce parameters

* ```SPPF``` (Spatial Pyramid Pooling - Fast): Aggregates features at different scales using concatenated max-pooling layers, significantly improving feature map robustness

* ```Backbone```: Defines the full architecture of the YOLOv8 feature extractor, structured according to the scaling factors (d,w,r) derived from the specified YOLO version (e.g., "s" for small)
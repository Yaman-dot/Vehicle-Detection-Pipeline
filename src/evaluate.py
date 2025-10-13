from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, split="val"):
    
    model = YOLO(model_path)
    results = model.val()
    
    #Print the summary
    print("Evaluation Results")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    return results
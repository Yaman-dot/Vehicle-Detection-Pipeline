from ultralytics import YOLO
import os

def run_inference(model_path, source, save_dirs):
    model = YOLO(model_path)
    results = model.predict(source=source, save=True ,show=False)
    
    print("Inference Complete")
    print(f"Results saved to: {save_dirs}")
    
    return results


#run_inference("yolov8s.pt", "E:/Documents/Codes\Python/vehicle detection pipeline/data/images/train/0a0a43730e647024.jpg",  r"E:\Documents\Codes\Python\vehicle detection pipeline\data")
from ultralytics import YOLO

def train_model():
    print("Starting YOLO11 training for SiteVision...")
    model = YOLO("yolo11s.pt")  # downloads pretrained weights automatically (~20MB)
    
    # Train the model
    results = model.train(
        data="data.yaml",   # points to your dataset folders
        epochs=20,
        imgsz=640,
        project="construction-detector",
        name="site_vision_model"
    )
    
    print("Training complete! Model weights saved in construction-detector/site_vision_model/weights/best.pt")

if __name__ == "__main__":
    train_model()

from ultralytics import YOLO

# Load the trained model
model_path = r"C:\Capstone\Yolo\YoloPoseTesting\runs\classify\train2\weights\best.pt"
model = YOLO(model_path)

# Print class names
print(model.names)
import cv2
import os
import pandas as pd
from prophet import Prophet
import torch
from sort import Sort  # Make sure to install SORT or Deep SORT library


# Function to extract frames from video
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0
    while success:
        cv2.imwrite(f"{output_folder}/frame{count}.jpg", frame)
        success, frame = video.read()
        count += 1

# Extract frames
video_path = 'sample.mp4'
output_folder = 'extract'
extract_frames(video_path, output_folder)




# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Function to detect vehicles in an image
def detect_vehicles(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    return results.pandas().xyxy[0]  # return detections as pandas dataframe

# Test detection on a single frame
frame_path = f"{output_folder}/frame0.jpg"
detections = detect_vehicles(frame_path)
print(detections)






# Initialize tracker
tracker = Sort()

# Function to track vehicles across frames
def track_vehicles(detections):
    tracked_objects = tracker.update(detections[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].to_numpy())
    return tracked_objects

# Track vehicles in the first frame
tracked_objects = track_vehicles(detections)
print(tracked_objects)




# Initialize counters
vehicle_counts = { 'Car': 0, 'Bus': 0, 'Truck': 0, 'Three-Wheeler': 0, 'Two-Wheeler': 0, 'LCV': 0, 'Bicycle': 0 }

# Function to update counts based on detections
def update_counts(detections):
    for _, row in detections.iterrows():
        vehicle_class = row['name']  # 'name' should correspond to vehicle class in the detection results
        if vehicle_class in vehicle_counts:
            vehicle_counts[vehicle_class] += 1

# Update counts for the first frame
update_counts(detections)
print(vehicle_counts)





# Create a dataframe for historical counts (sample data)
data = {'ds': pd.date_range(start='2023-01-01', periods=100, freq='T'),
        'y': [i + (i%7) for i in range(100)]}  # Sample counts
df = pd.DataFrame(data)

# Train prediction model
model = Prophet()
model.fit(df)

# Make future dataframe and predict
future = model.make_future_dataframe(periods=30, freq='T')
forecast = model.predict(future)
print(forecast[['ds', 'yhat']])



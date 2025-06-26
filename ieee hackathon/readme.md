
# Vehicle Detection, Tracking & Forecasting System

This project provides a complete pipeline for detecting, tracking, and forecasting vehicle traffic patterns from video files. It leverages computer vision, deep learning, and time series forecasting for traffic analysis.

---

## Features

- Detect vehicles in videos using YOLOv5
- Track moving vehicles across frames using SORT (Simple Online and Realtime Tracking)
- Count different types of vehicles (e.g., car, truck, bus)
- Predict future traffic volume using Facebook Prophet
- Visualize vehicle movement and counts on video frames

---

## Technologies Used

- **OpenCV**: For video processing and computer vision
- **YOLOv5 (Ultralytics)**: Object detection model
- **SORT**: Real-time tracking using Kalman filter and IoU matching
- **Facebook Prophet**: Time series forecasting
- **Matplotlib**: Visualization (optional for SORT display)

---

## Directory Structure

```
project/
│
├── video.mp4                         # Input video for counting
├── sample.mp4                        # Sample video for detection + tracking
├── extract/                          # Folder for extracted frames
├── sort.py                           # SORT tracking implementation
├── count_vehicles_opencv.py          # Basic vehicle counter using OpenCV
├── detect_track_forecast.py          # YOLO detection + SORT tracking + forecasting
├── README.md                         # Documentation
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-tracker.git
cd vehicle-tracker
```

---

### 2. Install Dependencies

Install the required libraries:

```bash
pip install opencv-python pandas torch torchvision torchaudio matplotlib scikit-image filterpy prophet
```

Make sure to also install `lap` if available (optional for better assignment performance):

```bash
pip install lap
```

---

## Modules Overview

### 1. `count_vehicles_opencv.py`

- Detects motion using background subtraction (MOG)
- Draws bounding boxes on moving objects
- Counts vehicles when they cross a defined line

### 2. `detect_track_forecast.py`

- Extracts frames from the video
- Applies YOLOv5 object detection
- Tracks objects using SORT
- Categorizes detected objects and updates vehicle type counts
- Applies Prophet to forecast traffic volume over time

### 3. `sort.py`

- Full Python implementation of SORT algorithm
- Includes Kalman filter, IOU matching, and tracker management

---

## How to Run

### 1. Basic Vehicle Counter (OpenCV + MOG)

```bash
python count_vehicles_opencv.py
```

Make sure `video.mp4` is available in the same directory.

---

### 2. Detection + Tracking + Forecasting

```bash
python detect_track_forecast.py
```

Make sure `sample.mp4` is in the directory or update the `video_path`.

---

## Outputs

- Frame-wise detections and object tracking IDs
- Vehicle counts per class:
  - Car
  - Bus
  - Truck
  - Bicycle
  - Two-Wheeler
  - Three-Wheeler
  - LCV

- Forecasted vehicle flow using Prophet:
  - `ds`: Timestamp
  - `yhat`: Predicted vehicle count

---

## Sample Output (Console)

```
   name        xmin   ymin   xmax   ymax  confidence
0   car       102.0   88.0   180.0  160.0       0.84
1  truck      200.0  100.0   300.0  180.0       0.76
...

Tracked Objects:
[[102.0 88.0 180.0 160.0  1]]
...

Vehicle Counts:
{'Car': 8, 'Bus': 2, 'Truck': 4, ...}

Forecast Output:
                   ds       yhat
0 2023-01-01 00:00:00   0.000000
1 2023-01-01 00:01:00   1.034938
...
```

---

## Notes

- YOLOv5 model is downloaded via `torch.hub`; ensure internet access during the first run.
- SORT is implemented from scratch inside `sort.py`, no external tracker needed.
- Prophet uses dummy historical data. For real-time use, integrate actual vehicle counts per minute/hour.

---

## Future Improvements

- Improve classification accuracy using custom-trained YOLO
- Store predictions in a database or dashboard
- Add support for real-time video streaming (e.g., webcam or CCTV)
- Generate plots from forecasting results

---

## License

This project is for research and educational purposes only.

---

## Credits

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [SORT by Bewley et al.](https://github.com/abewley/sort)
- [Facebook Prophet](https://facebook.github.io/prophet/)

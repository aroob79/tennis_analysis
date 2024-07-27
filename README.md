# Tennis Analysis

In this project, I developed a deep learning-based tennis game analysis system using YOLOv9 and TensorFlow. The system employs three models: one for predicting the ball, one for predicting the player, court, and net, and one for predicting the keypoints of the court. For keypoint detection, VGG19 serves as the base model. The system first detects the ball, player, net, court, and keypoints of the court. Subsequently, the ball and player positions are transformed to a mini court using homography transformation, mapping their positions onto this mini court. The average ball speed and the speed of each player are then calculated and displayed. Here is an image of the output video:

![Output Video Image](https://github.com/user-attachments/assets/0fab3869-5284-4e29-97ac-bf138193d7cf)

## Dataset

For training the model, the following datasets were used:
- **Ball Detection Dataset**: [Available here](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)
- **Player, Net, and Court Detection Dataset**: [Available here](https://universe.roboflow.com/roboten/tennis-vhrs9)
- **Keypoints Detection Dataset**: [Available here](https://github.com/yastrebksv/TennisCourtDetector) and in [pickle format](https://drive.google.com/drive/u/4/folders/1_GFFgyqhg2X3XyUnvgAvaP4NJj40qpgd)

## Model

Three models were used in this work:
- **YOLOv9c** for player, net, and court detection
- **YOLOv9c** for ball detection
- **TensorFlow model** for keypoints detection using VGG19 as the base network

The trained models are available [here](https://drive.google.com/drive/u/4/folders/1Qsi0-23gbv0OsQJPW0l6WZ9-SmVoZIBe).

## Requirements

- Python 3.11
- Ultralytics
- TensorFlow
- OpenCV
- NumPy
- Pandas

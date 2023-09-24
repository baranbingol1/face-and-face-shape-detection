# face-and-face-shape-detection
Detects the face and the shape of the face puts a real time bounding box from webcam

## Usage
You can use it via downloading detection.py and best_model.pth files, then run the detection.py(you need torch, torchvision, cv2, pillow dependicies). It will start the real time detection from webcam, press "q" to quit.

I also added the notebook that i used for the classification model. (The model is fine-tuned EfficientNetB4 and has ~85% accuracy on test set)

I used "Face Shape Dataset" that is uploaded on kaggle : https://www.kaggle.com/datasets/niten19/face-shape-dataset

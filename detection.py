import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = torchvision.models.efficientnet_b4(pretrained=True)
num_classes = 5
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(model.classifier[1].in_features, num_classes)
)

model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))

model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)

# Renkler Blue-Green-Red formatında
shape_colors = {
    'Kalp': (0, 0, 255),   # Kırmızı
    'Dikdortgen': (0, 255, 255),  # Sarı
    'Oval': (255, 0, 0),    # Mavi
    'Yuvarlak': (255, 255, 0), # Açık Mavi
    'Kare': (0, 255, 0)  # Yeşil
}

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            predicted_class = model(face_tensor)
            _, predicted_class = predicted_class.max(1)

        face_shape_idx_to_class = {0: 'Kalp', 1: 'Dikdortgen', 2: 'Oval', 3: 'Yuvarlak', 4: 'Kare'}
        shape_label = face_shape_idx_to_class[predicted_class.item()]

        bbox_color = shape_colors.get(shape_label, (0, 0, 0))

        cv2.rectangle(frame, (x, y), (x+w, y+h), bbox_color, 2)

        cv2.putText(frame, f"Yuz sekli: {shape_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

    cv2.imshow('Yuz Tespiti ve Yuz Sekli Siniflandirmasi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

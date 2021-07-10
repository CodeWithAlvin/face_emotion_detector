import cv2
import numpy as np
from tensorflow import keras

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model_path="models\CNN(64,128,512,512)Dense(512,1024)model.h5"

def load_model(model_path):
    return keras.models.load_model(model_path)

model=load_model(model_path)

video=cv2.VideoCapture(0)

while True:
    _,f=video.read()
    transform=cv2.cvtColor(cv2.resize(f,(48,48)),cv2.COLOR_BGR2GRAY).reshape(-1,48,48,1)
    res=model.predict([transform/255])
    cv2.putText(f,emotion_dict[np.argmax(res)],(25, 20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
    cv2.imshow("Emotion Detection",f)
    cv2.waitKey(1)
    

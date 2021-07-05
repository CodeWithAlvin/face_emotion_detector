import pickle
import cv2
import numpy as np

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model_path="/models/conv3-5-july.h5"

def load_model(model_path,type="keras"):
    #type --> pickle | keras
    if type == "pickle":
        with open(model_path,"rb") as f:
            return pickle.load(f)
    elif type == "keras":
        from tensorflow import keras
        return keras.model.load(model_path)
    else:
        raise ValueError("unknown value {type} must be either pickle or keras")

model=load_model(model_path,"keras")

video=cv2.VideoCapture(0)

while True:
    _,f=video.read()
    res=model.predict([cv2.cvtColor(cv2.resize(f,(48,48)),cv2.COLOR_BGR2GRAY).flatten()])
    cv2.putText(f,emotion_dict[np.argmax(result)],(25, 20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
    cv2.imshow("Emotion Detection",f)
    cv2.waitKey(1)

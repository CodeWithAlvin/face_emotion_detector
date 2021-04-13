import pickle
import cv2

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

with open("model_random.pkl","rb") as f:
    model=pickle.load(f)

video=cv2.VideoCapture(0)

while True:
    _,f=video.read()
    res=model.predict([cv2.cvtColor(cv2.resize(f,(48,48)),cv2.COLOR_BGR2GRAY).flatten()])
    cv2.putText(f,emotion_dict[res[0]],(25, 20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2)
    cv2.imshow("Emotion Detection",f)
    cv2.waitKey(1)

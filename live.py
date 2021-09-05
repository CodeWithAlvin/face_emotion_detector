# Importing all required packages
import cv2
import numpy as np
from tensorflow import keras

print("import done")

# Read in the cascade classifiers for face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# create a function to detect face
def adjusted_detect_face(img):
    face_rect = face_cascade.detectMultiScale(img,
                                              scaleFactor = 1.2,
                                              minNeighbors = 5)
    padding=20
    for (x, y, w, h) in face_rect:
        img = img[x:x + (w+(3*padding)),  y:y + (h+(3*padding))]
	    
    return img

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

print("loading model ")
model=keras.models.load_model("4CNN,4LSTM,2DENSE model.h5")
print("model loaded")

def predict(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(48,48))
    img=img/255
    img=img.reshape(-1,48,48,1)
    res=model.predict(img)
    res=np.argmax(res)
    return emotion_dict[res]


if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        img=adjusted_detect_face(image)
        res=predict(img)
        cv2.putText(image, res, (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 196, 255), 2)
        cv2.imshow('face Detection', image)
        cv2.imshow('mdl img', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

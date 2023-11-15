import cv2
import face_recognition
import joblib
import numpy as np

def load_model(model_path, label_encoder_path):
    # Load the trained model
    clf = joblib.load(model_path)
    
    # Load the label encoder
    label_encoder = joblib.load(label_encoder_path)

    return clf, label_encoder

def recognize_faces(video_capture, model, label_encoder):
    while True: 
        ret, frame = video_capture.read()
        
        # Find all face locations and face encoding in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Use the trained model to predict the person's label
            label_idx = model.predict([face_encoding])[0]
            
            # Map numerical label back to the original string label
            label = label_encoder.inverse_transform([label_idx])[0]
            
            # Draw a rectangle around the face and display the label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Specify the paths to the trained model and label encoder
    model_path = "face_recognition_model.joblib"
    label_encoder_path = "label_encoder.joblib"
    
    # Load the trained model and label encoder
    trained_model, label_encoder = load_model(model_path, label_encoder_path)
    
    # Open the webcam
    video_capture = cv2.VideoCapture(0)
    
    # Use the trained model for live face recognition
    recognize_faces(video_capture, trained_model, label_encoder)

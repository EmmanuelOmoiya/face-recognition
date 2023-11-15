# import os
# import face_recognition
# import numpy as np
# import joblib
# import cv2
# from tensorflow.keras import models, layers
# from sklearn.preprocessing import LabelEncoder


# def train_face_model(dataset_folder):
#     face_encodings = []
#     labels = []

#     for person_name in os.listdir(dataset_folder):
#         person_folder = os.path.join(dataset_folder, person_name)
        
#         if os.path.isdir(person_folder):
#             for filename in os.listdir(person_folder):
#                 if filename.lower().endswith((".jpg", ".png")):
#                     image_path = os.path.join(person_folder, filename)
#                     print(f"Loading image: {image_path}, Label: {person_name}")
#                     image = face_recognition.load_image_file(image_path)

#                     # Resize the image to a fixed size
#                     image = cv2.resize(image, (100, 100))

#                     # Extract face encodings
#                     face_encodings.extend(face_recognition.face_encodings(image))
#                     # labels.extend([person_name] * len(face_encodings))
#         # Extend labels for the current person after processing all images
#                     labels.extend([person_name] * len(face_recognition.face_encodings(image)))

#     # Encode string labels to numerical format
#     label_encoder = LabelEncoder()
#     y_train_encoded = label_encoder.fit_transform(labels)

#     return np.array(face_encodings), y_train_encoded, label_encoder  # Ensure y_train is encoded

# if __name__ == "__main__":
#     # Specify the folder containing the dataset
#     dataset_folder = "data"

#     # Train the face recognition model 
#     face_encodings, y_train, label_encoder = train_face_model(dataset_folder)

#     if not face_encodings.size or not y_train.size:
#         print('No data to work on')
#     else:
#         # Ensure consistent number of unique labels
#         num_classes = len(set(y_train.flatten()))
        
#         # Check
#         print(f"Number of samples in face_encodings: {face_encodings.shape[0]}")
#         print(f"Number of samples in labels: {y_train.shape[0]}: {y_train}")

#         # Check the number of samples in face_encodings and labels
#         assert face_encodings.shape[0] == y_train.shape[0], f"Mismatch in the number of samples between face_encodings ({face_encodings.shape[0]}) and labels ({y_train.shape[0]})"
#         print("Dataset loaded successfully.")


#         # Define a simple dense model
#         model = models.Sequential([
#             layers.Flatten(input_shape=(face_encodings.shape[1],)),
#             layers.Dense(128, activation='relu'),
#             layers.Dense(num_classes, activation='softmax')
#         ])

#         # Compile the model
#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#         # Train the model
#         model.fit(face_encodings, y_train, epochs=10)

#         # Save the trained model and label_encoder to files
#         joblib.dump(model, 'face_recognition_model.joblib')
#         joblib.dump(label_encoder, 'label_encoder.joblib')
#         print("Face recognition model and label encoder trained and saved")



import os
import face_recognition
import numpy as np
import joblib
import cv2
from tensorflow.keras import models, layers
from sklearn.preprocessing import LabelEncoder


def load_image(image_path):
    """Load and preprocess the image."""
    image = face_recognition.load_image_file(image_path)
    return cv2.resize(image, (100, 100))


def extract_face_encodings(image):
    """Extract face encodings from the given image."""
    return face_recognition.face_encodings(image)


def load_dataset(dataset_folder):
    """Load the dataset and extract face encodings with labels."""
    face_encodings = []
    labels = []

    for person_name in os.listdir(dataset_folder):
        person_folder = os.path.join(dataset_folder, person_name)

        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".jpg", ".png")):
                    image_path = os.path.join(person_folder, filename)
                    print(f"Loading image: {image_path}, Label: {person_name}")
                    image = load_image(image_path)

                    # Extract face encodings
                    face_encodings.extend(extract_face_encodings(image))
                    # Extend labels for the current person after processing all images
                    labels.extend([person_name] * len(extract_face_encodings(image)))

    return np.array(face_encodings), labels


def encode_labels(labels):
    """Encode string labels to numerical format."""
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels), label_encoder


def train_model(face_encodings, labels):
    """Train the face recognition model."""
    num_classes = len(set(labels))
    
    # Define a simple dense model
    model = models.Sequential([
        layers.Flatten(input_shape=(face_encodings.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(face_encodings, labels, epochs=10)

    return model


def save_model_and_encoder(model, label_encoder):
    """Save the trained model and label_encoder to files."""
    joblib.dump(model, 'face_recognition_model.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("Face recognition model and label encoder trained and saved")


if __name__ == "__main__":
    # Specify the folder containing the dataset
    dataset_folder = "data"

    # Load and preprocess the dataset
    face_encodings, labels = load_dataset(dataset_folder)

    if not face_encodings.size or not labels:
        print('No data to work on')
    else:
        # Encode string labels to numerical format
        y_train, label_encoder = encode_labels(labels)

        # Check
        print(f"Number of samples in face_encodings: {face_encodings.shape[0]}")
        print(f"Number of samples in labels: {len(labels)}")

        # Check the number of samples in face_encodings and labels
        assert face_encodings.shape[0] == len(labels), f"Mismatch in the number of samples between face_encodings ({face_encodings.shape[0]}) and labels ({len(labels)})"
        print("Dataset loaded successfully.")

        # Train the face recognition model 
        trained_model = train_model(face_encodings, y_train)

        # Save the trained model and label_encoder to files
        save_model_and_encoder(trained_model, label_encoder)

# import cv2
# from sqlalchemy.orm import Session
# from database import FeatureData
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from retinaface import RetinaFace
# from deepface import DeepFace
# import os

# def detect_faces(image_rgb):
#     faces = RetinaFace.detect_faces(image_rgb)
#     face_data = []
#     if faces:
#         for face_id, face_info in faces.items():
#             facial_area = face_info['facial_area']
#             x1, y1, x2, y2 = facial_area
#             person_image = image_rgb[y1:y2, x1:x2]
#             face_data.append((facial_area, person_image))
#     return face_data

# def extract_features(image):
#     # Replace with the actual implementation
#     transformations = [lambda x: x, 
#                        lambda x: cv2.flip(x, 1), 
#                        lambda x: cv2.GaussianBlur(x, (5, 5), 0)]
    
#     feature_list = []
#     for transform in transformations:
#         transformed_image = transform(image)
#         features = DeepFace.represent(img_path=transformed_image, model_name="Facenet", enforce_detection=False)
#         if features and len(features) > 0:
#             feature_list.append(np.array(features[0]["embedding"], dtype=np.float32))
    
#     return np.mean(feature_list, axis=0)

# # Save person embeddings to the database
# def save_person_embedding(db: Session, person_id: int,embedding: np.ndarray):
#     embedding_binary = embedding.tobytes()
#     person = FeatureData(id=person_id ,face_embedding=embedding_binary)
#     db.add(person)
#     db.commit()

# # Function to recognize face from the database
# def recognize_face(db: Session, features: np.ndarray, threshold=0.6):
#     best_match_id = None
#     best_similarity = 0

#     # Fetch all known persons and their embeddings from the database
#     persons = db.query(FeatureData).all()
    
#     for person in persons:
#         stored_features = np.frombuffer(person.face_embedding, dtype=np.float32)
#         similarity = cosine_similarity([features], [stored_features])[0][0]
        
#         if similarity > best_similarity and similarity >= threshold:
#             best_similarity = similarity
#             best_match_id = person.id

#     # If no match is found, assign a new person_id and save the embedding
#     if best_match_id is None:
#         best_match_id = len(persons) + 1
#         save_person_embedding(db,best_match_id,features)

#     return best_match_id, best_similarity

import cv2
from sqlalchemy.orm import Session
from database import FeatureData
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from retinaface import RetinaFace
from deepface import DeepFace

def detect_faces(image_rgb):
    """Detect faces in an image using RetinaFace."""
    faces = RetinaFace.detect_faces(image_rgb)
    face_data = []
    if faces:
        for face_id, face_info in faces.items():
            facial_area = face_info['facial_area']
            x1, y1, x2, y2 = facial_area
            person_image = image_rgb[y1:y2, x1:x2]
            face_data.append((facial_area, person_image))
    return face_data

def extract_features(image):
    """Extract face embeddings using DeepFace."""
    transformations = [
        lambda x: x,
        lambda x: cv2.flip(x, 1),
        lambda x: cv2.GaussianBlur(x, (5, 5), 0)
    ]
    feature_list = []
    for transform in transformations:
        transformed_image = transform(image)
        features = DeepFace.represent(
            img_path=transformed_image, model_name="Facenet", enforce_detection=False
        )
        if features and len(features) > 0:
            feature_list.append(np.array(features[0]["embedding"], dtype=np.float32))
    return np.mean(feature_list, axis=0)

def save_person_embedding(db: Session, person_id: int, embedding: np.ndarray):
    """Save face embeddings to the database."""
    embedding_binary = embedding.tobytes()
    person = FeatureData(id=person_id, face_embedding=embedding_binary)
    db.add(person)
    db.commit()

def recognize_face(db: Session, features: np.ndarray, threshold=0.6):
    """Recognize face and return person ID and similarity score."""
    best_match_id = None
    best_similarity = 0
    persons = db.query(FeatureData).all()
    for person in persons:
        stored_features = np.frombuffer(person.face_embedding, dtype=np.float32)
        similarity = cosine_similarity([features], [stored_features])[0][0]
        if similarity > best_similarity and similarity >= threshold:
            best_similarity = similarity
            best_match_id = person.id
    if best_match_id is None:
        best_match_id = len(persons) + 1
        save_person_embedding(db, best_match_id, features)
    return best_match_id, best_similarity

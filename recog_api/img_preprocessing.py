# from fastapi import APIRouter, UploadFile, Depends, HTTPException, File, Form
# from sqlalchemy.orm import Session
# from database import get_db, Prediction
# from utils import detect_faces, extract_features, recognize_face
# import cv2
# import numpy as np
# import os

# router = APIRouter()

# def process_image(image, filename, db: Session):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     faces = detect_faces(image_rgb)
#     detected_classes = set()

#     if faces:
#         for face_info in faces:
#             facial_area, person_image = face_info
#             features = extract_features(person_image)
#             person_id, similarity = recognize_face(db, features)

#             detected_classes.add(f'person_{person_id}')

#             # Convert numpy.float32 to Python float
#             confidence_value = round(float(similarity),2)

#             # Store the prediction in the database
#             db_image_record = Prediction(
#                 file_path=filename,
#                 person_id=person_id,
#                 confidence=confidence_value,
#                 predicted_class=f'person_{person_id}'
#             )
#             db.add(db_image_record)
#             db.commit()

#             # Draw a rectangle and put text for the detected person
#             cv2.rectangle(image, facial_area[0:2], facial_area[2:4], (0, 255, 0), 2)
#             cv2.putText(image, f'Person {person_id}', facial_area[0:2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#         # Save the modified image in class-specific folders
#         output_base_dir = 'classified_images'
#         os.makedirs(output_base_dir, exist_ok=True)

#         for class_name in detected_classes:
#             class_output_dir = os.path.join(output_base_dir, class_name)
#             os.makedirs(class_output_dir, exist_ok=True)
            
#             output_path = os.path.join(class_output_dir, filename)
#             cv2.imwrite(output_path, image)

#     return {"status": "success", "file_path": filename}


# @router.post("/upload/")
# async def upload_image(
#     db: Session = Depends(get_db),
#     files: list[UploadFile] = File(...),  # Accept one or multiple files
# ):
#     if not files:
#         raise HTTPException(status_code=422, detail="No files were uploaded.")

#     results = []
    
#     # Handle multiple image uploads
#     for file in files:
#         image_data = await file.read()
#         image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
#         if image is None:
#             raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
#         result = process_image(image, file.filename, db)
#         results.append(result)

#     return {"results": results}
    
    
from fastapi import FastAPI, APIRouter, UploadFile, Depends, HTTPException, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError
from database import engine, get_db, Base, Prediction
from utils import detect_faces, extract_features, recognize_face
import cv2
import numpy as np
import os

app = FastAPI()
router = APIRouter()

def ensure_email_field_exists():
    """Ensure 'email' field exists in the 'predictions' table."""
    inspector = inspect(engine)
    if "predictions" in inspector.get_table_names():
        columns = [col["name"] for col in inspector.get_columns("predictions")]
        if "email" not in columns:
            with engine.connect() as connection:
                try:
                    connection.execute(text("ALTER TABLE predictions ADD COLUMN email VARCHAR(255)"))
                    print("Added 'email' column to 'predictions' table.")
                except OperationalError as e:
                    print(f"Error adding 'email' column: {e}")
        else:
            print("'email' column already exists.")
    else:
        print("'predictions' table does not exist. Creating tables...")
        Base.metadata.create_all(bind=engine)

def process_image(image, filename, db: Session, email: str = None):
    """Process the uploaded image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(image_rgb)
    detected_classes = set()
    matched_email = None
    if faces:
        for face_info in faces:
            facial_area, person_image = face_info
            features = extract_features(person_image)
            person_id, similarity = recognize_face(db, features)
            detected_classes.add(f'person_{person_id}')
            confidence_value = round(float(similarity), 2)
            db_image_record = Prediction(
                file_path=filename,
                person_id=person_id,
                confidence=confidence_value,
                predicted_class=f'person_{person_id}',
                email=email
            )
            db.add(db_image_record)
            db.commit()
            if email is None and confidence_value >= 0.8:
                matched_email = db.query(Prediction).filter_by(person_id=person_id).first()
                matched_email = matched_email.email if matched_email else None
            cv2.rectangle(image, facial_area[0:2], facial_area[2:4], (0, 255, 0), 2)
            cv2.putText(image, f'Person {person_id}', facial_area[0:2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        output_base_dir = 'classified_images'
        os.makedirs(output_base_dir, exist_ok=True)
        for class_name in detected_classes:
            class_output_dir = os.path.join(output_base_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            output_path = os.path.join(class_output_dir, filename)
            cv2.imwrite(output_path, image)
    return {"status": "success", "file_path": filename, "matched_email": matched_email}

@router.post("/upload/")
async def upload_image(
    email: str = Form(None),
    db: Session = Depends(get_db),
    files: list[UploadFile] = File(...)
):
    if not files:
        raise HTTPException(status_code=422, detail="No files were uploaded.")
    results = []
    for file in files:
        image_data = await file.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
        result = process_image(image, file.filename, db, email=email)
        results.append(result)
    return {"results": results}

@app.on_event("startup")
def startup_event():
    ensure_email_field_exists()

app.include_router(router)

# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import Column, Integer, String, Float,LargeBinary
# import numpy as np


# # Update the database URL if using environment variables
# SQLALCHEMY_DATABASE_URL = "mysql+pymysql://ec_user:user%4088@3.110.123.38:3306/ecommerce_db"
# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class Prediction(Base):
#     __tablename__ = "predictions"

#     id = Column(Integer, primary_key=True, index=True)  # Unique ID for each record
#     file_path = Column(String(255), index=True)
#     person_id = Column(Integer, index=True)
#     confidence = Column(Float)
#     predicted_class = Column(String(255), index=True)
#     email = Column(String, nullable=True)  # Optional email field

# # New table to store person details and face embeddings
# class FeatureData(Base):
#     __tablename__ = "feature_data"
    
#     id = Column(Integer, primary_key=True, index=True)
#     # name = Column(String, index=True)
#     face_embedding = Column(LargeBinary)  # Store the face embedding as binary data

# Base.metadata.create_all(bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()



from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database connection
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://ec_user:user%4088@3.110.123.38:3306/ecommerce_db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Predictions table
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String(255), index=True)
    person_id = Column(Integer, index=True)
    confidence = Column(Float)
    predicted_class = Column(String(255), index=True)
    email = Column(String(255), nullable=True)  # Ensure email is here


# FeatureData table for embeddings
class FeatureData(Base):
    __tablename__ = "feature_data"
    id = Column(Integer, primary_key=True, index=True)
    face_embedding = Column(LargeBinary)  # Binary data for face embeddings

# Create tables if they don't exist

Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

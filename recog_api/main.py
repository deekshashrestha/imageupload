from fastapi import FastAPI
from img_preprocessing import router as image_processing_router

app = FastAPI()

# Include the image processing router
app.include_router(image_processing_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


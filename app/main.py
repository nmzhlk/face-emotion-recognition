from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Face Emotion Recognition API")

@app.get("/")
def root():
    return {"message": "Face Emotion Recognition API is running"}

from admin_panel.auth import router as admin_auth_router
from admin_panel.routes import router as admin_routes

app.include_router(admin_auth_router)
app.include_router(admin_routes)
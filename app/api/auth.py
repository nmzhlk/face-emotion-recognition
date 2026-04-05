import uuid

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    """Stub: always returns success with a generated UUID."""
    return {
        "user_id": str(uuid.uuid4()),
        "username": body.username,
        "message": "Registration successful",
    }


@router.post("/login")
async def login(body: LoginRequest):
    """Stub: always returns user_id=0 with a stub token."""
    return {
        "user_id": 0,
        "access_token": "stub-token",
    }

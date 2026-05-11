from pydantic import BaseModel


class AuthRequest(BaseModel):
    user: str
    password: str

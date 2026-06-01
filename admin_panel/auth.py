from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.db.db import get_connection

router = APIRouter(prefix="/admin/auth", tags=["admin_auth"])
templates = Jinja2Templates(directory="admin_panel/templates")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

@router.get("/login")
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
async def login(request: Request, response: Response, username: str, password: str):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT ID, USERNAME, PASSWORD_HASH FROM USERS WHERE USERNAME = :1 AND IS_ADMIN = 1", [username])
        admin = cur.fetchone()
        if not admin or not verify_password(password, admin[2]):
            return templates.TemplateResponse("login.html", {"request": request, "error": "Неверные данные"})
        response = RedirectResponse(url="/admin/dashboard", status_code=303)
        response.set_cookie(key="admin_id", value=str(admin[0]), httponly=True)
        return response
    finally:
        cur.close()
        conn.close()

@router.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/admin/auth/login", status_code=303)
    response.delete_cookie("admin_id")
    return response
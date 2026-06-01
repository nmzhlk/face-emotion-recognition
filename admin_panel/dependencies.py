from fastapi import Request
from fastapi.responses import RedirectResponse
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.db.db import get_connection

async def get_current_admin(request: Request):
    admin_id = request.cookies.get("admin_id")
    if not admin_id:
        return RedirectResponse(url="/admin/auth/login", status_code=303)
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT ID, USERNAME FROM USERS WHERE ID = :1 AND IS_ADMIN = 1", [int(admin_id)])
        admin = cur.fetchone()
        if not admin:
            return RedirectResponse(url="/admin/auth/login", status_code=303)
        return {"id": admin[0], "username": admin[1]}
    finally:
        cur.close()
        conn.close()
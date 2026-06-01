from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.templating import Jinja2Templates
from .dependencies import get_current_admin
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.api.db.db import get_connection

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="admin_panel/templates")

@router.get("/dashboard")
async def dashboard(request: Request, admin=Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM USERS")
        total_users = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM UPLOADED_IMAGES")
        total_uploads = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM FACE_DETECTIONS")
        total_detections = cur.fetchone()[0]

        cur.execute("""
            SELECT EMOTION_CODE, COUNT(*) 
            FROM FACE_DETECTIONS 
            WHERE CREATED_AT >= SYSDATE - 30 
            GROUP BY EMOTION_CODE 
            ORDER BY COUNT(*) DESC
        """)
        emotion_stats = cur.fetchall()
        emotions_labels = [row[0] for row in emotion_stats]
        emotions_counts = [row[1] for row in emotion_stats]

        return templates.TemplateResponse("dashboard.html", {
            "request": request, "admin": admin,
            "total_users": total_users, "total_uploads": total_uploads,
            "total_detections": total_detections,
            "emotions_labels": emotions_labels, "emotions_counts": emotions_counts
        })
    finally:
        cur.close()
        conn.close()

@router.get("/users")
async def list_users(request: Request, admin=Depends(get_current_admin), search: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    try:
        if search:
            cur.execute("""
                SELECT ID, UUID, USERNAME, EMAIL, CREATED_AT, IS_ADMIN 
                FROM USERS 
                WHERE LOWER(USERNAME) LIKE :1 OR LOWER(EMAIL) LIKE :1 
                ORDER BY ID
            """, [f"%{search.lower()}%"])
        else:
            cur.execute("SELECT ID, UUID, USERNAME, EMAIL, CREATED_AT, IS_ADMIN FROM USERS ORDER BY ID")
        users = cur.fetchall()
        return templates.TemplateResponse("users.html", {"request": request, "admin": admin, "users": users, "search": search})
    finally:
        cur.close()
        conn.close()

@router.post("/users/{user_id}/toggle-admin")
async def toggle_admin(user_id: int, admin=Depends(get_current_admin)):
    if admin["id"] == user_id:
        raise HTTPException(403, "Нельзя изменить свой статус")
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("UPDATE USERS SET IS_ADMIN = CASE WHEN IS_ADMIN = 1 THEN 0 ELSE 1 END WHERE ID = :1", [user_id])
        conn.commit()
        return {"message": "Статус изменён"}
    finally:
        cur.close()
        conn.close()

@router.post("/users/{user_id}/delete")
async def delete_user(user_id: int, admin=Depends(get_current_admin)):
    if admin["id"] == user_id:
        raise HTTPException(403, "Нельзя удалить себя")
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM USERS WHERE ID = :1", [user_id])
        conn.commit()
        return {"message": "Пользователь удалён"}
    finally:
        cur.close()
        conn.close()

@router.get("/photos")
async def list_photos(request: Request, admin=Depends(get_current_admin), emotion: str = "", user_id: int = 0):
    conn = get_connection()
    cur = conn.cursor()
    try:
        query = """
            SELECT ui.ID, ui.ORIGINAL_FILENAME, ui.CREATED_AT, u.USERNAME, u.ID as USER_ID,
                   (SELECT COUNT(*) FROM FACE_DETECTIONS WHERE SOURCE_PHOTO_ID = ui.UUID) as FACES_COUNT
            FROM UPLOADED_IMAGES ui
            JOIN USERS u ON ui.USER_ID = u.UUID
            WHERE 1=1
        """
        params = []
        if emotion:
            query += " AND EXISTS (SELECT 1 FROM FACE_DETECTIONS fd WHERE fd.SOURCE_PHOTO_ID = ui.UUID AND fd.EMOTION_CODE = :1)"
            params.append(emotion)
        if user_id:
            query += " AND u.ID = :2"
            params.append(user_id)
        query += " ORDER BY ui.CREATED_AT DESC"
        cur.execute(query, params)
        photos = cur.fetchall()
        
        cur.execute("SELECT DISTINCT EMOTION_CODE FROM FACE_DETECTIONS WHERE EMOTION_CODE IS NOT NULL")
        emotions_list = [row[0] for row in cur.fetchall()]
        
        return templates.TemplateResponse("photos.html", {
            "request": request, "admin": admin, "photos": photos,
            "emotions_list": emotions_list, "selected_emotion": emotion, "selected_user_id": user_id
        })
    finally:
        cur.close()
        conn.close()

@router.post("/photos/{photo_id}/delete")
async def delete_photo(photo_id: int, admin=Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT UUID FROM UPLOADED_IMAGES WHERE ID = :1", [photo_id])
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Фото не найдено")
        uuid = row[0]
        cur.execute("DELETE FROM FACE_DETECTIONS WHERE SOURCE_PHOTO_ID = :1", [uuid])
        cur.execute("DELETE FROM UPLOADED_IMAGES WHERE ID = :1", [photo_id])
        conn.commit()
        return {"message": "Удалено"}
    finally:
        cur.close()
        conn.close()

@router.get("/detections")
async def list_detections(request: Request, admin=Depends(get_current_admin), emotion: str = "", user_id: int = 0):
    conn = get_connection()
    cur = conn.cursor()
    try:
        query = """
            SELECT fd.ID, fd.UUID, fd.EMOTION_CODE, fd.CONFIDENCE, fd.CREATED_AT,
                   ui.ORIGINAL_FILENAME, u.USERNAME, u.ID as USER_ID
            FROM FACE_DETECTIONS fd
            JOIN UPLOADED_IMAGES ui ON fd.SOURCE_PHOTO_ID = ui.UUID
            JOIN USERS u ON ui.USER_ID = u.UUID
            WHERE 1=1
        """
        params = []
        if emotion:
            query += " AND fd.EMOTION_CODE = :1"
            params.append(emotion)
        if user_id:
            query += " AND u.ID = :2"
            params.append(user_id)
        query += " ORDER BY fd.CREATED_AT DESC"
        cur.execute(query, params)
        detections = cur.fetchall()
        
        cur.execute("SELECT DISTINCT EMOTION_CODE FROM FACE_DETECTIONS WHERE EMOTION_CODE IS NOT NULL")
        emotions_list = [row[0] for row in cur.fetchall()]
        
        return templates.TemplateResponse("detections.html", {
            "request": request, "admin": admin, "detections": detections,
            "emotions_list": emotions_list, "selected_emotion": emotion, "selected_user_id": user_id
        })
    finally:
        cur.close()
        conn.close()

@router.post("/detections/{detection_id}/delete")
async def delete_detection(detection_id: int, admin=Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM FACE_DETECTIONS WHERE ID = :1", [detection_id])
        conn.commit()
        return {"message": "Детекция удалена"}
    finally:
        cur.close()
        conn.close()

@router.get("/validate")
async def validate_page(request: Request, admin=Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT fd.ID, fd.UUID, fd.EMOTION_CODE, fd.CONFIDENCE, ui.ORIGINAL_FILENAME, u.USERNAME
            FROM FACE_DETECTIONS fd
            JOIN UPLOADED_IMAGES ui ON fd.SOURCE_PHOTO_ID = ui.UUID
            JOIN USERS u ON ui.USER_ID = u.UUID
            WHERE (fd.CONFIDENCE < 0.7 OR fd.VALIDATED IS NULL)
            ORDER BY DBMS_RANDOM.VALUE
            FETCH FIRST 1 ROW ONLY
        """)
        detection = cur.fetchone()
        if not detection:
            return templates.TemplateResponse("validate.html", {"request": request, "admin": admin, "detection": None})
        return templates.TemplateResponse("validate.html", {
            "request": request, "admin": admin,
            "detection": {
                "id": detection[0], "uuid": detection[1], "emotion": detection[2],
                "confidence": detection[3], "filename": detection[4], "username": detection[5]
            }
        })
    finally:
        cur.close()
        conn.close()

@router.post("/validate/{detection_id}")
async def validate_detection(detection_id: int, is_correct: bool, admin=Depends(get_current_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE FACE_DETECTIONS 
            SET VALIDATED = :1, VALIDATED_BY = :2, VALIDATED_AT = SYSDATE 
            WHERE ID = :3
        """, [1 if is_correct else 0, admin["id"], detection_id])
        conn.commit()
        return {"message": "Сохранено"}
    finally:
        cur.close()
        conn.close()
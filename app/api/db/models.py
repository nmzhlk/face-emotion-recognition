from .db import get_connection

TABLES = {
    "USERS": """
        CREATE TABLE USERS (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) NOT NULL UNIQUE,
            EMAIL VARCHAR2(255) NOT NULL,
            USERNAME VARCHAR2(255) NOT NULL,
            PASSWORD_HASH VARCHAR2(255) NOT NULL,
            FIRST_NAME VARCHAR2(255) NOT NULL,
            LAST_NAME VARCHAR2(255) NOT NULL,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "HUMANS": """
        CREATE TABLE HUMANS (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) NOT NULL UNIQUE,
            USER_ID VARCHAR2(36),
            FIRST_NAME VARCHAR2(255),
            LAST_NAME VARCHAR2(255),
            KNOWN_FACE_URL VARCHAR2(500),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT HUMANS_USER_FK FOREIGN KEY (USER_ID) REFERENCES USERS(UUID) ON DELETE CASCADE
        )
    """,
    "UPLOADED_IMAGES": """
        CREATE TABLE UPLOADED_IMAGES (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) NOT NULL UNIQUE,
            USER_ID VARCHAR2(36) NOT NULL,
            IMAGE_URL VARCHAR2(1000) NOT NULL,
            ORIGINAL_FILENAME VARCHAR2(255),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            STATUS_CODE VARCHAR2(50),
            CONSTRAINT UPLOADED_IMAGES_USER_FK FOREIGN KEY (USER_ID) REFERENCES USERS(UUID) ON DELETE CASCADE
        )
    """,
    "USER_PHOTOS": """
        CREATE TABLE USER_PHOTOS (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) NOT NULL UNIQUE,
            USER_ID VARCHAR2(36) NOT NULL,
            HUMAN_ID VARCHAR2(36) NOT NULL,
            PHOTO_URL VARCHAR2(1000) NOT NULL,
            YOLO_FACE_BBOX CLOB,
            IS_PRIMARY NUMBER(1) DEFAULT 0 CHECK (IS_PRIMARY IN (0, 1)),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            STATUS_CODE VARCHAR2(50),
            CONSTRAINT USER_PHOTOS_USER_FK FOREIGN KEY (USER_ID) REFERENCES USERS(UUID) ON DELETE CASCADE,
            CONSTRAINT USER_PHOTOS_HUMAN_FK FOREIGN KEY (HUMAN_ID) REFERENCES HUMANS(UUID) ON DELETE CASCADE
        )
    """,
    "FACE_DETECTIONS": """
        CREATE TABLE FACE_DETECTIONS (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) NOT NULL UNIQUE,
            SOURCE_PHOTO_ID VARCHAR2(36) NOT NULL,
            DETECTED_HUMAN_ID VARCHAR2(36),
            DETECTED_BBOX CLOB,
            CONFIDENCE NUMBER(5,4) NOT NULL,
            EMOTION_CODE VARCHAR2(50),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT FACE_DET_PHOTO_FK FOREIGN KEY (SOURCE_PHOTO_ID) REFERENCES UPLOADED_IMAGES(UUID) ON DELETE CASCADE
        )
    """,
}

INDEXES = [
    "CREATE INDEX FACE_DET_PHOTO_IDX ON FACE_DETECTIONS(SOURCE_PHOTO_ID)",
    "CREATE INDEX FACE_DET_HUMAN_IDX ON FACE_DETECTIONS(DETECTED_HUMAN_ID)",
    "CREATE INDEX UPLOADED_IMG_USER_IDX ON UPLOADED_IMAGES(USER_ID)",
    "CREATE INDEX UPLOADED_IMG_STATUS_IDX ON UPLOADED_IMAGES(STATUS_CODE)",
]


def create_tables() -> None:
    conn = get_connection()
    cur = conn.cursor()
    for table_name, sql in TABLES.items():
        try:
            cur.execute(sql)
            print(f"Table {table_name} created successfully.")
        except Exception as e:
            if "already exists" in str(e).lower() or "ORA-00955" in str(e):
                print(f"Table {table_name} already exists.")
            else:
                print(f"Error creating {table_name}: {e}")
    conn.commit()
    cur.close()
    conn.close()


def seed_data() -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM USERS WHERE USERNAME = 'admin'")
        if cur.fetchone()[0] == 0:
            cur.execute(
                """
                INSERT INTO USERS (UUID, EMAIL, USERNAME, PASSWORD_HASH, FIRST_NAME, LAST_NAME)
                 VALUES (:1, :2, :3, :4, :5, :6)
            """,
                (
                    "admin-uuid-001",
                    "admin@mail.ru",
                    "admin",
                    "none",
                    "Admin",
                    "Adminovich",
                ),
            )
            conn.commit()
            print("[DB] Admin user created successfully.")
        else:
            print("[DB] Admin user already exists. Skipping.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    create_tables()
    seed_data()

from .db import get_connection

TABLES = {
    "USERS": """
        CREATE TABLE USERS (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) UNIQUE,
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
            UUID VARCHAR2(36) UNIQUE,
            USER_ID VARCHAR2(36),
            FIRST_NAME VARCHAR2(255),
            LAST_NAME VARCHAR2(255),
            KNOWN_FACE_URL VARCHAR2(500),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "UPLOADED_IMAGES": """
        CREATE TABLE UPLOADED_IMAGES (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) UNIQUE,
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
            UUID VARCHAR2(36) UNIQUE,
            USER_ID VARCHAR2(36) NOT NULL,
            HUMAN_ID VARCHAR2(36) NOT NULL,
            PHOTO_URL VARCHAR2(1000) NOT NULL,
            YOLO_FACE_BBOX CLOB,
            IS_PRIMARY NUMBER(1) DEFAULT 0 CHECK (IS_PRIMARY IN (0, 1)),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            STATUS_CODE VARCHAR2(50)
        )
    """,
    "FACE_DETECTIONS": """
        CREATE TABLE FACE_DETECTIONS (
            ID NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            UUID VARCHAR2(36) UNIQUE,
            UPLOADED_IMAGE_ID VARCHAR2(36) NOT NULL,
            DETECTED_HUMAN_ID VARCHAR2(36) NOT NULL,
            SOURCE_PHOTO_ID VARCHAR2(36) NOT NULL,
            DETECTED_BBOX CLOB,
            CONFIDENCE NUMBER(5,4) NOT NULL,
            EMOTION_CODE VARCHAR2(50),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT FACE_DETECTIONS_UPLOADED_FK FOREIGN KEY (UPLOADED_IMAGE_ID) REFERENCES UPLOADED_IMAGES(UUID) ON DELETE CASCADE,
            CONSTRAINT FACE_DETECTIONS_HUMAN_FK FOREIGN KEY (DETECTED_HUMAN_ID) REFERENCES HUMANS(UUID) ON DELETE CASCADE,
            CONSTRAINT FACE_DETECTIONS_SOURCE_PHOTO_FK FOREIGN KEY (SOURCE_PHOTO_ID) REFERENCES USER_PHOTOS(UUID) ON DELETE CASCADE
        )
    """,
}

INDEXES = [
    "CREATE INDEX FACE_DETECTIONS_UPLOADED_IDX ON FACE_DETECTIONS(UPLOADED_IMAGE_ID)",
    "CREATE INDEX FACE_DETECTIONS_HUMAN_IDX ON FACE_DETECTIONS(DETECTED_HUMAN_ID)",
    "CREATE INDEX FACE_DETECTIONS_SOURCE_PHOTO_IDX ON FACE_DETECTIONS(SOURCE_PHOTO_ID)",
    "CREATE INDEX UPLOADED_IMAGES_USER_IDX ON UPLOADED_IMAGES(USER_ID)",
    "CREATE INDEX UPLOADED_IMAGES_STATUS_IDX ON UPLOADED_IMAGES(STATUS_CODE)",
]


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    for table_name, sql in TABLES.items():
        try:
            cur.execute(
                f"BEGIN EXECUTE IMMEDIATE '{sql}'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
            )
            print(f"Created/checked {table_name} table.")
        except Exception as e:
            print(f"Could not create table {table_name}: {e}")

    for sql in INDEXES:
        try:
            cur.execute(
                f"BEGIN EXECUTE IMMEDIATE '{sql}'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -955 THEN RAISE; END IF; END;"
            )
        except Exception as e:
            print(f"Could not create index: {e}")

    conn.commit()
    cur.close()
    conn.close()


def drop_all_tables() -> None:
    conn = get_connection()
    cur = conn.cursor()
    for table in reversed(list(TABLES.keys())):
        try:
            cur.execute(f"DROP TABLE {table} CASCADE CONSTRAINTS")
            print(f"Deleted {table} table.")
        except Exception as e:
            print(f"Could not delete table {table}: {e}")
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    init_db()

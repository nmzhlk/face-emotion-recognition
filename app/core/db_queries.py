import json
from datetime import datetime
from typing import Any, Dict, List

from app.core.database import get_pool


async def create_tables() -> None:
    pool = get_pool()
    queries = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(36) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL,
            username VARCHAR(255) NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(255) NOT NULL,
            last_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS humans (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(36) NOT NULL UNIQUE,
            user_id VARCHAR(36),
            first_name VARCHAR(255),
            last_name VARCHAR(255),
            known_face_url VARCHAR(500),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT humans_user_fk FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS uploaded_images (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(36) NOT NULL UNIQUE,
            user_id VARCHAR(36) NOT NULL,
            image_url VARCHAR(1000) NOT NULL,
            original_filename VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            status_code VARCHAR(50),
            CONSTRAINT uploaded_images_user_fk FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_photos (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(36) NOT NULL UNIQUE,
            user_id VARCHAR(36) NOT NULL,
            human_id VARCHAR(36) NOT NULL,
            photo_url VARCHAR(1000) NOT NULL,
            yolo_face_bbox TEXT,
            is_primary SMALLINT DEFAULT 0 CHECK (is_primary IN (0,1)),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            status_code VARCHAR(50),
            CONSTRAINT user_photos_user_fk FOREIGN KEY (user_id) REFERENCES users(uuid) ON DELETE CASCADE,
            CONSTRAINT user_photos_human_fk FOREIGN KEY (human_id) REFERENCES humans(uuid) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS face_detections (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(36) NOT NULL UNIQUE,
            source_photo_id VARCHAR(36) NOT NULL,
            detected_human_id VARCHAR(36),
            detected_bbox TEXT,
            confidence NUMERIC(5,4) NOT NULL,
            emotion_code VARCHAR(50),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT face_det_photo_fk FOREIGN KEY (source_photo_id) REFERENCES uploaded_images(uuid) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS processed_frames (
            id SERIAL PRIMARY KEY,
            edge_id VARCHAR(255) NOT NULL,
            camera_id VARCHAR(255) NOT NULL,
            frame_id UUID NOT NULL,
            timestamp BIGINT NOT NULL,
            store_path VARCHAR(1000) NOT NULL,
            processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            items JSONB NOT NULL,
            UNIQUE(edge_id, camera_id, frame_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS ingest_batches (
            id SERIAL PRIMARY KEY,
            batch_id UUID NOT NULL UNIQUE,
            edge_id VARCHAR(255) NOT NULL,
            camera_ids JSONB,
            processed_count INTEGER,
            received_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """,
    ]
    for q in queries:
        await pool.execute(q)

    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_frames_edge_camera ON processed_frames(edge_id, camera_id)",
        "CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON processed_frames(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_batches_edge ON ingest_batches(edge_id)",
        "CREATE INDEX IF NOT EXISTS idx_face_det_source ON face_detections(source_photo_id)",
        "CREATE INDEX IF NOT EXISTS idx_face_det_human ON face_detections(detected_human_id)",
        "CREATE INDEX IF NOT EXISTS idx_uploaded_images_user ON uploaded_images(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_uploaded_images_status ON uploaded_images(status_code)",
    ]
    for q in index_queries:
        await pool.execute(q)


async def seed_admin_user() -> None:
    pool = get_pool()
    row = await pool.fetchval(
        "SELECT COUNT(*) FROM users WHERE username = 'admin'"
    )
    if row == 0:
        await pool.execute(
            """
            INSERT INTO users (uuid, email, username, password_hash, first_name, last_name)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            "admin-uuid-001",
            "admin@mail.ru",
            "admin",
            "none",
            "Admin",
            "Adminovich",
        )
        print("[DB] Admin user created successfully.")
    else:
        print("[DB] Admin user already exists. Skipping.")


async def save_batch_to_db(
    batch_id: str,
    edge_id: str,
    camera_ids: List[str],
    processed_count: int,
    received_at: datetime,
) -> None:
    pool = get_pool()
    await pool.execute(
        """
        INSERT INTO ingest_batches (batch_id, edge_id, camera_ids, processed_count, received_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        batch_id,
        edge_id,
        json.dumps(camera_ids),
        processed_count,
        received_at,
    )


async def save_frames_to_db(frames: List[Dict[str, Any]]) -> None:
    if not frames:
        return
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            for frame in frames:
                await conn.execute(
                    """
                    INSERT INTO processed_frames (edge_id, camera_id, frame_id, timestamp, store_path, items)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (edge_id, camera_id, frame_id) DO UPDATE
                    SET items = EXCLUDED.items, processed_at = CURRENT_TIMESTAMP
                    """,
                    frame["edge_id"],
                    frame["camera_id"],
                    frame["frame_id"],
                    frame["timestamp"],
                    frame.get("store_path", ""),
                    json.dumps(frame.get("items", [])),
                )


async def delete_frame_from_db(
    edge_id: str, camera_id: str, frame_id: str
) -> None:
    pool = get_pool()
    await pool.execute(
        "DELETE FROM processed_frames WHERE edge_id=$1 AND camera_id=$2 AND frame_id=$3",
        edge_id,
        camera_id,
        frame_id,
    )


async def delete_stream_from_db(edge_id: str, camera_id: str) -> int:
    pool = get_pool()
    result = await pool.execute(
        "DELETE FROM processed_frames WHERE edge_id=$1 AND camera_id=$2",
        edge_id,
        camera_id,
    )
    try:
        deleted = int(result.split()[1])
    except Exception:
        deleted = 0
    return deleted


async def delete_edge_from_db(edge_id: str) -> int:
    pool = get_pool()
    result = await pool.execute(
        "DELETE FROM processed_frames WHERE edge_id=$1", edge_id
    )
    try:
        deleted = int(result.split()[1])
    except Exception:
        deleted = 0
    return deleted

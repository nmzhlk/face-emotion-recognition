from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StreamPayload(BaseModel):
    user_id: str
    stream_id: str
    frame_id: str
    timestamp: int
    image: bytes


class MergedItem(BaseModel):
    bbox: List[int]
    identity: str
    identity_confidence: float
    embedding: Any
    emotion: str
    emotion_confidence: float


class ProcessedFrameData(BaseModel):
    user_id: str
    stream_id: str
    frame_id: str
    timestamp: int
    items: List[Dict[str, Any]]


class StreamResponse(BaseModel):
    status: int
    task_id: str
    message: Optional[str] = None


class ETLReturnResult(BaseModel):
    user_id: str
    stream_id: str
    frame_id: str
    timestamp: int
    store_path: str
    items: Optional[List[MergedItem]] = None

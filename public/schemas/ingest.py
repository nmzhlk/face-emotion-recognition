from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FrameResult(BaseModel):
    edge_id: str
    camera_id: str
    frame_id: str
    timestamp: int

    items: List[Dict[str, Any]] = Field(default_factory=list)


class IngestBatchRequest(BaseModel):
    edge_id: str
    batch_id: str

    camera_ids: List[str] = Field(default_factory=list)
    processed_count: int

    received_at: Optional[datetime] = None

    frames: List[FrameResult]

    def model_post_init(self, __context: Any) -> None:
        if self.received_at is None:
            return


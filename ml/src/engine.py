from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as t
from numpy.typing import NDArray
from PIL import Image
from ultralytics import YOLO

from .recognizer import FaceRecognizer
from .resnet import EMOTION_LABELS, get_resnet_emotion_model


class EmotionEngine:
    def __init__(
        self,
        yolo_path: str,
        resnet_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.detector = YOLO(yolo_path)

        self.emotion_model = get_resnet_emotion_model(
            output_shape=len(EMOTION_LABELS),
            load_path=resnet_path,
        )
        self.emotion_model.to(self.device)
        self.emotion_model.eval()

        self.recognizer = FaceRecognizer()

        self.transforms = t.Compose(
            [
                t.ToImage(),
                t.ToDtype(torch.float32, scale=True),
                t.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    @torch.inference_mode()
    def process_image(
        self, image_bytes: bytes
    ) -> Tuple[NDArray[Any], Dict[str, Any]]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        results = self.detector(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.25, verbose=False
        )

        faces_results: List[Dict[str, Any]] = []

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_roi = img[y1:y2, x1:x2]

                if face_roi.size == 0:
                    continue

                face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (70, 84))
                face_tensor = (
                    self.transforms(Image.fromarray(face_gray))
                    .unsqueeze(0)
                    .to(self.device)
                )

                emotion_logits = self.emotion_model(face_tensor)
                probs = torch.softmax(emotion_logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()

                emotion_name = EMOTION_LABELS[int(pred_idx)]
                emotion_conf = float(probs[0][int(pred_idx)].item())

                human_uuid, identity_conf, embedding = (
                    self.recognizer.process_face(face_roi)
                )

                color = (247, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{emotion_name} {emotion_conf:.2f}"
                cv2.putText(
                    img,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                faces_results.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "emotion": emotion_name,
                        "emotion_confidence": emotion_conf,
                        "identity": human_uuid,
                        "identity_confidence": identity_conf,
                        "embedding": (
                            embedding.tolist()
                            if embedding is not None
                            else None
                        ),
                    }
                )

        return img, {"faces_count": len(faces_results), "faces": faces_results}

from ultralytics.models.sam import SAM
from detectors.obj_detector import Object_Detector
import torch
import numpy as np


class Sam2_Detector(Object_Detector):
    def __init__(
        self,
        to_tensor=False,
        device="cuda",
        model="models/sam2_b.pt",
        inference=False,
        use_n_frame=1,
        imgsz=(256, 128),
        **kwargs
    ) -> None:
        super().__init__(to_tensor, device)
        self.model = SAM(model)
        self.inference = inference
        self.use_n_frame = use_n_frame
        self.prediction = None
        self.imgsz = imgsz
        self.kwargs = kwargs

    def predict(self, input, **kwargs):
        super().predict(input)
        if self.inference and self.prediction:
            self.predict_inference(input, imgsz=self.imgsz, **kwargs)
        else:
            self.prediction = self.model.predict(input, imgsz=self.imgsz, **self.kwargs)

    def predict_inference(self, input, **kwargs):
        self.prediction = self.model.predict(
            input, bboxes=self.prediction[-1].boxes.xyxy, **kwargs
        )
        self.input = input

    def get_box_feature(self):
        p = self.prediction[-1]  # type: ignore
        num_boxes = p.boxes.shape[0]

        features = torch.zeros(p.orig_shape + (num_boxes,), dtype=torch.int32)

        b = self.prediction[0].boxes.xyxy.int()

        for i in range(num_boxes):
            box = b[i]
            features[box[1] : box[3], box[0] : box[2], i] = 1

        if not self.to_tensor:
            features = features.cpu().numpy()
        return features

    def get_mask_feature(self):
        p = self.prediction[-1].masks.data.to(self.device).permute((1, 2, 0))
        if not self.to_tensor:
            p = p.cpu().numpy()
        return p

    def get_bbox(self):
        return self.prediction[-1].boxes.xyxy

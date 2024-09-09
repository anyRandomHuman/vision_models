from ultralytics.models.sam import SAM
from detectors.obj_detector import Object_Detector
import torch
import numpy as np
class Sam2_Detector(Object_Detector):
    def __init__(self, to_tensor=False, device="cuda", model='') -> None:
        super().__init__(to_tensor, device)
        self.model = SAM(model)

    def predict(self, img):
        super().predict(img)
        self.prediction = self.model.predict(img)
        

        def get_box_feature(self):
        p = self.prediction[0]  # type: ignore
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
        p = self.prediction[0].masks.data.to(self.device).permute((1, 2, 0))
        if not self.to_tensor:
            p = p.cpu().numpy()
        return p
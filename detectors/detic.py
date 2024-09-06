import numpy as np
import cv2

from detectors.obj_detector import Object_Detector

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch

# import sys
# sys.path.insert(0, '../third_party/CenterNet2/')
from third_party.CenterNet2.centernet.config import add_centernet_config
# from ...centernet.config import add_centernet_config
from third_party.Detic.detic.config import add_detic_config
from third_party.Detic.detic.modeling.utils import reset_cls_test
cup_pred_class = 41
stacked_cups_class = 39
BOX_FEATURE = "box"
MASK_FEATURE = "mask"


class Detectron(Object_Detector):
    def __init__(
        self,path='', to_tensor=False,
        device="cuda",
    ):
        super().__init__(path=path, to_tensor=to_tensor, device=device)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        super().predict(img)
        self.results = self.predictor(img)

    def get_box_feature(self):
        instances = self.results["instances"]

        box_bounds = instances.pred_boxes.tensor
        features = torch.zeros(self.img.shape[:2] + (len(instances),)).to(self.device)
        for i in range(len(instances)):
            t = box_bounds[i].to(dtype=torch.long)
            features[t[1] : t[3], t[0] : t[2], i] = 1

        return features

    def get_mask_feature(self):
        instances = self.results["instances"]
        features = instances.pred_masks.permute((1,2,0))
        if not self.to_tensor:
            features = features.cpu().numpy()
        else:
            features = features.to(self.device)
        return features




import numpy as np
import cv2
import sys
from obj_detector import Object_Detector
sys.path.insert(0, 'Detectron2')

from Detectron2.detectron2 import model_zoo
from Detectron2.detectron2.engine import DefaultPredictor
from Detectron2.detectron2.config import get_cfg
from Detectron2.detectron2.utils.visualizer import Visualizer
from Detectron2.detectron2.data import MetadataCatalog
import torch


sys.path.insert(0, 'Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test
cup_pred_class = 41
stacked_cups_class = 39
BOX_FEATURE = "box"
MASK_FEATURE = "mask"


class Detectron(Object_Detector):
    def __init__(
        self,  obj_classes=[cup_pred_class, stacked_cups_class], to_tensor=False,
        device="cuda",
    ):
        super.__init__(to_tensor, device)
        self.cfg = get_cfg()
        self.obj_classes = obj_classes
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
        self.img = img
        self.results = self.predictor(img)

    def get_box_feature(self):
        instances = self.results["instances"]

        box_bounds = instances.pred_boxes.tensor
        features = torch.zeros((len(instances),) + self.img.shape[:2]).cuda()
        for i in range(len(instances)):
            t = box_bounds[i].to(dtype=torch.long)
            features[i, t[1] : t[3], t[0] : t[2]] = 1

        return features

    def get_mask_feature(self):
        instances = self.results["instances"]
        features = instances.pred_masks
        return features




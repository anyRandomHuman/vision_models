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
from third_party.Detic.detic.modeling.text.text_encoder import build_text_encoder

cup_pred_class = 41
stacked_cups_class = 39
BOX_FEATURE = "box"
MASK_FEATURE = "mask"


class Detectron(Object_Detector):
    def __init__(
        self,
        path="https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
        to_merge="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
        to_tensor=False,
        device="cuda",
        classes=None,
        to_detect = None
    ):
        super().__init__(to_tensor=to_tensor, device=device)
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(to_merge)
        cfg.MODEL.WEIGHTS = path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
            True  # For better visualization purpose. Set to False for all classes.
        )
        # cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.

        self.predictor = DefaultPredictor(cfg)

        if classes:
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = (
                classes  # Change here to try your own vocabularies!
            )
            classifier = Detectron._get_clip_embeddings(self.metadata.thing_classes)
        else:
            BUILDIN_CLASSIFIER = {
                "lvis": "third_party/Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
                "objects365": "third_party/Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
                "openimages": "third_party/Detic/datasets/metadata/oid_clip_a+cname.npy",
                "coco": "third_party/Detic/datasets/metadata/coco_clip_a+cname.npy",
            }

            BUILDIN_METADATA_PATH = {
                "lvis": "lvis_v1_val",
                "objects365": "objects365_v2_val",
                "openimages": "oid_val_expanded",
                "coco": "coco_2017_val",
            }

            vocabulary = (
                "lvis"  # change to 'lvis', 'objects365', 'openimages', or 'coco'
            )
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
            classifier = BUILDIN_CLASSIFIER[vocabulary]
        num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)
        
        if to_detect:
            self.to_detect = list(map(lambda x: self.metadata.thing_classes.index(x), to_detect))
        else:
            self.to_detect = []
        
        self.detected_classes = set()

    def predict(self, img, **kwargs):
        super().predict(img)
        self.results = self.predictor(img)
        pred_classes = self.results['instances'].pred_classes.cpu().tolist()
        class_names = self.metadata.thing_classes
        pred_class_names = set(map(lambda x: class_names[x], pred_classes))
        
        self.detected_classes=self.detected_classes.union(pred_class_names)
        

    def get_box_feature(self):
        instances = self.results["instances"]

        box_bounds = instances.pred_boxes.tensor
        features = torch.zeros(self.input.shape[:2] + (len(instances),)).to(self.device)
        for i in range(len(instances)):
            t = box_bounds[i].to(dtype=torch.long)
            features[t[1] : t[3], t[0] : t[2], i] = 1

        return features

    def get_mask_feature(self):
        instances = self.results["instances"]
        
        if len(self.to_detect):
            features = torch.concatenate([instances[i].pred_masks for i in range(len(instances)) if instances[i].pred_classes[0] in self.to_detect], 0)
        else:
            features = instances.pred_masks
        features = features.permute((1, 2, 0))
        if not self.to_tensor:
            features = features.cpu().numpy()
        else:
            features = features.to(self.device)
        return features

    @staticmethod
    def _get_clip_embeddings(vocabulary, prompt="a "):
        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        texts = [prompt + x for x in vocabulary]
        emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def get_visualized_imgs(self):
        v = Visualizer(self.input, self.metadata)
        out = v.draw_instance_predictions(self.results["instances"].to("cpu"))

        print(
            [
                self.metadata.thing_classes[x]
                for x in self.results["instances"].pred_classes.cpu().tolist()
            ]
        )

        return out.get_image()

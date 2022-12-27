from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former.mask2former import add_maskformer2_config
from detectron2.engine.defaults import DefaultPredictor
import json

def get_colors():
    with open('mask2former/class.json','r') as f:
        j = json.load(f)
    return j['labels']

def get_predict_func():
    opts = ['MODEL.WEIGHTS', 'mask2former/model/model.pkl']
    config = 'mask2former/config/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml'
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config)
    cfg.merge_from_list(opts)
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    return predictor

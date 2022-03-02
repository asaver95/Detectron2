from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from utility.LossEvalHook import LossEvalHook
from detectron2.data import DatasetMapper
from detectron2.config import *
import os


cfg = get_cfg()

class MyTrainer(DefaultTrainer):
    #@classmethod
    #def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #    if output_folder is None:
    #        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #    return COCOEvaluator(dataset_name, cfg, True, output_folder)


     @classmethod
     def build_train_loader(cls, cfg):
         dataloader = build_detection_train_loader(cfg,
         mapper=DatasetMapper(cfg, is_train=True, augmentations=[
    #         T.Resize((1000,1000)),
         T.RandomFlip(0.5, horizontal=True, vertical=False),
         T.RandomFlip(0.5, horizontal=False, vertical=True),
         T.RandomRotation(angle=[-60,60], expand=True, center=None, sample_style='range', interp=None),
         T.RandomContrast(intensity_min=-1, intensity_max=2),
         ]))
         return dataloader   
        # use this dataloader instead of the default
        
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks

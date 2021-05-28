from ..builder import DETECTORS
from .single_stage_pc import SingleStageDetector_pc


@DETECTORS.register_module()
class FCOS_pc(SingleStageDetector_pc):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FCOS_pc, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

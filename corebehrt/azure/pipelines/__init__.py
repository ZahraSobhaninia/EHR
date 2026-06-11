from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from .FINETUNE import FINETUNE, create
from .eval_only import EVAL_ONLY
from .xgb_train_eval import XGB_TRAIN_EVAL

PIPELINE_REGISTRY = [E2E, FINETUNE, EVAL_ONLY, XGB_TRAIN_EVAL]

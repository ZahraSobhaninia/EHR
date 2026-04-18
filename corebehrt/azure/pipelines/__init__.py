from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
#from .Finetune_Holdout import FINETUNE_HOLDOUT, create
from .FINETUNE import FINETUNE, create
from .eval_only import EVAL_ONLY
#FINETUNE_HOLDOUT

PIPELINE_REGISTRY = [E2E, FINETUNE, EVAL_ONLY]

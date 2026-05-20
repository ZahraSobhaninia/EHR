from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from .FINETUNE import FINETUNE, create
from .eval_only import EVAL_ONLY


PIPELINE_REGISTRY = [E2E, FINETUNE, EVAL_ONLY]

from corebehrt.azure.pipelines.E2E import E2E
from corebehrt.azure.pipelines.FINETUNE import FINETUNE
from corebehrt.azure.pipelines.XGB_FROM_PREPARED import XGB_FROM_PREPARED
PIPELINE_REGISTRY = [E2E, FINETUNE,XGB_FROM_PREPARED]

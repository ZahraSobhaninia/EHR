"""
Pipeline for running XGBoost CV followed by evaluation
from already prepared data.
"""

from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg


XGB_FROM_PREPARED = PipelineMeta(
    name="XGB_TRAIN_EVAL",
    help="Run XGBoost CV + evaluation from prepared data.",
    inputs=[
        PipelineArg(
            name="prepared_data",
            help="Path to prepared training data (CV folds).",
            required=True,
        ),
        PipelineArg(
            name="test_data",
            help="Path to held-out test data.",
            required=True,
        ),
    ],
)


def create(component: callable):
    from azure.ai.ml import dsl, Input

    @dsl.pipeline(
        name="xgb_train_eval_pipeline",
        description="Train XGBoost with CV and evaluate on held-out data",
    )
    def pipeline(prepared_data: Input, test_data: Input):

        # ---------------------------
        # Step 1: XGBoost CV Training
        # ---------------------------
        xgboost = component(
            "xgboost_cv",
        )(
            prepared_data=prepared_data,
        )

        # ---------------------------
        # Step 2: Evaluation
        # ---------------------------
        evaluate = component(
            "evaluate_xgboost",
        )(
            model=xgboost.outputs,  # مطابق الگوی E2E_XGB
            folds_dir=prepared_data,
            test_data_dir=test_data,
        )

        return {
            "model": xgboost.outputs.model,
            "predictions": evaluate.outputs.predictions,
        }

    return pipeline
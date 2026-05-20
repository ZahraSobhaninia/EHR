from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

XGB_TRAIN_EVAL = PipelineMeta(
    name="xgb_train_eval",
    help="Train and evaluate XGBoost on prepared data",
    inputs=[
        PipelineArg("prepared_data", required=True, help="Path to prepared data"),
        PipelineArg("test_data_dir", required=True, help="Path to Test data"),
    ],
)


def create(component):

    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="xgb_train_eval_pipeline")
    def pipeline(
        prepared_data: Input,
        test_data_dir: Input,
    ):

        xgboost = component("xgboost_cv")(
            prepared_data=prepared_data,
        )

        evaluate = component("evaluate_xgboost")(
            model=xgboost.outputs.model,
            folds_dir=prepared_data,
            test_data_dir=test_data_dir,
        )

        return {
            "model": xgboost.outputs.model,
            "predictions": evaluate.outputs.predictions,
        }

    return pipeline
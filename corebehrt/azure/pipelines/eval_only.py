from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

EVAL_ONLY = PipelineMeta(
    name="eval_only",
    help="Evaluate model on held-out data",
    inputs=[
        PipelineArg("features", help="Path to features", required=True),
        PipelineArg("tokenized", help="Path to tokenized data", required=True),
        PipelineArg("outcomes", help="Train outcomes", required=True),
        PipelineArg("holdout_outcomes", help="Held-out outcomes", required=True),
        PipelineArg("model", help="Path to trained model", required=True),
    ],
)


def create(component):

    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="eval_only_pipeline")
    def pipeline(
        features: Input,
        tokenized: Input,
        outcomes: Input,
        holdout_outcomes: Input,
        model: Input,
    ):

        # train folds
        select_cohort = component("select_cohort")(
            features=features,
            outcomes=outcomes,
        )

        prepare_finetune = component("prepare_training_data")(
            features=features,
            tokenized=tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        # held-out
        select_heldout = component("select_cohort")(
            features=features,
            outcomes=holdout_outcomes,
        )

        prepare_heldout = component("prepare_training_data")(
            features=features,
            tokenized=tokenized,
            cohort=select_heldout.outputs.cohort,
            outcomes=holdout_outcomes,
        )

        evaluate = component("evaluate_finetune")(
            model=model,
            folds_dir=prepare_finetune.outputs.prepared_data,
            test_data_dir=prepare_heldout.outputs.prepared_data,
        )

        return {
            "predictions": evaluate.outputs.predictions,
        }

    return pipeline

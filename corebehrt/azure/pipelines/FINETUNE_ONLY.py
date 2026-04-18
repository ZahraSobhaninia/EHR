from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg
from azure.ai.ml import dsl, Input

EVAL_ONLY = PipelineMeta(
    name="eval_only",
    help="Evaluate model on held-out data",
    inputs=[
        PipelineArg("features", required=True),
        PipelineArg("tokenized", required=True),
        PipelineArg("model", required=True),
        PipelineArg("outcomes", required=True),
        PipelineArg("holdout_outcomes", required=True),
    ],
)

def create(component):

    @dsl.pipeline(
        name="eval_only_pipeline",
        description="Evaluate on held-out data",
    )
    def pipeline(
        features: Input,
        tokenized: Input,
        model: Input,
        outcomes: Input,
        holdout_outcomes: Input,
    ):

        # train cohort (برای folds_dir)
        select_cohort = component("select_cohort")(
            features=features,
            outcomes=outcomes,
        )

        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        # held-out cohort
        select_heldout = component(
            "select_cohort",
            name="select_heldout",
        )(
            features=features,
            outcomes=holdout_outcomes,
        )

        prepare_heldout = component(
            "prepare_training_data",
            name="prepare_heldout",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_heldout.outputs.cohort,
            outcomes=holdout_outcomes,
        )

        # evaluation
        evaluate = component("evaluate_finetune")(
            model=model,
            folds_dir=prepare_finetune.outputs.prepared_data,
            test_data_dir=prepare_heldout.outputs.prepared_data,
        )

        return {
            "predictions": evaluate.outputs.predictions,
        }

    return pipeline
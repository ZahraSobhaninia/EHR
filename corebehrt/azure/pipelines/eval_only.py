from corebehrt.azure.pipelines.base import PipelineMeta, PipelineArg

EVAL_ONLY = PipelineMeta(
    name="eval_only",
    help="Evaluate model on held-out data",
    inputs=[
        PipelineArg("features", help="Path to features", required=True),
        PipelineArg("tokenized", help="Path to tokenized data", required=True),
        PipelineArg("outcomes", help="Held-out outcomes", required=True),
        PipelineArg("model", help="Path to trained model", required=True),
        PipelineArg("folds_dir", help="Path to finetune data", required=True), 
    ],
)


def create(component):

    from azure.ai.ml import dsl, Input

    @dsl.pipeline(name="eval_only_pipeline")
    def pipeline(
        features: Input,
        tokenized: Input,
        outcomes: Input,
        model: Input,
        folds_dir: Input,
    ):

        #  STEP 1: SELECT HELD-OUT COHORT
        select_heldout = component("select_cohort",
        name="select_cohort_held_out",
        )(
            features=features,
            outcomes=outcomes,
        )


        #  STEP 2: PREPARE HELD-OUT DATA
        prepare_heldout = component("prepare_training_data",
        name="prepare_held_out",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_heldout.outputs.cohort,
            outcomes=outcomes,
        )

        #  STEP 3: EVALUATE
        evaluate = component("evaluate_finetune")(
            model=model,
            folds_dir=folds_dir,  
            test_data_dir=prepare_heldout.outputs.prepared_data,
        )

        return {
            "predictions": evaluate.outputs.predictions,
        }

    return pipeline
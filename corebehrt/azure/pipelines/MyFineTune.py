from azure.ai.ml import dsl, Input

def create_simple_finetune_pipeline(component):

    @dsl.pipeline(
        name="my_finetune_pipeline",
        description="Simple finetune pipeline (no holdout yet)",
    )
    def pipeline(
        features: Input,
        tokenized: Input,
        pretrain_model: Input,
        outcomes: Input,
    ):

        #  Select cohort
        select_cohort = component("select_cohort")(
            features=features,
            outcomes=outcomes,
        )

        #  Prepare finetune data
        prepare_finetune = component(
            "prepare_training_data",
            name="prepare_finetune",
        )(
            features=features,
            tokenized=tokenized,
            cohort=select_cohort.outputs.cohort,
            outcomes=outcomes,
        )

        #  Finetune
        finetune = component("finetune_cv")(
            prepared_data=prepare_finetune.outputs.prepared_data,
            pretrain_model=pretrain_model,
        )

        return {
            "model": finetune.outputs.model,
        }

    return pipeline


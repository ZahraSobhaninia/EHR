


python -m corebehrt.azure job prepare_training_data CPU-20-LP -e 1_PreparePretrain_MDP_WOP -c corebehrt/Myconfigs/prepare_pretrain.yaml

python -m corebehrt.azure job prepare_training_data CPU-20-LP -e 1_PreparePretrain_MDP_OOT -c corebehrt/Myconfigs/prepare_pretrain.yaml

python -m corebehrt.azure job pretrain GPU-dedicated  -e  3_PreTrain_MDP_OOT -c corebehrt/Myconfigs/pretrain.yaml

GPU-A100-Single

python -m corebehrt.azure job pretrain testHP -e val_test -c azure_configs/pretrain.yaml -lsm

python -m corebehrt.azure job create_outcomes CPU-20-LP -e kfold_test -c azure_configs/outcome_breast_cancer.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e kfold_test -c azure_configs/outcome_diabetes.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e kfold_test -c azure_configs/outcome_stroke.yaml
python -m corebehrt.azure job create_outcomes CPU-20-LP -e val_test -c azure_configs/outcome_synthetic.yaml

python -m corebehrt.azure job select_cohort CPU-20-LP -e 4_SelectCohort_MDP_OOT_Mortality -c corebehrt/Myconfigs/select_cohort.yaml


python -m corebehrt.azure job prepare_training_data CPU-20-LP -e 4_PrePare_Finetune_Mor_MDP_OOT -c corebehrt/Myconfigs/prepare_finetune.yaml


python -m corebehrt.azure job finetune_cv GPU-A100-Single -e kfold_test -c azure_configs/finetune.yaml

python -m corebehrt.azure job select_cohort CPU-20-LP -e val_test -c azure_configs/select_cohort.yaml
python -m corebehrt.azure job prepare_training_data CPU-20-LP -e val_test -c azure_configs/prepare_finetune.yaml
python -m corebehrt.azure job finetune_cv GPU-A100-small -e val_test -c azure_configs/finetune.yaml


python -m corebehrt.azure job xgboost_cv GPU-A100-small -e val_test -c azure_configs/xgboost.yaml
python -m corebehrt.azure job evaluate_xgboost GPU-A100-small -e val_test -c azure_configs/evaluate_xgboost.yaml



python -m corebehrt.azure job finetune_cv testHP -e bc_recreate -c azure_configs/finetune_bc.yaml


python -m corebehrt.azure job evaluate_finetune GPU-A100-Single -e bc_recreate -c azure_configs/evaluate_finetune.yaml
python -m corebehrt.azure job evaluate_finetune GPU-A100-Single -e bc_recreate -c azure_configs_bc_recreate_example/evaluate_finetune.yaml

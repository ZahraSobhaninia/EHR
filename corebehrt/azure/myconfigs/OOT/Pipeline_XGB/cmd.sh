python -m corebehrt.azure pipeline xgb_train_eval \
  --experiment "xgb-PRI-MDPS-CV" \
  --prepared_data "researcher_data:Zahra/062026/Corebehrt_CV/MDPS/2_Finetune/PRI/Data/Finetunedata" \
  --test_data_dir "researcher_data:Zahra/062026/Corebehrt_CV/MDPS/3_Holdout/PRI/Data/HeldoutData" \
  -cp xgboost_cv=GPU-test   \
  -cp evaluate_xgboost=GPU-test \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_Integragted/corebehrt/azure/myconfigs/CV/Pipeline_XGB



UNPREADM
SSSI
PNEUMONIA
  folds_dir: "researcher_data:Zahra/062026/CoreBehrt_OOT/MDPS/2_Finetune/UNPREADM/Data/Finetunedata"  
  test_data_dir:  "researcher_data:Zahra/062026/CoreBehrt_OOT/MDPS/3_Holdout/SSSI/Data/Holdout_Data"  
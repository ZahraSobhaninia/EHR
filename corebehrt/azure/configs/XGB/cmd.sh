python -m corebehrt.azure pipeline xgb_train_eval \
  --experiment "xgb-ARF-MDPS-OOT" \
  --prepared_data "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/ARF/Data/Finetunedata" \
  --test_data_dir "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/3_Holdout/ARF/Data/Holdout_Data" \
  -cp xgboost_cv=A100-dedicated   \
  -cp evaluate_xgboost=A100-dedicated \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/azure/configs/XGB

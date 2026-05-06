python -m corebehrt.azure pipeline eval_only \
  --experiment "eval-Mortality-MDPS-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/tokenized" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Holdout/Mortality" \
  --model "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/Mortality/Model" \
  --folds_dir "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/Mortality/Data/Finetunedata" \
  -cp select_cohort=CPU \
  -cp prepare_training_data=CPU \
  -cp evaluate_finetune=GPU-A100-small \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/azure/configs/Evaluate


  python -m corebehrt.azure pipeline eval_only \
  --experiment "eval-Mortality-MD-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MD/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MD/0_CreateData/tokenized" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Holdout/Mortality" \
  --model "researcher_data:Zahra/032026/CoreBehrt_OOT/MD/2_Finetune/Mortality/Model" \
  --folds_dir "researcher_data:Zahra/032026/CoreBehrt_OOT/MD/2_Finetune/Mortality/Data/Finetunedata" \
  -cp select_cohort=CPU \
  -cp prepare_training_data=CPU \
  -cp evaluate_finetune=GPU-A100-small \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/azure/configs/Evaluate


   python -m corebehrt.azure pipeline eval_only \
  --experiment "eval-Mortality-MDPS-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/tokenized" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Holdout/ARF" \
  --model "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/ARF/Model_BCE" \
  --folds_dir "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/ARF/Data/Finetunedata" \
  -cp select_cohort=CPU \
  -cp prepare_training_data=CPU \
  -cp evaluate_finetune=GPU-A100-small \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/azure/configs/Evaluate
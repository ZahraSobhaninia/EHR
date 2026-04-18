python -m corebehrt.azure pipeline FINETUNE_HOLDOUT \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/tokenized" \
  --pretrain_model "researcher_data:Zahra/032026/CoreBehrt_OOT/MDS/1_Pretrain/Model" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Finetune/Mortality" \
  --holdout_outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Holdout/Mortality" \
  -cp finetune_cv=GPU-A100-small \
  -cp evaluate_finetune=A100-reserved \
  -e finetune_holdout_test \
  CPU-20-LP \
  Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/Myconfigs


  python -m corebehrt.azure pipeline FINETUNE \
  --experiment "finetune-Mortality-MDP-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDP/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDP/0_CreateData/tokenized" \
  --pretrain_model "researcher_data:Zahra/032026/CoreBehrt_OOT/MDP/1_Pretrain/Model" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Finetune/Mortality" \
  -cp finetune_cv=GPU-A100-Single \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/Myconfigs

python -m corebehrt.azure pipeline EVAL_ONLY \
  --experiment "eval-Mortality-MDPS-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/tokenized" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Finetune/Mortality" \
  --model "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/Mortality/Model_Focal" \
  -cp evaluate_finetune=GPU-dedicated \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/Myconfigs


python -m corebehrt.azure pipeline eval_only \
  --experiment "eval-Mortality-MDPS-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/tokenized" \
  --outcomes "researcher_data:Zahra/032026/CoreBehrt_OOT/CreateOutcome_Finetune/Mortality" \
  --model "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/2_Finetune/Mortality/Model_Focal" \
  -cp evaluate_finetune=GPU-dedicated \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_OOT/corebehrt/Myconfigs
  

  
  BigStorageCPU
BigStorageCPU page

  CPU-20-LP

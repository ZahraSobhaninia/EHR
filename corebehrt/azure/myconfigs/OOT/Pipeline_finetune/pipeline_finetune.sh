


GPU-A100-Single
  GPU-A100-small
  BigStorageCPU
BigStorageCPU page

  CPU-20-LP

032026 == for oot
022026 =  for cv
052026 : model of pretrain of OOT
GPU-dedicated
  
python -m corebehrt.azure pipeline FINETUNE \
  --experiment "finetune-PNEUMONIA-MDPS-OOT" \
  --features "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/features" \
  --tokenized "researcher_data:Zahra/032026/CoreBehrt_OOT/MDPS/0_CreateData/tokenized" \
  --pretrain_model "researcher_data:Zahra/052026/CoreBehrt_OOT/MDPS/1_Pretrain/Model" \
  --outcomes "researcher_data:Zahra/062026/CoreBehrt_OOT/CreateOutcome_Finetune/PNEUMONIA" \
  -cp finetune_cv=GPU-A100-Single \
  BigStorageCPU \
  /mnt/batch/tasks/shared/LS_root/mounts/clusters/zahracpu/code/Users/zahra.sobhaninia/MODELS/Corebehrt_Integragted/corebehrt/azure/myconfigs/OOT/Pipeline_finetune
import torch

def dynamic_padding(batch: list) -> dict:
    max_len = max(sample["concept"].shape[0] for sample in batch)
    for sample in batch:
        seq_len = sample["concept"].shape[0]
        diff = max_len - seq_len
        for key, tensor_field in sample.items():
            if tensor_field.dim() == 0:
                continue
            if key == "target":
                is_finetune = batch[0].get("is_finetune", torch.tensor(False)).item()
                if is_finetune:
                    continue
                if tensor_field.dim() == 1 and tensor_field.shape[0] == seq_len:
                    filler = torch.full((diff,), -100, dtype=tensor_field.dtype)
                    sample[key] = torch.cat([tensor_field, filler], dim=0)
                continue
            if tensor_field.dim() == 1 and tensor_field.shape[0] == seq_len:
                filler = torch.zeros(diff, dtype=tensor_field.dtype)
                sample[key] = torch.cat([tensor_field, filler], dim=0)
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)
    return collated

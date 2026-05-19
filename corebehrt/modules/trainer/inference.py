from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch
import torch.nn.functional as F
from logging import getLogger

class EHRInferenceRunner(EHRTrainer):
    
# ------------------ Embedding Export Controls ------------------
# return_token_level:
#   False (default): Save patient-level embeddings only → shape [N, H]
#     + Compact and directly usable for t-SNE/UMAP/ML
#   True: Also save token-level embeddings → shape [N, L, H]
#     + Useful for sequence tagging or token-level analysis
#     - Heavy (can reach tens of GBs)
#
# token_truncate_len:
#   Only active when return_token_level=True.
#   Limits token length to L=token_truncate_len to control storage size.
#
# Size estimates (float32):
#   patient-level: N × H × 4B  (e.g. 50k × 384 × 4 ≈ 77 MB)
#   token-level:   N × L × H × 4B (e.g. 50k × 512 × 384 × 4 ≈ 39 GB)


    return_token_level: bool = False    # False=patient-level (recommended), True=token-level
    token_truncate_len: int = 128       # Only used when return_token_level=True



    def inference_loop(self, return_embeddings: bool = False) -> tuple:
        if self.test_dataset is None:
            raise ValueError("No test dataset provided")

        logger = getLogger(__name__)
        dataloader = self.get_dataloader(self.test_dataset, mode="test")
        self.model.eval()
        if return_embeddings and hasattr(self.model, "cls"):
            self.model.cls.eval()

        loop = get_tqdm(dataloader)
        loop.set_description("Running inference with embeddings" if return_embeddings else "Running inference")

        logits, targets = [], []
        patient_embs = []            # [B, H] → cat → [N, H]
        token_embs, att_masks = [], []  # If return_token_level=True

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                if return_embeddings:
                    # Try to get embeddings from the head; if it fails, fallback to masked mean pooling
                    try:
                        head = self.model.cls(
                            outputs.last_hidden_state,
                            attention_mask=batch["attention_mask"],
                        )
                        if head.dim() != 2:
                            raise ValueError("Head returned unexpected shape")
                        head = head.float().detach().cpu()   # [B, H]
                    except Exception as e:
                        logger.info("Fallback to masked mean pooling: %s", e)
                        last_hidden = outputs.last_hidden_state.float()            # [B, L, H]
                        mask = batch["attention_mask"].unsqueeze(-1).float()       # [B, L, 1]
                        head = ((last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)).cpu()  # [B, H]
                    patient_embs.append(head)

                    if getattr(self, "return_token_level", False):
                        K = int(getattr(self, "token_truncate_len", 128))
                        lh  = outputs.last_hidden_state.float().detach().cpu()     # [B, L, H]
                        am  = batch["attention_mask"].float().detach().cpu()       # [B, L]
                        token_embs.append(lh[:, :K, :])    # [B, K, H]
                        att_masks.append(am[:, :K])        # [B, K]

                logits.append(outputs.logits.float().detach().cpu())
                targets.append(batch["target"].detach().cpu())

        logits_cat = torch.cat(logits, dim=0)
        targets_cat = torch.cat(targets, dim=0)
        if logits_cat.dim() == 3 and logits_cat.shape[-1] == 1:
            logits_cat = logits_cat.squeeze(-1)

        if targets_cat.dim() == 3 and targets_cat.shape[-1] == 1:
            targets_cat = targets_cat.squeeze(-1)
            
        if logits_cat.dim() > 1 and logits_cat.shape[1] > 1:
            logits_tensor = logits_cat    # [N, n_tasks]
            targets_tensor = targets_cat  # [N, n_tasks]
        else:
            logits_tensor = logits_cat.view(-1)   # [N]
            targets_tensor = targets_cat.view(-1)  # [N]
        if not return_embeddings:
            return logits_tensor, targets_tensor, None

        head_cat = torch.cat(patient_embs, dim=0)  #

        if getattr(self, "return_token_level", False):
            model_cat = torch.cat(token_embs, dim=0)  # [N, K, H]
            att_cat   = torch.cat(att_masks, dim=0)   # [N, K]
            embeddings = [model_cat, head_cat, att_cat]
        else:
            embeddings = head_cat  

        return logits_tensor, targets_tensor, embeddings

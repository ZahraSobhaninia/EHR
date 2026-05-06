from corebehrt.modules.trainer.trainer import EHRTrainer
from corebehrt.modules.monitoring.logger import get_tqdm
import torch
import torch.nn.functional as F
from logging import getLogger

class EHRInferenceRunner(EHRTrainer):
    
    # ------------------ Embedding Export Controls ------------------
# چرا این سوییچ‌ها را اضافه کردیم؟
# هدف: کنترل «سطحِ ذخیره‌سازی امبدینگ» و مهار حجم فایل‌ها (تا ده‌ها GB نشه).
#
# return_token_level
#   False (پیشنهادی): فقط امبدینگ سطح بیمار را ذخیره کن → شکل [N, H].
#     + کوچک و جمع‌وجور
#     + مستقیم برای t-SNE/UMAP/ML قابل استفاده
#     + مناسب سناریوی «فقط کلاسیفیکیشن/پیش‌بینی»
#   True: امبدینگ سطح توکن را هم ذخیره کن → شکل [N, L, H] (+ attention_mask)
#     + برای کارهای آینده مثل sequence tagging، تحلیل در سطح توکن،
#       یا contrastive pretraining روی توکن‌ها لازم می‌شود
#     − حجیم و سنگین (ممکن است ده‌ها گیگابایت شود)
#
# token_truncate_len
#   فقط وقتی return_token_level=True اثر دارد.
#   طول توکن‌ها را هنگام ذخیره به L=token_truncate_len محدود می‌کند
#   تا حجم/سرعت ذخیره‌سازی قابل کنترل باشد (مثلاً 128 یا 256).
#
# نکته‌ها:
#  • t-SNE روی [N, H] (سطح بیمار) مستقیم جواب می‌دهد؛
#    برای [N, L, H] باید قبلش pool (مثلاً mean با ماسک) یا نمونه‌برداری توکن انجام شود.
#  • تخمین حجم (float32):
#      patient-level:  N × H × 4B  ← مثلاً 50,000 × 384 × 4 ≈ 76.8 MB (~73 MiB)
#      token-level:    N × L × H × 4B ← مثلاً 50,000 × 512 × 384 × 4 ≈ 39.3 GB (~36.6 GiB)
#  • اگر ناچار به token-level هستی:
#      (1) truncate کن (token_truncate_len)
#      (2) روی دیسک float16/bfloat16 ذخیره کن (حجم ~ نصف)
#      (3) فایل‌ها را shard کن (مثلاً هر 5k بیمار یک .pt جدا)

    return_token_level: bool = False    # حالت A=False (پیشنهادی) / حالت B=True
    token_truncate_len: int = 128       # فقط وقتی حالت B فعاله



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
        token_embs, att_masks = [], []  # فقط اگر return_token_level=True

        with torch.no_grad():
            for batch in loop:
                self.batch_to_device(batch)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    outputs = self.model(batch)

                if return_embeddings:
                    # سعی می‌کنیم از head بگیریم؛ اگر نشد، میانگین ماسک‌دار
                    try:
                        head = self.model.cls(
                            outputs.last_hidden_state,
                            attention_mask=batch["attention_mask"],
                            return_embedding=True,
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

                    # حالت B: اگر فرمت قدیمی لازم داری (توکن-لول کوتاه‌شده)
                    if getattr(self, "return_token_level", False):
                        K = int(getattr(self, "token_truncate_len", 128))
                        lh  = outputs.last_hidden_state.float().detach().cpu()     # [B, L, H]
                        am  = batch["attention_mask"].float().detach().cpu()       # [B, L]
                        token_embs.append(lh[:, :K, :])    # [B, K, H]
                        att_masks.append(am[:, :K])        # [B, K]

                logits.append(outputs.logits.float().detach().cpu())
                targets.append(batch["target"].detach().cpu())

        logits_tensor  = torch.cat(logits,  dim=0).view(-1)   # [N]
        targets_tensor = torch.cat(targets, dim=0).view(-1)   # [N]

        if not return_embeddings:
            return logits_tensor, targets_tensor, None

        head_cat = torch.cat(patient_embs, dim=0)  # [N, H] روی CPU و float32

        if getattr(self, "return_token_level", False):
            model_cat = torch.cat(token_embs, dim=0)  # [N, K, H]
            att_cat   = torch.cat(att_masks, dim=0)   # [N, K]
            embeddings = [model_cat, head_cat, att_cat]
        else:
            embeddings = head_cat  # [N, H]  ← بهترین برای t-SNE و classification

        return logits_tensor, targets_tensor, embeddings

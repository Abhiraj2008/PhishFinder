"""
BERT-PhishFinder PyTorch Architecture
Direct verbatim implementation from phishfinder.ipynb (Cell 43)
Trained on 5 consolidated datasets (442,387 URLs) with DistilBERT backbone,
SpatialDropout1D, mask-aware dual pooling (1,536-D), and 8 independent dense heads.
"""

import os
import math

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PhishFinder(nn.Module):
        """
        Exact architecture verbatim from Cell 43 of phishfinder.ipynb.
        Compatible 100% with checkpoints saved from train_one_epoch/best_model.pt.
        """
        def __init__(self):
            super().__init__()

            # Pretrained DistilBERT backbone
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

            # Spatial dropout
            self.spatial_dropout = nn.Dropout1d(0.1)

            # Eight independent classification heads
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(1536, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                for _ in range(8)
            ])

        def forward(self, input_ids, attention_mask, return_dict=False):
            # 1. Extract contextual token representations
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            x = outputs.last_hidden_state

            # 2. Apply spatial dropout (transpose to B, C, L for Dropout1d, then back)
            x = self.spatial_dropout(x.transpose(1, 2)).transpose(1, 2)

            # 3. Prepare attention mask for pooling
            mask = attention_mask.unsqueeze(-1).float()

            # 4. Mean pooling
            mean_pool = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            # 5. Max pooling (-inf on pad tokens so pads never corrupt negative activations)
            x_for_max = x.masked_fill(
                mask == 0,
                torch.finfo(x.dtype).min
            )
            max_pool = x_for_max.max(dim=1).values

            # 6. Concatenate mean and max features (768 + 768 = 1536-D)
            features = torch.cat([mean_pool, max_pool], dim=1)

            # 7. Get predictions from all eight heads (each through Sigmoid)
            head_outputs = [
                torch.sigmoid(head(features))
                for head in self.heads
            ]

            # 8. Average predictions across the 8 heads
            # head_outputs shape: 8 tensors of shape (B, 1)
            stacked_heads = torch.stack(head_outputs, dim=1).squeeze(-1) # (B, 8)
            predictions = stacked_heads.mean(dim=1) # (B,)

            if return_dict:
                return {
                    "p_final": predictions,
                    "head_probs": stacked_heads,
                    "features": features
                }

            return predictions

    # Alias for backward compatibility
    BERTPhishFinderMini = PhishFinder

else:
    class PhishFinder:
        def __init__(self, *args, **kwargs):
            self.is_stub = True

        def __call__(self, *args, **kwargs):
            raise RuntimeError("PyTorch/Transformers not installed. Running in heuristic emulation mode.")

    BERTPhishFinderMini = PhishFinder


def build_phishfinder_model(weights_path=None, device="cpu"):
    """
    Factory helper to instantiate PhishFinder and load weights from phishfinder.ipynb.
    Supports both raw state_dict and checkpoint dicts containing 'model_state_dict'.
    """
    if not TORCH_AVAILABLE:
        return None

    try:
        model = PhishFinder()
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[Model] Successfully loaded weights into PhishFinder from {weights_path}")
        return model.to(device)
    except Exception as e:
        print(f"[Model] Could not initialize PyTorch PhishFinder: {e}")
        return None

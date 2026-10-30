"""
BERT-PhishFinder PyTorch Architecture
Implementation based on:
"BERT-PhishFinder: A Robust Model for Accurate Phishing URL Detection With Optimized DistilBERT"
IEEE Transactions on Dependable and Secure Computing (IEEE TDSC, 2025)
Reimplementation by Yash Bhatia (Project Explainer, October 2026)

Key Architectural Components:
1. Tokenizer: distilbert-base-uncased (WordPiece vocabulary: 30,522)
2. Backbone: DistilBERT (6 transformer layers, 12 attention heads, hidden dim = 768)
3. Regularization: SpatialDropout1D (deactivates entire feature channels)
4. Dual Pooling: Concatenation of Global Average Pooling (768) + Global Max Pooling (768) -> 1,536 features
5. Ensemble Heads: 8 parallel dense layers (each: Dropout(0.5) -> Linear(1536, 64) -> ReLU -> Linear(64, 1) -> Sigmoid)
6. Consensus Output: Average pooling across 8 head probabilities -> P_final in [0, 1]
"""

import os
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class SpatialDropout1D(nn.Module):
        """
        SpatialDropout1D randomly drops 1D feature channels across all sequence positions.
        Input tensor shape: (batch_size, sequence_length, feature_dim)
        """
        def __init__(self, p=0.1):
            super().__init__()
            self.p = p

        def forward(self, x):
            if not self.training or self.p == 0:
                return x
            # x shape: (B, L, C) -> transpose to (B, C, L, 1) for Dropout2d
            x = x.unsqueeze(3).permute(0, 2, 1, 3)
            x = F.dropout2d(x, p=self.p, training=True)
            return x.permute(0, 2, 1, 3).squeeze(3)


    class EnsembleDenseHead(nn.Module):
        """
        Single dense classifier head:
        Dropout -> Linear(1536 -> 64) -> ReLU -> Linear(64 -> 1) -> Sigmoid
        """
        def __init__(self, in_features=1536, hidden_dim=64, dropout_rate=0.5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)


    class BERTPhishFinderMini(nn.Module):
        """
        PyTorch implementation of BERT-PhishFinder (Mini version for single Kaggle dataset replication).
        """
        def __init__(self, pretrained_model_name="distilbert-base-uncased", num_heads=8, dropout_embed=0.1, dropout_head=0.5):
            super().__init__()
            self.num_heads = num_heads

            try:
                from transformers import DistilBertModel
                self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)
            except Exception:
                # Fallback to custom transformer encoder if transformers package is not yet initialized
                encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, batch_first=True)
                self.distilbert = nn.TransformerEncoder(encoder_layer, num_layers=6)
                self.embedding = nn.Embedding(30522, 768)

            # SpatialDropout on contextual token representations
            self.spatial_dropout = SpatialDropout1D(p=dropout_embed)

            # 8 Parallel dense classifier heads
            self.heads = nn.ModuleList([
                EnsembleDenseHead(in_features=1536, hidden_dim=64, dropout_rate=dropout_head)
                for _ in range(num_heads)
            ])

        def forward(self, input_ids, attention_mask=None):
            """
            Args:
                input_ids: Tensor of shape (batch_size, sequence_length)
                attention_mask: Tensor of shape (batch_size, sequence_length) - 1 for real tokens, 0 for padding
            Returns:
                dict: {
                    'p_final': Tensor of shape (batch_size, 1),
                    'head_probs': Tensor of shape (batch_size, num_heads),
                    'pooled_features': Tensor of shape (batch_size, 1536)
                }
            """
            # Extract DistilBERT contextual representations
            if hasattr(self.distilbert, "transformer"):
                outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
                sequence_output = outputs.last_hidden_state  # (B, L, 768)
            else:
                x = self.embedding(input_ids)
                sequence_output = self.distilbert(x)

            # Apply SpatialDropout1D
            augmented_seq = self.spatial_dropout(sequence_output)

            # Dual Pooling: Global Average Pooling + Global Max Pooling with mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
                
                # Global Average Pooling (masked)
                sum_embeddings = torch.sum(augmented_seq * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                avg_pool = sum_embeddings / sum_mask  # (B, 768)

                # Global Max Pooling (masked)
                masked_seq = augmented_seq.clone()
                masked_seq[mask_expanded == 0] = -1e9
                max_pool, _ = torch.max(masked_seq, dim=1)  # (B, 768)
            else:
                avg_pool = torch.mean(augmented_seq, dim=1)
                max_pool, _ = torch.max(augmented_seq, dim=1)

            # Concatenate to 1,536 features
            pooled_features = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 1536)

            # Evaluate each of the 8 dense ensemble heads
            head_outputs = []
            for head in self.heads:
                prob = head(pooled_features)  # (B, 1)
                head_outputs.append(prob)

            # Stack to (B, 8)
            head_probs = torch.cat(head_outputs, dim=-1)

            # Final prediction: average pooling across 8 heads (Eq. 13 in IEEE TDSC paper)
            p_final = torch.mean(head_probs, dim=-1, keepdim=True)  # (B, 1)

            return {
                "p_final": p_final,
                "head_probs": head_probs,
                "pooled_features": pooled_features
            }

else:
    # PyTorch stub when torch is not yet installed in the active environment
    class BERTPhishFinderMini:
        def __init__(self, *args, **kwargs):
            self.num_heads = 8
            self.is_stub = True

        def __call__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is not installed. Use PhishFinderInferenceEngine fallback mode.")


def build_phishfinder_model(weights_path=None, device="cpu"):
    """
    Factory helper to instantiate and optionally load pre-trained weights.
    """
    if not TORCH_AVAILABLE:
        return None
    model = BERTPhishFinderMini()
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"[Model] Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"[Model] Warning: Could not load weights from {weights_path}: {e}")
    return model

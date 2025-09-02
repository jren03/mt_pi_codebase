from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FeatureCompressor(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        prop_dim,
        obs_horizon,
        num_layers=1,
        learn_prop_fusion=True,
        img_dropout: float = 0.0,
        regress_kpts: bool = False,
        adapt_domain: bool = False,
        keypoint_map_predictor: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder_fc = create_mlp(
            in_dim, out_dim, hidden_dims=[hid_dim] * num_layers
        )

        if img_dropout > 0.0:
            self.img_feats_dropout = nn.Dropout(p=img_dropout)
        else:
            self.img_feats_dropout = nn.Identity()

        self.learn_prop_fusion = learn_prop_fusion
        if learn_prop_fusion:
            self.prop_fc = nn.Sequential(
                nn.Linear(out_dim + prop_dim, out_dim + prop_dim),
                nn.LayerNorm(out_dim + prop_dim),
                nn.ReLU(),
                nn.Linear(out_dim + prop_dim, out_dim + prop_dim),
            )
        else:
            self.prop_fc = nn.Identity()

        self.keypoint_map_predictor = keypoint_map_predictor

        # Network for auxiliary keypoint prediction loss
        self.regress_kpts = regress_kpts
        if regress_kpts:
            self.keypoint_attention = KeypointAttention(
                out_dim, obs_horizon=obs_horizon, num_keypoints=int(prop_dim // 2)
            )

        # Network for auxiliary domain adaptation loss
        self.adapt_domain = adapt_domain
        if adapt_domain:
            self.domain_classifier = nn.Sequential(
                nn.Linear(out_dim + prop_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, 1),  # Binary classification: human vs robot
            )

        self.prop_dim = prop_dim
        self.obs_horizon = obs_horizon
        self.out_dim = out_dim + (prop_dim * obs_horizon)

    def forward(
        self,
        view_feats: List[torch.Tensor],
        prop: Optional[torch.Tensor],
        alpha: float = 1.0,
    ):
        """
        View_feats is a list of tensors, each of shape (B, oh, emb_dim).
        Each index in the list represents the embedding vector for a different view.
        Both have shape (B, oh, emb_dim) and (B, oh, prop_dim) respectively.
        """
        B = prop.shape[0]

        # List[(B, oh, emb_dim)] -> (B, oh, emb_dim * len(view_feats))
        view_feats = torch.cat(view_feats, dim=-1)
        # (B, oh, emb_dim * len(view_feats)) -> (B * oh, emb_dim * len(view_feats))
        view_feats = view_feats.reshape(-1, view_feats.shape[-1])
        encoded_feats = self.encoder_fc(view_feats)

        if self.regress_kpts:
            attended_feats, predicted_keypoints = self.keypoint_attention(
                encoded_feats, prop
            )
        else:
            attended_feats = encoded_feats
            predicted_keypoints = None

        # (B * oh, emb_dim) -> (B, oh, emb_dim)
        attended_feats = attended_feats.reshape(
            B, self.obs_horizon, attended_feats.shape[-1]
        )

        attended_feats = self.img_feats_dropout(attended_feats)

        if self.keypoint_map_predictor:
            prop = self.keypoint_map_predictor(prop)
        combined_feats = torch.cat([attended_feats, prop], dim=-1)

        if self.learn_prop_fusion:
            out_feats = self.prop_fc(combined_feats)
        else:
            out_feats = combined_feats
        out_feats = out_feats.reshape(B, self.obs_horizon, out_feats.shape[-1])

        if self.adapt_domain:
            reversed_features = GradientReversal.apply(out_feats, alpha)
            # (B, oh, emb_dim) -> (B * oh, emb_dim)
            reversed_features = reversed_features.reshape(B, -1)
            domain_logits = self.domain_classifier(reversed_features)
            # (B, 1) -> (B,)
            domain_logits = domain_logits.squeeze(-1)
        else:
            domain_logits = None

        return out_feats, predicted_keypoints, domain_logits


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class KeypointAttention(nn.Module):
    def __init__(
        self, feature_dim, num_keypoints, obs_horizon=1, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.feature_dim = feature_dim
        self.num_keypoints = num_keypoints

        # Ensure the feature_dim is divisible by num_heads
        assert (
            feature_dim % num_heads == 0
        ), "feature_dim must be divisible by num_heads"

        # MultiheadAttention layer
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Learnable query for keypoints
        self.keypoint_query = nn.Parameter(torch.randn(1, obs_horizon, feature_dim))

        # Keypoint prediction
        self.keypoint_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * num_keypoints),  # x, y coordinates for each keypoint
        )

        # True keypoint embedding
        self.keypoint_embedding = nn.Linear(2 * num_keypoints, feature_dim)

    def forward(self, features, true_keypoints=None):
        """
        features.shape == (B * oh, D * len(view_feats))
        true_keypoints.shape == (B, oh, 2 * num_keypoints)
        """
        B = features.shape[0]  # Batch, Dimension

        # Expand keypoint query for batch size
        features = features.reshape(-1, self.obs_horizon, self.feature_dim)
        query = self.keypoint_query.expand(B, -1, -1)

        # Apply multihead attention
        attended_features, attention_weights = self.multihead_attn(
            query=query, key=features, value=features
        )

        # Predict keypoints: (B*oh, 2*num_keypoints)
        predicted_keypoints = self.keypoint_predictor(attended_features)

        if true_keypoints is not None and self.training:
            # Generate keypoint-based attention
            keypoint_attention = self.generate_keypoint_attention(
                true_keypoints, features
            )

            # Combine learned and keypoint-based attention
            combined_attention = 0.7 * attention_weights + 0.3 * keypoint_attention

            # Re-apply combined attention
            attended_features = torch.bmm(combined_attention, features)

        return attended_features, predicted_keypoints

    def generate_keypoint_attention(self, true_keypoints, features):
        B, T, D = features.shape

        # Embed true keypoints into the feature space
        keypoint_embedding = self.keypoint_embedding(true_keypoints)  # B, K, D

        # Compute similarity between keypoint embeddings and features
        similarity = torch.bmm(keypoint_embedding, features.transpose(1, 2))  # B, K, T

        # Apply softmax to get attention weights
        attention = F.softmax(similarity / (D**0.5), dim=-1)

        return attention


def create_mlp(
    input_dim,
    output_dim,
    hidden_dims=None,
    norm_layer=nn.LayerNorm,
    activation_layer=nn.ReLU,
):
    layers = []
    current_dim = input_dim

    if hidden_dims is not None:
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            if activation_layer is not None:
                layers.append(activation_layer())
            current_dim = hidden_dim

    layers.append(nn.Linear(current_dim, output_dim))

    return nn.Sequential(*layers)

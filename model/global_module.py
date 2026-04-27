import torch
import torch.nn as nn


class GlobalLayer(nn.Module):
    def __init__(self, d_model_list, nhead_list, d_ff, dropout, act, self_attn_list):
        super().__init__()

        # Giai đoạn 1: Self-Attention Layers
        self.self_attn_lh = self_attn_list[0]
        self.self_attn_rh = self_attn_list[1]
        self.self_attn_body = self_attn_list[2]
        self.norm1_lh = nn.LayerNorm(d_model_list[0])
        self.norm1_rh = nn.LayerNorm(d_model_list[1])
        self.norm1_body = nn.LayerNorm(d_model_list[2])

        # rh truyền cho lh
        self.lh_from_rh_attn = nn.MultiheadAttention(
            d_model_list[0],
            nhead_list[0],
            kdim=d_model_list[1],
            vdim=d_model_list[1],
            dropout=dropout,
            batch_first=True,
        )
        self.rh_from_lh_attn = nn.MultiheadAttention(
            d_model_list[1],
            nhead_list[1],
            kdim=d_model_list[0],
            vdim=d_model_list[0],
            dropout=dropout,
            batch_first=True,
        )

        # Fusion layer chỉ nhận đầu ra từ một chú ý chéo
        self.norm2_lh = nn.LayerNorm(d_model_list[0])
        self.norm2_rh = nn.LayerNorm(d_model_list[1])

        # Giai đoạn 3: Feed-Forward Networks
        self.ffn_lh = nn.Sequential(
            nn.Linear(d_model_list[0], d_ff),
            nn.ReLU() if act == "relu" else nn.GELU(),
            nn.Linear(d_ff, d_model_list[0]),
        )
        self.ffn_rh = nn.Sequential(
            nn.Linear(d_model_list[1], d_ff),
            nn.ReLU() if act == "relu" else nn.GELU(),
            nn.Linear(d_ff, d_model_list[1]),
        )

        self.ffn_body = nn.Sequential(
            nn.Linear(d_model_list[2], d_ff),
            nn.ReLU() if act == "relu" else nn.GELU(),
            nn.Linear(d_ff, d_model_list[2]),
        )

        self.norm3_lh = nn.LayerNorm(d_model_list[0])
        self.norm3_rh = nn.LayerNorm(d_model_list[1])
        self.norm3_body = nn.LayerNorm(d_model_list[2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, l_hand_x, r_hand_x, body_x):  # post-norm
        # l_hand_x, r_hand_x, body_x: [B, L, D]

        # --- 1. Self-Attention ---
        lh_self, _ = self.self_attn_lh(l_hand_x, l_hand_x, l_hand_x)
        l_hand_x = self.norm1_lh(l_hand_x + self.dropout(lh_self))

        rh_self, _ = self.self_attn_rh(r_hand_x, r_hand_x, r_hand_x)
        r_hand_x = self.norm1_rh(r_hand_x + self.dropout(rh_self))

        body_self, _ = self.self_attn_body(body_x, body_x, body_x)
        body_x = self.norm1_body(body_x + self.dropout(body_self))

        # --- 2. Cross-Attention ---
        lh_from_rh, _ = self.lh_from_rh_attn(l_hand_x, r_hand_x, r_hand_x)
        rh_from_lh, _ = self.rh_from_lh_attn(
            query=r_hand_x, key=l_hand_x, value=l_hand_x
        )
        l_hand_x = self.norm2_lh(l_hand_x + self.dropout(lh_from_rh))
        r_hand_x = self.norm2_rh(r_hand_x + self.dropout(rh_from_lh))

        # --- 3. Feed-Forward Network ---
        l_hand_x = self.norm3_lh(l_hand_x + self.dropout(self.ffn_lh(l_hand_x)))
        r_hand_x = self.norm3_rh(r_hand_x + self.dropout(self.ffn_rh(r_hand_x)))
        body_x = self.norm3_body(body_x + self.dropout(self.ffn_body(body_x)))

        return l_hand_x, r_hand_x, body_x


if __name__ == "__main__":
    batch_size = 24
    seq_len = 204
    # input_dim =
    hidden_dim = 2048

    lh = torch.randn(batch_size, seq_len, 42)
    rh = torch.randn(batch_size, seq_len, 42)
    bd = torch.randn(batch_size, seq_len, 24)

    model = GlobalLayer(d_model_list=[42, 42, 24], hidden_dim=2048)

    y, _, _ = model(lh, rh, bd)

    print(y.shape)

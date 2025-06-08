__all__ = ["SwinPatchTST_backbone"]

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

# from collections import OrderedDict
from .SwinPatchTST_layers import *
from .RevIN import RevIN
from flash_attn.modules.mha import MHA


# Cell
class SwinPatchTST_backbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model=128,
        n_heads=16,
        n_windows=16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = "auto",
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout=0,
        padding_patch=None,
        head_type="flatten",
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        verbose: bool = False,
    ):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_windows=n_windows,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
        )

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.head_type = head_type
        self.individual = individual

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x patch_len x patch_num]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        max_seq_len=1024,
        n_layers=3,
        d_model=128,
        n_heads=16,
        n_windows=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        act="gelu",
        store_attn=False,
        key_padding_mask="auto",
        padding_var=None,
        attn_mask=None,
        pre_norm=False,
        pe="zeros",
        learn_pe=True,
        verbose=False,
    ):

        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, d_model
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            d_model,
            n_heads,
            n_windows=n_windows,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            n_layers=n_layers,
            store_attn=store_attn,
        )

    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


# Cell
class TSTEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_windows=16,
        d_k=None,
        d_v=None,
        d_ff=None,
        norm="BatchNorm",
        attn_dropout=0.0,
        dropout=0.0,
        activation="gelu",
        n_layers=1,
        pre_norm=False,
        store_attn=False,
    ):
        super().__init__()
        self.n_windows = n_windows

        self.layers1 = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers // 2)
            ]
        )
        self.layers2 = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers // 2)
            ]
        )
        self.layers3 = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers // 2)
            ]
        )
        self.layers4 = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers // 2)
            ]
        )

        self.patch_merge1 = PatchMerging(d_model)
        self.patch_merge2 = PatchMerging(d_model)
        self.patch_merge3 = PatchMerging(d_model)

        self.cls_1 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_2 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_3 = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cls_4 = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        # src: [bs*nvars x patch_num x d_model]
        B, N, L = src.shape
        output = src

        # stage 1
        output = output.reshape(-1, N // self.n_windows, L)
        B_w, P, D = output.shape
        cls_token = self.cls_1.expand(B_w, -1, -1).to(src.device)
        output = torch.cat((cls_token, output), dim=1)  # [B_w x (P+1) x D]
        for mod in self.layers1:
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        cls_output1 = output[:, 0, :].unsqueeze(1)  # [B_w x D]
        cls_output1 = cls_output1.reshape(B, -1, L)  # [B x n_window x D]

        # output = output[:, 1:, :]  # [B_w x P x D]
        output = output.reshape(B, -1, L)
        output = self.patch_merge1(output)

        # stage 2
        output = output.reshape(B * (self.n_windows // 2), -1, L)
        B_w, P, D = output.shape
        # cls_token = self.cls_2.expand(B_w, -1, -1).to(src.device)
        # output = torch.cat((cls_token, output), dim=1)  # [B_w x (P+1) x D]
        for mod in self.layers2:
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        cls_output2 = output[:, 0, :].unsqueeze(1)  # [B_w x 1 x D]
        cls_output2 = cls_output2.reshape(B, -1, L)  # [B x 1 x D]

        # output = output[:, 1:, :]  # [B_w x P x D]
        output = output.reshape(B, -1, L)
        output = self.patch_merge2(output)

        # stage 3
        output = output.reshape(B * (self.n_windows // 2**2), -1, L)
        B_w, P, D = output.shape
        # cls_token = self.cls_3.expand(B_w, -1, -1).to(src.device)
        # output = torch.cat((cls_token, output), dim=1)  # [B_w x (P+1) x D]
        for mod in self.layers3:
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        cls_output3 = output[:, 0, :].unsqueeze(1)  # [B_w x 1 x D]
        cls_output3 = cls_output3.reshape(B, -1, L)  # [B x 1 x D]

        # output = output[:, 1:, :]  # [B_w x P x D]
        output = output.reshape(B, -1, L)
        output = self.patch_merge3(output)

        # stage 4
        output = output.reshape(B * (self.n_windows // 2**3), -1, L)
        B_w, P, D = output.shape
        # cls_token = self.cls_4.expand(B_w, -1, -1).to(src.device)
        # output = torch.cat((cls_token, output), dim=1)  # [B_w x (P+1) x D]
        for mod in self.layers4:
            output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        cls_output4 = output[:, 0, :].unsqueeze(1)  # [B_w x 1 x D]
        cls_output4 = cls_output4.reshape(B, -1, L)  # [B x 1 x D]
        # output = output[:, 1:, :]  # [B_w x P x D]
        output = output.reshape(B, -1, L)

        final_cls_output = torch.cat(
            (cls_output1, cls_output2, cls_output3, cls_output4), dim=1
        )  # [B_w x 3 x D]
        # output = self.patch_merge3(output)

        return final_cls_output


class TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=256,
        store_attn=False,
        norm="BatchNorm",
        attn_dropout=0,
        dropout=0.0,
        bias=True,
        activation="gelu",
        pre_norm=False,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.self_attn = FlashMultiHeadAttention(
            d_model,
            n_heads,
            dropout=attn_dropout,
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        ## Multi-Head attention
        src2 = self.self_attn(src)

        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        return src


class PatchMerging(nn.Module):
    def __init__(self, patch_dim, norm=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(patch_dim * 2, patch_dim)
        self.norm = norm(patch_dim * 2)

    def forward(self, x):
        # x: [bs*nvars, patch_num, d_model]
        b, patch_num, d_model = x.shape
        if patch_num % 2 != 0:
            x = x[:, :, :-1, :]  # remove the last patch if patch_num is odd

        x = x.reshape(b, patch_num // 2, 2, d_model)
        x = x.reshape(b, patch_num // 2, 2 * d_model)
        x = self.norm(x)
        x = self.reduction(x)
        # x: [bs*nvars, patch_num//2, d_model]
        return x


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.attn = MHA(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, x, key_padding_mask=None):
        # input shape is [bs, seq_len, d_model]
        # key_padding_mask: [bs, seq_len] -> bool, True indicates that the position has been masked off
        return self.attn(x, key_padding_mask=key_padding_mask)

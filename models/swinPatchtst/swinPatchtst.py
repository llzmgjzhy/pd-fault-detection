__all__ = ["SwinPatchTST"]

# Cell
from torch import nn
import torch

from .layers.backbone import SwinPatchTST_backbone
from .layers.SwinPatchTST_layers import series_decomp
from flash_attn.modules.mha import MHA


class SwinPatchTST(nn.Module):
    def __init__(
        self,
        config,
    ):

        super().__init__()

        # load parameters
        c_in = config.enc_in
        context_window = config.seq_len
        target_window = config.pred_len

        n_layers = config.n_layers
        n_heads = config.n_heads
        n_windows = config.window_num
        d_model = config.d_model
        d_ff = config.d_ff
        dropout = config.dropout
        fc_dropout = config.dropout
        head_dropout = config.dropout

        individual = getattr(config, "individual", False)

        patch_len = config.patch_size
        stride = config.stride
        padding_patch = getattr(config, "padding_patch", None)

        revin = getattr(config, "revin", False)
        affine = getattr(config, "affine", True)
        subtract_last = getattr(config, "subtract_last", False)

        decomposition = getattr(config, "decomposition", False)
        kernel_size = getattr(config, "kernel_size", 25)

        max_seq_len = getattr(config, "max_seq_len", 1024)
        d_k = getattr(config, "d_k", None)
        d_v = getattr(config, "d_v", None)
        norm = getattr(config, "norm", "BatchNorm")
        attn_dropout = getattr(config, "attn_dropout", 0.0)
        act = getattr(config, "act", "gelu")
        key_padding_mask = getattr(config, "key_padding_mask", "auto")
        padding_var = getattr(config, "padding_var", None)
        attn_mask = getattr(config, "attn_mask", None)
        pre_norm = getattr(config, "pre_norm", False)
        store_attn = getattr(config, "store_attn", False)
        pe = getattr(config, "pe", "zeros")
        learn_pe = getattr(config, "learn_pe", True)
        pretrain_head = getattr(config, "pretrain_head", False)
        head_type = getattr(config, "task", "flatten")
        verbose = getattr(config, "verbose", False)

        patch_num = int((context_window - patch_len) / stride + 1)

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = SwinPatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
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
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )
            self.model_res = SwinPatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
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
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )
        else:
            self.model = SwinPatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                patch_len=patch_len,
                stride=stride,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                n_windows=n_windows,
                d_k=d_k,
                d_v=d_v,
                d_ff=d_ff,
                norm=norm,
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
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )

        # head
        if pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs
        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                c_in,
                self.head_nf,
                target_window,
                head_dropout=head_dropout,
            )
        elif head_type == "classification":
            final_n_patch = n_windows + n_windows // 2 + n_windows // 2**2
            # self.head = ClassificationHead(
            #     c_in, d_model, 2, head_dropout=head_dropout, patch_num=final_n_patch
            # )
            self.head = windowClassification(
                d_model, n_heads, head_dropout=head_dropout
            )
        elif head_type == "windowClassification":
            self.head = windowClassification(d_model, head_dropout=head_dropout)

    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout), nn.Conv1d(head_nf, vars, 1))

    def forward(self, x):  # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(
                0, 2, 1
            )  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = self.head(x)

        return x


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, n_vars, d_model, n_classes, head_dropout, patch_num):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(n_vars * d_model * patch_num, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x n_classes]
        """
        # x = x[
        #     :, :, :, -1
        # ]  # only consider the last item in the sequence, x: bs x nvars x d_model
        x = self.flatten(x)  # x: bs x nvars * d_model
        x = self.dropout(x)
        y = self.linear(x)  # y: bs x n_classes
        return y


class windowClassification(nn.Module):
    def __init__(self, d_model, n_heads, head_dropout):
        super().__init__()
        self.cls_fusion = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=head_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 8, d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=head_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        """
        x: [bs x nvars x d_model x n_window]
        output: [bs x n_classes]
        """

        x = x.permute(0, 1, 3, 2)  # x: bs x nvars x n_window x d_model
        x = x.mean(dim=1)
        B, N, D = x.shape
        query, _ = x.max(dim=1, keepdim=True)
        # x = self.cls_fusion(x)
        # q = self.query.expand(B, -1, -1)  # q: bs x 1 x d_model
        out, _ = self.attn(query, x, x)  # out: bs x 1 x d_model
        out = out.squeeze(1)
        # out = out.mean(dim=1)

        return self.mlp(out)  # y: bs x 1

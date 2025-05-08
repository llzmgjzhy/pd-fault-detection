__all__ = ["PatchTST"]

# Cell
from torch import nn

from .layers.PatchTST_backbone import PatchTST_backbone
from .layers.PatchTST_layers import series_decomp


class PatchTST(nn.Module):
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
        d_model = config.d_model
        d_ff = config.d_ff
        dropout = config.dropout
        fc_dropout = getattr(config, "fc_dropout", 0.0)
        head_dropout = getattr(config, "head_dropout", 0.0)

        individual = getattr(config, "individual", False)

        patch_len = config.patch_size
        stride = config.stride
        padding_patch = getattr(config, "padding_patch", None)

        revin = getattr(config, "revin", True)
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
        res_attention = getattr(config, "res_attention", True)
        pre_norm = getattr(config, "pre_norm", False)
        store_attn = getattr(config, "store_attn", False)
        pe = getattr(config, "pe", "zeros")
        learn_pe = getattr(config, "learn_pe", True)
        pretrain_head = getattr(config, "pretrain_head", False)
        head_type = getattr(config, "head_type", "flatten")
        verbose = getattr(config, "verbose", False)

        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
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
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )
            self.model_res = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
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
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )
        else:
            self.model = PatchTST_backbone(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
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
                res_attention=res_attention,
                pre_norm=pre_norm,
                store_attn=store_attn,
                pe=pe,
                learn_pe=learn_pe,
                fc_dropout=fc_dropout,
                head_dropout=head_dropout,
                padding_patch=padding_patch,
                pretrain_head=pretrain_head,
                head_type=head_type,
                individual=individual,
                revin=revin,
                affine=affine,
                subtract_last=subtract_last,
                verbose=verbose,
            )

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
            x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        return x

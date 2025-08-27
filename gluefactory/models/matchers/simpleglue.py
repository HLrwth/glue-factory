import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

def make_check_grad(name):
    def check_grad(grad):
        if torch.isnan(grad).any():
            print(f"NaN in gradients of {name}!")
        if torch.isinf(grad).any():
            print(f"Inf in gradients of {name}!")
    return check_grad

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

        # Precompute frequency vector dim_t and its inverse as buffers.
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        self.register_buffer("inv_dim_t", 1.0 / dim_t, persistent=False)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, kpts, size):
        if self.normalize:
            kpts = kpts / size.unsqueeze(1) * self.scale

        x_embed = kpts[..., 0]
        y_embed = kpts[..., 1]

        pos_x = x_embed[..., None] * self.inv_dim_t
        pos_y = y_embed[..., None] * self.inv_dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos
    

class EmCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, data):
        res0_1_sq = data["gt_res0_1_sq"]
        valid0_1 = data['gt_valid0_1']
        p_rp_01 = pred["p_rp_01"]

        res1_0_sq = data["gt_res1_0_sq"]
        valid1_0 = data['gt_valid1_0']
        p_rp_10 = pred["p_rp_10"]

        logvar_01 = pred['logvar_01']
        logvar_10 = pred['logvar_10']

        if not data.get('tr_logvar'):
            logvar_01 = logvar_01 * 0
            logvar_10 = logvar_10 * 0

        # frame0 to frame1
        res0_1_sq = (res0_1_sq * p_rp_01.unsqueeze(-1)).sum(-2)
        res0_1_sq = (res0_1_sq * torch.exp(-logvar_01)).mean(-1) / 2.0
        valid0_1_num = valid0_1.sum(-1).clamp(min=1.0)
        loss_rp_01 = (res0_1_sq * valid0_1).sum(-1) / valid0_1_num
        loss_logvar_01 = (logvar_01.mean(-1) * valid0_1).sum(-1) / valid0_1_num

        # frame1 to frame0
        res1_0_sq = (res1_0_sq * p_rp_10.unsqueeze(-1)).sum(-2)
        res1_0_sq = (res1_0_sq * torch.exp(-logvar_10)).mean(-1) / 2.0
        valid1_0_num = valid1_0.sum(-1).clamp(min=1.0)
        loss_rp_10 = (res1_0_sq * valid1_0).sum(-1) /valid1_0_num
        loss_logvar_10 = (logvar_10.mean(-1) * valid1_0).sum(-1) / valid1_0_num

        loss_rp_all = (loss_rp_01 + loss_rp_10) / 2.0
        loss_logvar_all = (loss_logvar_01 + loss_logvar_10) / 2.0

        return loss_rp_all, loss_logvar_all



@torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int, logvar_thr = 0) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.logvar_thr = logvar_thr

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, p_rp_now, logvar_now, p_rp_final, logvar_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        a_now0 = torch.cat((p_rp_now[0], torch.sigmoid(logvar_now[0] + self.logvar_thr)), -1)
        a_now1 = torch.cat((p_rp_now[1], torch.sigmoid(logvar_now[1] + self.logvar_thr)), -1)

        a_final0 = torch.cat((p_rp_final[0], torch.sigmoid(logvar_final[0] + self.logvar_thr)), -1)
        a_final1 = torch.cat((p_rp_final[1], torch.sigmoid(logvar_final[1] + self.logvar_thr)), -1)

        correct0 = (
            a_final0.max(-1).indices == a_now0.max(-1).indices
        )
        correct1 = (
            a_final1.max(-1).indices == a_now1.max(-1).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = self.embed_dim // num_heads
        assert self.embed_dim % num_heads == 0

        self.num_heads = num_heads * 2
        # self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        
        self.Wqk = nn.Linear(embed_dim, 2 * 2 * embed_dim, bias=bias)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

        self.Wv_geom = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj_geom = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn_geom = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:        
        x_vis = x[..., :self.embed_dim]
        x_geom = x[..., self.embed_dim:]

        qk = self.Wqk(x_vis)
        qk = qk.unflatten(-1, (self.num_heads, -1, 2)).transpose(1, 2)  # BxMxD(512 * 2) -> BxH(8)xMxD(64)x2
        q, k = qk[..., 0], qk[..., 1]
        q = apply_cached_rotary_emb(encoding, q)  # BxH(8)xMxD(64)
        k = apply_cached_rotary_emb(encoding, k)
        v_vis = self.Wv(x_vis)  # BxHxMxD(256)
        v_geo = self.Wv_geom(x_geom)  # BxHxMxD(256)
        v = torch.cat([v_vis, v_geo], dim=-1)
        v = v.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        context = self.inner_attn(q, k, v, mask=mask).transpose(1, 2).flatten(start_dim=-2)
        message_vis = self.out_proj(context[..., :self.embed_dim])
        message_geom = self.out_proj_geom(context[..., self.embed_dim:])
        x_vis = x_vis + self.ffn(torch.cat([x_vis, message_vis], -1))
        x_geom = x_geom + self.ffn_geom(torch.cat([x_geom, message_geom], -1))
        return x_vis, x_geom

    
class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn_vis = CrossBlock(*args, **kwargs)
        self.cross_attn_geom = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0_vis, desc0_geom = self.self_attn(desc0, encoding0)
            desc1_vis, desc1_geom = self.self_attn(desc1, encoding1)
            x0_vis, x1_vis =  self.cross_attn_vis(desc0_vis, desc1_vis)
            x0_geom, x1_geom =  self.cross_attn_geom(desc0_geom, desc1_geom)

            return torch.cat([x0_vis, x0_geom], -1), torch.cat([x1_vis, x1_geom], -1)
    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0_vis, desc0_geom = self.self_attn(desc0, encoding0, mask0)
        desc1_vis, desc1_geom = self.self_attn(desc1, encoding1, mask1)
        x0_vis, x1_vis =  self.cross_attn_vis(desc0_vis, desc1_vis, mask)
        x0_geom, x1_geom =  self.cross_attn_geom(desc0_geom, desc1_geom, mask)

        return torch.cat([x0_vis, x0_geom], -1), torch.cat([x1_vis, x1_geom], -1)


def double_softmax(sim: torch.Tensor) -> torch.Tensor:
    scores0 = F.softmax(sim - sim.max(2, keepdim=True).values, 2)
    sim_t = sim.transpose(-1, -2).contiguous()
    scores1 = F.softmax(sim_t - sim_t.max(2, keepdim = True).values, 2).transpose(-1, -2)
    scores = scores0 * scores1
    return scores


class ReprojLikelihood(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.logvar_proj = nn.Linear(dim, 2, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        desc0_vis = desc0[..., :self.dim]
        desc0_geom = desc0[..., self.dim:]
        desc1_vis = desc1[..., :self.dim]
        desc1_geom = desc1[..., self.dim:]

        mdesc0, mdesc1 = self.final_proj(desc0_vis), self.final_proj(desc1_vis)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)

        # numerical stable softmax
        p_rp_01 = F.softmax((sim - sim.max(2, keepdim=True).values), 2)
        sim_t = sim.transpose(-1, -2).contiguous()
        p_rp_10 = F.softmax((sim_t - sim_t.max(2, keepdim = True).values), 2).transpose(-1, -2)

        logvar_01 = self.logvar_proj(desc0_geom)
        logvar_10 = self.logvar_proj(desc1_geom)

        return p_rp_01, p_rp_10, logvar_01, logvar_10

    def get_var(self, desc: torch.Tensor):
        return torch.exp(self.logvar_proj(desc))


def filter_matches(scores, logvar_01, logvar_10, score_th, logvar_th):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores.max(2), scores.max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    logvar_01_value = logvar_01.max(-1).values
    logvar_10_value = logvar_10.max(-1).values
    valid0 = mutual0 & (mscores0 > score_th) & (logvar_01_value < logvar_th) & (logvar_10_value.gather(1, m0) < logvar_th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class SimpleGlue(nn.Module):
    default_conf = {
        "name": "simpleglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "logvar_filter_threshold": 0.0,  # logvar threshold as unmatched
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
        }
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    url = ""

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        self.abs_posenc = PositionEmbeddingSine(conf.descriptor_dim // 2)

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, conf.flash) for _ in range(n)]
        )

        self.reproj_likelihood = nn.ModuleList([ReprojLikelihood(d) for _ in range(n)])
        self.loss_fn = EmCrossEntropyLoss()

        state_dict = None
        if conf.weights is not None:
            # weights can be either a path or an existing file from official LG
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(
                    str(DATA_PATH / conf.weights), map_location="cpu"
                )
            else:
                if 'lightglue' in conf.weights and self.training:
                    self.url = "https://github.com/cvg/LightGlue/releases/download/{}/{}.pth"
                    pprint(f"Initialize state from lightglue for training")
                    # rename old state dict entries
                    for i in range(self.conf.n_layers):
                        pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                        pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                        state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                    state_dict = {k.replace("log_assignment", "reproj_likelihood") if "log_assignment" in k else k: v for k, v in state_dict.items()}
                else:
                    fname = (
                        f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                        + ".pth"
                    )
                    state_dict = torch.hub.load_state_dict_from_url(
                        self.url.format(conf.weights_from_version, conf.weights),
                        file_name=fname,
                    )

        if state_dict:
            strict = False if 'lightglue' in conf.weights else True
            self.load_state_dict(state_dict, strict=strict)

    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # add geom info
        abs_pe0 = self.abs_posenc(kpts0, size0)
        abs_pe1 = self.abs_posenc(kpts1, size1)
        desc0_geom = desc0 + abs_pe0
        desc1_geom = desc1 + abs_pe1
        desc0 = torch.cat([desc0, desc0_geom], -1)
        desc1 = torch.cat([desc1, desc1_geom], -1)

        all_desc0, all_desc1 = [], []

        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.transformers[i], 
                    desc0, 
                    desc1, 
                    encoding0, 
                    encoding1, 
                    use_reentrant=True
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)

            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

        # eval with last valid layer
        p_rp_01, p_rp_10, logvar_01, logvar_10 = self.reproj_likelihood[i](desc0, desc1)
        scores = p_rp_01 * p_rp_10
        m0, m1, mscores0, mscores1 = filter_matches(scores, logvar_01, logvar_10, self.conf.filter_threshold, self.conf.logvar_filter_threshold)

        prune0 = torch.ones_like(mscores0) * self.conf.n_layers
        prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            "prune0": prune0,
            "prune1": prune1,
            "p_rp_01": p_rp_01,
            "p_rp_10": p_rp_10,
            "logvar_01": logvar_01,
            "logvar_10": logvar_10,
        }

        return pred


    def loss(self, pred, data):
        def loss_params(pred, i):
            p_rp_01, p_rp_10, logvar_01, logvar_10 = self.reproj_likelihood[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "p_rp_01": p_rp_01, "p_rp_10": p_rp_10, "logvar_01": logvar_01, "logvar_10": logvar_10,
            }

        sum_weights = 1.0
        loss_rp, loss_logvar = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": loss_rp + loss_logvar, "last_rp": loss_rp.clone().detach(), "last_logvar": loss_logvar.clone().detach()}

        for i in range(N - 1):
            params_i = loss_params(pred, i)
            loss_rp, loss_logvar = self.loss_fn(params_i, data)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + (loss_rp + loss_logvar) * weight


            del params_i
        losses["total"] /= sum_weights


        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = SimpleGlue

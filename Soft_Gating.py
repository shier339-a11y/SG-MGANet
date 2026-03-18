import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftGating(nn.Module):

    def __init__(self):
        super(SoftGating, self).__init__()

    def forward(self, rgb_features, inf_features, fusion_features):
        if not (isinstance(rgb_features, (list, tuple)) and
                isinstance(inf_features, (list, tuple)) and
                isinstance(fusion_features, (list, tuple))):
            raise TypeError("rgb_features, inf_features, fusion_features 必须是 list 或 tuple")

        if not (len(rgb_features) == len(inf_features) == len(fusion_features)):
            raise ValueError("rgb_features, inf_features, fusion_features 的层数必须一致")

        rgb_weighted_list = []
        inf_weighted_list = []
        rgb_weight_list = []
        inf_weight_list = []
        rgb_dist_list = []
        inf_dist_list = []

        for level, (rgb, inf, fused) in enumerate(zip(rgb_features, inf_features, fusion_features)):
            if not (rgb.shape == inf.shape == fused.shape):
                raise ValueError(
                    f"第 {level} 层特征尺寸不一致: "
                    f"rgb={rgb.shape}, inf={inf.shape}, fused={fused.shape}"
                )

            # 欧式距离平方: D_rgb(x) = ||F_fused(x) - F_rgb(x)||_2^2
            # 按通道维求和，得到 [B, 1, H, W]
            D_rgb = torch.sum((fused - rgb) ** 2, dim=1, keepdim=True)
            D_inf = torch.sum((fused - inf) ** 2, dim=1, keepdim=True)

            # softmax(-D)，距离越小，权重越大
            logits = torch.cat([-D_rgb, -D_inf], dim=1)   # [B, 2, H, W]
            weights = F.softmax(logits, dim=1)

            w_rgb = weights[:, 0:1, :, :]   # [B, 1, H, W]
            w_inf = weights[:, 1:2, :, :]   # [B, 1, H, W]

            # 特征重加权
            rgb_weighted = w_rgb * rgb
            inf_weighted = w_inf * inf

            rgb_weighted_list.append(rgb_weighted)
            inf_weighted_list.append(inf_weighted)
            rgb_weight_list.append(w_rgb)
            inf_weight_list.append(w_inf)
            rgb_dist_list.append(D_rgb)
            inf_dist_list.append(D_inf)

        return (
            rgb_weighted_list,
            inf_weighted_list,
        )
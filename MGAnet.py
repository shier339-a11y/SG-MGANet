import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic import Conv, CPAB, Encoder, Decoder, f, Edge
from .FE import FE
from .deconv import default_conv,D4BlockTrain
from .fusion import MGAFusion
from .Soft_Gating import SoftGating

class MGAnet(nn.Module):
    def __init__(self, input_nc, output_nc, n_feat=64, kernel_size=3, reduction=4, bias=False):   # n_feat=80,reduction=4,scale_unetfeats=48,
        super(MGAnet, self).__init__()

        self.cat_layer1 = nn.Sequential(Conv(2*input_nc, n_feat, kernel_size, bias=bias),
                                        CPAB(n_feat, kernel_size, bias),
                                        CPAB(n_feat, kernel_size, bias))
        
        self.inf_layer1 = nn.Sequential(Conv(input_nc, n_feat, kernel_size, bias=bias),
                                              CPAB(n_feat, kernel_size, bias),
                                              CPAB(n_feat, kernel_size, bias))
        self.rgb_layer1 = nn.Sequential(Conv(input_nc, n_feat, kernel_size, bias=bias),
                                              CPAB(n_feat, kernel_size, bias),
                                              CPAB(n_feat, kernel_size, bias))

        self.inf_encoder = Encoder(n_feat, kernel_size, bias, atten=False)
        self.inf_decoder = Decoder(n_feat, kernel_size, bias, residual=True)

        self.rgb_encoder = Encoder(n_feat, kernel_size, bias, atten=True)
        self.rgb_decoder = Decoder(n_feat, kernel_size, bias, residual=True)

        self.conv = Conv(n_feat, output_nc, kernel_size=1, bias=bias)

        self.inf_structure = FE(n_feat, kernel_size, bias)
        self.rgb_structure = FE(n_feat, kernel_size, bias)

        self.cat_start_conv = default_conv(in_channels=6, out_channels=128, kernel_size=3, bias=True)

        self.start_conv = default_conv(in_channels = 3, out_channels = 128, kernel_size = 3, bias = True)
        self.d4_block1 = D4BlockTrain(default_conv, n_feat*2, 3)
        self.d4_block2 = D4BlockTrain(default_conv, n_feat*2, 3)
        self.final_conv = default_conv(in_channels=128, out_channels=64, kernel_size=3, bias=True)


        self.fe_level = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride=1,
                                  padding=1)

        # feature fusion
        self.mix2 = MGAFusion(n_feat * 4, reduction=16)
        self.mix1 = MGAFusion(n_feat * 2, reduction=8)
        self.mix0 = MGAFusion(n_feat, reduction=4)

        self.sg = SoftGating()

    def forward(self, rgb, inf):

        inf_fea0 = self.start_conv(inf) #1 128 480 640
        inf_fea0 = self.d4_block1(inf_fea0) #1 128 480 640
        inf_fea0 = self.d4_block2(inf_fea0) #1 128 480 640
        inf_fea0 = self.final_conv(inf_fea0) #1 64 480 640


        inf_encode_feature = self.inf_encoder(inf_fea0)
        inf_decode_feature = self.inf_decoder(inf_encode_feature)
        inf_feature = self.inf_structure(inf_encode_feature, inf_decode_feature)

        rgb_fea0 = self.start_conv(rgb)  # 1 128 480 640
        rgb_fea0 = self.d4_block1(rgb_fea0)  # 1 128 480 640
        rgb_fea0 = self.d4_block2(rgb_fea0)  # 1 128 480 640
        rgb_fea0 = self.final_conv(rgb_fea0)  # 1 64 480 640

        # visible image feature extraction branch

        rgb_encode_feature = self.rgb_encoder(rgb_fea0)
        rgb_decode_feature = self.rgb_decoder(rgb_encode_feature)
        rgb_feature = self.rgb_structure(rgb_encode_feature, rgb_decode_feature)
        # rgb_feature[0]: torch.Size([1, 64, 480, 640])
        # rgb_feature[1]: torch.Size([1, 128, 240, 320])
        # rgb_feature[2]: torch.Size([1, 256, 120, 160])


        fusion_feature = []
        for i in range(len(rgb_feature)):
            mix_function = getattr(self, f"mix{i}")  # 动态获取 mix{i} 方法
            fusion_feature.append(mix_function(rgb_feature[i], inf_feature[i]))

        rgb_weighted_fea, inf_weighted_fea = self.sg(
            rgb_feature, inf_feature, fusion_feature
        )

        fusion__weighted_fea = []
        for i in range(len(rgb_weighted_fea)):
            mix_function = getattr(self, f"mix{i}")  # 动态获取 mix{i} 方法
            fusion_feature.append(mix_function(rgb_weighted_fea[i], inf_weighted_fea[i]))


        out = self.conv(fusion__weighted_fea[0])

        return out


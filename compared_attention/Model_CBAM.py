import torch
from einops import repeat
from torch import nn
from RetNet_v2.retnet.modeling_retnet import RetNetModel
from Affnet.aff_block import Block as Frequency_Block
from attention_mechanisms.cbam import CBAM
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexReLU

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DoubleConv2D_keep_size(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      stride=(1, 1),
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      padding_mode="reflect"),
            nn.GroupNorm(num_groups=mid_channels, num_channels=mid_channels),
            nn.SELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      stride=(1, 1),
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      padding_mode="reflect"),
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            nn.SELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Resdual_DoubleConv_2D(nn.Module):
    def __init__(self, nums=1, in_channels=1, mid_channels=None):
        super(Resdual_DoubleConv_2D, self).__init__()
        self.nums = nums
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.mid_channels = mid_channels
        self.bulit_model()

    def forward(self, x):
        return self.blocks(x) + x

    def bulit_model(self):
        blocks = []
        for _ in range(self.nums):
            blocks.append(DoubleConv2D_keep_size(in_channels=self.in_channels,
                                                 out_channels=self.out_channels,
                                                 mid_channels=self.mid_channels))
        self.blocks = nn.Sequential(*blocks)


class SFB_FFC(nn.Module):
    def __init__(self, channels=1):
        super(SFB_FFC, self).__init__()
        self.channels = channels
        self.C_Relu_1 = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=channels,
                                              kernel_size=(3, 3), stride=(1, 1),
                                              padding=(1, 1), padding_mode="reflect"),
                                    nn.LeakyReLU())

        self.C_Relu_2 = nn.Sequential(ComplexConv2d(in_channels=channels, out_channels=channels,
                                              kernel_size=(3, 3), stride=(1, 1),
                                              padding=(1, 1)),
                                        ComplexReLU()
                                      )#.type(torch.complex64)
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.C_Relu_1(x)
        x = self.FLF(x)
        return self.conv(x)

    def FLF(self, x):
        x_ = torch.fft.rfft2(x)
        # x_ = x_.C
        x_ = self.C_Relu_2(x_)
        # c = x.shape[2:]
        x = torch.fft.irfft2(x_, x.shape[-2:]) + x
        return x

class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      stride=(2, 2),
                      kernel_size=(5, 5)),
            nn.GroupNorm(num_groups=mid_channels, num_channels=mid_channels), # todo
            nn.ReLU(inplace=True), # todo
            nn.Conv2d(mid_channels, out_channels,
                      stride=(1, 1),
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      padding_mode="reflect"),
             nn.GroupNorm(num_groups=out_channels, num_channels=out_channels), # todo
            nn.ReLU(inplace=True) # todo
        )

    def forward(self, x):
        return self.double_conv(x)



class feature_enhance(nn.Module):
    def __init__(self, F_bins, num_heads=8):
        super(feature_enhance, self).__init__()
        self.series_linear = nn.Linear(in_features=F_bins, out_features=1)
        self.SoftMax = nn.Softmax(dim=1)
        self.Attention = nn.MultiheadAttention(embed_dim=F_bins, num_heads=num_heads)

    def forward(self, x):
        # B T F
        B, T, F = x.shape
        f_weight = self.SoftMax(torch.sum(x, dim=1))
        f_weight = repeat(f_weight, "b f -> b T f", T=T)
        t_weight = self.series_linear(x)
        t_weight = repeat(t_weight, "b t f -> b t (f F)", F=F)
        output, weight = self.Attention(query=f_weight, key=t_weight, value=x)
        return output



class classifier(nn.Module):
    def __init__(self, layer_num, mid_channels, linear_feas, class_num=4):
        super(classifier, self).__init__()
        # self.cnn = DoubleConv2D(in_channels=1, out_channels=1, mid_channels=mid_channels)
        self.layer_num, self.mid_channels, self.class_num, self.linear_feas = \
            layer_num, mid_channels, class_num, linear_feas
        self.bulit_model()

    def bulit_model(self):
        blocks = []
        for i in range(self.layer_num):
            blocks.append(DoubleConv2D(in_channels=1, out_channels=1, mid_channels=self.mid_channels))
        #blocks.append(SFB_FFC()) # todo     #这个位置
        # for i in range(self.layer_num):
        #     blocks.append(DoubleConv2D(in_channels=1, out_channels=1, mid_channels=self.mid_channels))
        self.cnn = nn.Sequential(*blocks)
        self.linear = Mlp(in_features=self.linear_feas, out_features=self.class_num)##nn.Linear(in_features=self.linear_feas, out_features=self.class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]
        x = self.cnn(x.unsqueeze(1)).squeeze(1)
        x = self.linear(x.reshape(B, -1))
        return self.softmax(x)

class N_Aff_Ret(nn.Module):
    def __init__(self, N_nums, RetNet_param, aff_config):
        super(N_Aff_Ret, self).__init__()
        self.nums = N_nums
        self.RetNet_param = RetNet_param
        self.aff_config = aff_config
        self.bulit_model()

    def bulit_model(self):
        self.Affs = nn.ModuleDict()
        self.Rets = nn.ModuleDict()
        for i in range(self.nums):
            self.Affs.add_module(str(i)+"_AffNet", Frequency_Block(cfg=self.aff_config,
                                                                   dim=self.aff_config.dim,
                                                                   hidden_size=self.aff_config.hidden_size,
                                                                   num_blocks=self.aff_config.num_blocks,
                                                                   double_skip=self.aff_config.double_skip,
                                                                   attn_norm_layer=self.aff_config.attn_norm_layer))
            self.Rets.add_module(str(i)+"_RetNet", RetNetModel(self.RetNet_param))

    def forward_ith(self, x, layer_id):
        residual = x
        x = self.Affs[str(layer_id)+"_AffNet"](x.unsqueeze(1))
        x = self.Rets[str(layer_id)+"_RetNet"](inputs_embeds=x.squeeze(1),
                                               forward_impl='parallel',
                                               use_cache=True).last_hidden_state
        return x + residual # todo

    def forward(self, x):
        for lay_id in range(self.nums):
            x = self.forward_ith(x, lay_id)
        return x



class Proposed_model_v1(nn.Module):
    def __init__(self, nums, RetNet_param, aff_config, feature_dim, num_head, cls_midch, cls_lnum, cls_linearfeas,
                 class_num, t_bins=342):
        super(Proposed_model_v1, self).__init__()
        self.nums, self.feature_dim, self.num_head, self.cls_midch, self.cls_lnum, \
        self.class_num, self.cls_linearfeas, self.t_bins = nums, feature_dim, num_head, cls_midch,\
                                              cls_lnum, class_num, cls_linearfeas, t_bins
        self.RetNet_param = RetNet_param
        self.aff_config = aff_config
        self.bulit_model()

    def bulit_model(self):
        self.in_norm = nn.GroupNorm(num_groups=self.t_bins, num_channels=self.t_bins)
        self.Affs = nn.ModuleDict()
        self.Rets = nn.ModuleDict()
        self.fea_enhan = feature_enhance(F_bins=self.feature_dim, num_heads=self.num_head)
        for i in range(self.nums):
            self.Affs.add_module(str(i)+"_AffNet", Frequency_Block(cfg=self.aff_config,
                                                                   dim=self.aff_config.dim,
                                                                   hidden_size=self.aff_config.hidden_size,
                                                                   num_blocks=self.aff_config.num_blocks,
                                                                   double_skip=self.aff_config.double_skip,
                                                                   attn_norm_layer=self.aff_config.attn_norm_layer))
            self.Rets.add_module(str(i)+"_RetNet", RetNetModel(self.RetNet_param))
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=self.feature_dim//2,
                            bidirectional=True)
        self.cls_cnn = classifier(layer_num=self.cls_lnum, mid_channels=self.cls_midch,
                                  linear_feas=self.cls_linearfeas, class_num=self.class_num)

    def forward_ith(self, x, layer_id):
        residual = x
        x = self.Affs[str(layer_id)+"_AffNet"](x.unsqueeze(1))
        x = self.Rets[str(layer_id)+"_RetNet"](inputs_embeds=x.squeeze(1),
                                               forward_impl='parallel',
                                               use_cache=True).last_hidden_state
        return x + residual

    def forward(self, x):
        # x = self.in_norm(x)
        x = self.fea_enhan(x)
        for lay_id in range(self.nums):
            x = self.forward_ith(x, lay_id)
        x, _ = self.lstm(x)
        x = self.cls_cnn(x)
        return x


class Proposed_model_v2(nn.Module):
    def __init__(self, nums, RetNet_param, aff_config, feature_dim, num_head, cls_midch, cls_lnum, cls_linearfeas,
                 class_num, t_bins=342):
        super(Proposed_model_v2, self).__init__()
        self.nums, self.feature_dim, self.num_head, self.cls_midch, self.cls_lnum, \
        self.class_num, self.cls_linearfeas, self.t_bins = nums, feature_dim, num_head, cls_midch,\
                                              cls_lnum, class_num, cls_linearfeas, t_bins
        self.RetNet_param = RetNet_param
        self.aff_config = aff_config
        self.bulit_model()

    def bulit_model(self):
        self.in_norm = nn.GroupNorm(num_groups=self.t_bins, num_channels=self.t_bins)
        self.Affs = nn.ModuleDict()
        self.Rets = nn.ModuleDict()
        self.Downs = nn.ModuleDict()
        self.fea_enhan = feature_enhance(F_bins=self.feature_dim, num_heads=self.num_head)
        ret_param = self.RetNet_param
        for i in range(self.nums):
            self.Affs.add_module(str(i)+"_AffNet", Frequency_Block(cfg=self.aff_config,
                                                                   dim=self.aff_config.dim,
                                                                   hidden_size=self.aff_config.hidden_size,
                                                                   num_blocks=self.aff_config.num_blocks,
                                                                   double_skip=self.aff_config.double_skip,
                                                                   attn_norm_layer=self.aff_config.attn_norm_layer))
            self.Rets.add_module(str(i)+"_RetNet", RetNetModel(ret_param))
            if i != self.nums-1:
                self.Downs.add_module(str(i) + "down", nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1,
                                                                               kernel_size=(5, 5), stride=(2, 2),
                                                                               padding=(2, 2), padding_mode="reflect"),
                                                                     nn.SELU(inplace=True),
                                                                     nn.GroupNorm(num_groups=1, num_channels=1),
                                                                     nn.Conv2d(in_channels=1, out_channels=1,
                                                                               kernel_size=(3, 3), stride=(1, 1),
                                                                               padding=(1, 1), padding_mode="reflect")
                                                                     )
                                      )
                ret_param.hidden_size = ret_param.hidden_size//2

        self.lstm = nn.LSTM(input_size=ret_param.hidden_size, hidden_size=ret_param.hidden_size//2,
                            bidirectional=True)
        self.cls_cnn = classifier(layer_num=self.cls_lnum, mid_channels=self.cls_midch,
                                  linear_feas=self.cls_linearfeas, class_num=self.class_num)

    def forward_ith(self, x, layer_id):
        residual = x
        x = self.Affs[str(layer_id)+"_AffNet"](x.unsqueeze(1))
        x = self.Rets[str(layer_id)+"_RetNet"](inputs_embeds=x.squeeze(1),
                                               forward_impl='parallel',
                                               use_cache=True).last_hidden_state
        return x + residual

    def forward(self, x):
        x = self.in_norm(x)
        x = self.fea_enhan(x)
        for lay_id in range(self.nums):
            x = self.forward_ith(x, lay_id)
            if lay_id != self.nums - 1:
                x = self.Downs[str(lay_id) + "down"](x.unsqueeze(1)).squeeze(1)
        x, _ = self.lstm(x)
        x = self.cls_cnn(x)
        return x


class Proposed_model_v3(nn.Module):
    def __init__(self, nums, N_nums, RetNet_param, aff_config, feature_dim, num_head, cls_midch, cls_lnum, cls_linearfeas,
                 class_num, t_bins=342):
        super(Proposed_model_v3, self).__init__()
        self.nums, self.feature_dim, self.num_head, self.cls_midch, self.cls_lnum, \
        self.class_num, self.cls_linearfeas, self.t_bins, self.N_nums = nums, feature_dim, num_head, cls_midch,\
                                              cls_lnum, class_num, cls_linearfeas, t_bins, N_nums
        self.RetNet_param = RetNet_param
        self.aff_config = aff_config
        self.bulit_model()

    def bulit_model(self):
        self.in_norm = nn.GroupNorm(num_groups=self.t_bins, num_channels=self.t_bins)
        self.in_conv = DoubleConv2D_keep_size(in_channels=1, out_channels=1, mid_channels=1)
        self.Aff_Rets = nn.ModuleDict()
        self.Downs = nn.ModuleDict()
        self.fea_enhan = CBAM(channel=self.t_bins, reduction=16)
        # feature_enhance(F_bins=self.feature_dim, num_heads=self.num_head)
        ret_param = self.RetNet_param
        for i in range(self.nums):
            self.Aff_Rets.add_module(str(i)+"_AffRetNet", N_Aff_Ret(N_nums=self.N_nums,
                                                                    RetNet_param=ret_param,
                                                                    aff_config=self.aff_config))
            if i != self.nums-1:
                self.Downs.add_module(str(i) + "down", nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1,
                                                                               kernel_size=(5, 5), stride=(2, 2),
                                                                               padding=(2, 2), padding_mode="reflect"),
                                                                     nn.SELU(inplace=True),# todo
                                                                     nn.GroupNorm(num_groups=1, num_channels=1), # todo
                                                                     nn.Conv2d(in_channels=1, out_channels=1,
                                                                               kernel_size=(3, 3), stride=(1, 1),
                                                                               padding=(1, 1), padding_mode="reflect")
                                                                     )
                                      )
                ret_param.hidden_size = ret_param.hidden_size//2

        self.lstm = nn.LSTM(input_size=ret_param.hidden_size, hidden_size=ret_param.hidden_size//2,
                            bidirectional=True)
        self.cls_cnn = classifier(layer_num=self.cls_lnum, mid_channels=self.cls_midch,
                                  linear_feas=self.cls_linearfeas, class_num=self.class_num)

    def forward_ith(self, x, layer_id):
        residual = x
        x = self.Aff_Rets[str(layer_id)+"_AffRetNet"](x)
        # x = self.Affs[str(layer_id)+"_AffNet"](x.unsqueeze(1))
        # x = self.Rets[str(layer_id)+"_RetNet"](inputs_embeds=x.squeeze(1),
        #                                        forward_impl='parallel',
        #                                        use_cache=True).last_hidden_state
        return x + residual # todo

    def forward(self, x):
        x = self.in_norm(x) # todo
        x = self.in_conv(x.unsqueeze(1)).squeeze(1) # todo
        x = self.fea_enhan(x)
        for lay_id in range(self.nums):
            x = self.forward_ith(x, lay_id)
            if lay_id != self.nums - 1:
                x = self.Downs[str(lay_id) + "down"](x.unsqueeze(1)).squeeze(1)
        x, _ = self.lstm(x)
        x = self.cls_cnn(x)
        return x

if __name__ == '__main__':
    ins = torch.randn(2, 342, 32)
    from config import AffnetConfig, RetNet_param
    aff_config = AffnetConfig()
    # model = feature_enhance(F_bins=32)
    model = Proposed_model_v3(nums=2, N_nums=10, RetNet_param=RetNet_param,
                              aff_config=aff_config,
                              feature_dim=32, num_head=8, cls_midch=2,
                              cls_lnum=1, cls_linearfeas=86*8, class_num=2)
    #cls_linearfeas=169*14, class_num=2)
                              #84*6, class_num=2)
    r = model(ins)
    torch.save(model.state_dict(), "6.pth")

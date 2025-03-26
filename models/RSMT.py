import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.layers.cbam import SpatialAttn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from mamba_ssm import Mamba
# from geoseg.models.vmamba import VSSBlock
# from geoseg.models.RS3Mamba import GlobalLocalAttention, Mlp

from vmamba import VSSBlock




class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class MambaLayer(nn.Module):
    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super().__init__()
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 8)
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(
                    ConvBNReLU(in_chs, dim, kernel_size=1),
                    nn.AdaptiveAvgPool2d(1)
                    ))
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvBNReLU(in_chs, dim, kernel_size=1)
                    ))
        self.mamba = Mamba(
            d_model=dim*self.pool_len+in_chs,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand # Block expansion factor
        )

    def forward(self, x): # B, C, H, W
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        x = torch.cat(ppm_out, dim=1)
        _, chs, _, _ = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c', b=B, c=chs, h=H, w=W)
        x = self.mamba(x)
        x = (x.transpose(2, 1).view(B, chs, H, W)).contiguous()
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence


class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=512, out_ch=128, drop=0.):
        super(ConvFFN, self).__init__()
        self.conv = ConvBNReLU(in_ch, in_ch, kernel_size=3)
        self.fc1 = Conv(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = Conv(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class BlockCFLSpatial(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(BlockCFLSpatial, self).__init__()
        #self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size)
        self.mamba = VSSBlock(hidden_dim = in_chs)
        self.conv_ffn = ConvFFN(in_ch=in_chs, hidden_ch=in_chs, out_ch=out_ch, drop=drop)
        self.sig = nn.Sigmoid()

        self.convy = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                              bias=False)

    def forward(self, x,  y):
        x1_re = x.permute(0, 2, 3, 1)
        vss = self.mamba(x1_re)
        xm = vss.permute(0, 3, 1, 2)

        xm = self.conv_ffn(xm)

        max = torch.max(y,1)[0].unsqueeze(1)
        avg = torch.mean(y,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        y = self.convy(concat)

        x =  self.sig(y) * xm
        return x

class BlockCFLChannel(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(BlockCFLChannel, self).__init__()
        #self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size)
        self.mamba = VSSBlock(hidden_dim = in_chs)
        self.conv_ffn = ConvFFN(in_ch=in_chs, hidden_ch=in_chs, out_ch=out_ch, drop=drop)
        self.sig = nn.Sigmoid()

        self.channels = in_chs
        self.r = 2
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))
        self.maxpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.maxpool2= nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x,  y):

        x1_re = x.permute(0, 2, 3, 1)
        vss = self.mamba(x1_re)
        xm = vss.permute(0, 3, 1, 2)
        xm = self.conv_ffn(xm)


        max = self.maxpool(y) 

        avg = self.maxpool(y) 
        b, c, _, _ = y.size()
        linear_max = (self.linear(max.view(b,c)).view(b, c, 1, 1)).contiguous()
        linear_avg = (self.linear(avg.view(b,c)).view(b, c, 1, 1)).contiguous()
        output = linear_max + linear_avg

        x =  self.sig(output) * xm
        return x

class BlockVSS(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(BlockVSS, self).__init__()
        #self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size)
        self.mamba = VSSBlock(hidden_dim = in_chs)
        self.conv_ffn = ConvFFN(in_ch=in_chs, hidden_ch=in_chs//2, out_ch=out_ch, drop=drop)
   
    def forward(self, x):
        x1_re = x.permute(0, 2, 3, 1)
        vss = self.mamba(x1_re)
        xm = vss.permute(0, 3, 1, 2)
        #xm = self.mamba(x)
        x = self.conv_ffn(xm)

        return x




class Block(nn.Module):
    def __init__(self, in_chs=512, dim=128, hidden_ch=512, out_ch=128, drop=0.1, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super(Block, self).__init__()
        self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, last_feat_size=last_feat_size)
        self.conv_ffn = ConvFFN(in_ch=dim*self.mamba.pool_len+in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)

        return x
    

class refineDecoderM2(nn.Module):
    def __init__(self, encoder_channels=(48, 80, 160, 960), decoder_channels=128, num_classes=6, last_feat_size=16, last_act = None):
        super().__init__()

        self.last_act = None

        self.mambaconv = nn.Sequential(ConvBNReLU(encoder_channels[-2], decoder_channels,),
                                       ConvBNReLU(in_channels=decoder_channels, out_channels=decoder_channels, stride=2),
                                       BlockVSS(in_chs=decoder_channels, hidden_ch=decoder_channels, out_ch=decoder_channels)
                                       )
        self.b3 = nn.Sequential(ConvBNReLU(encoder_channels[-5], decoder_channels, ),
                                ConvBNReLU(decoder_channels, decoder_channels), )

        self.up_conv2 = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.b2 = nn.Sequential(ConvBNReLU(encoder_channels[-3], decoder_channels,),
                                ConvBNReLU(decoder_channels, decoder_channels), )
   
        self.up_conv1 = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.b1 = nn.Sequential(ConvBNReLU(encoder_channels[-4], decoder_channels,kernel_size=1),
                                ConvBNReLU(decoder_channels, decoder_channels),
                                )
         
        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels//2,),
                                  ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  ConvBNReLU(decoder_channels//2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),)


        self.lastconv = nn.Sequential(ConvBNReLU(decoder_channels // 2 + encoder_channels[-1], decoder_channels // 2,
                                                 kernel_size=1),
                                      ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                      ConvBNReLU(decoder_channels // 2, decoder_channels // 2),  #added
                                      )

        self.out = Conv(decoder_channels // 2, num_classes, kernel_size=1)

        if last_act != None:
            self.last_act = nn.Sigmoid()

        self.apply(self._init_weights)

    def forward(self, input):
        x3, x1, x2, fmamba, f_segd = input

        fm  = self.mambaconv(fmamba)
        x3r  = fm + self.b3(x3)

        x2r  = self.up_conv2(x3r)
        x2r  = x2r + self.b2(x2)

        x1r = self.up_conv1(x2r)
        x1  = self.b1(x1)
        x1  = x1 + x1r

        x = self.head(x1)

        c = self.lastconv(torch.cat((f_segd, x), dim=1))
        x = self.out(c)

        if self.last_act:
            x  = self.last_act(x)

        return x, c
    
    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=1)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



class Decoder(nn.Module):
    def __init__(self, encoder_channels=(48, 80, 160, 960), decoder_channels=128, num_classes=6, last_feat_size=16, last_act = None):
        super().__init__()

        self.last_act = None

        self.b4 = nn.Sequential(ConvBNReLU(encoder_channels[-1], decoder_channels))

        self.up_conv3 = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

        self.b3 =nn.Sequential(ConvBNReLU(encoder_channels[-2], decoder_channels))

        self.up_conv2 = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.b2 = nn.Sequential(ConvBNReLU(encoder_channels[-3], decoder_channels))

        self.up_conv1 = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                      nn.Upsample(scale_factor=2))
        self.b1 = nn.Sequential(ConvBNReLU(encoder_channels[-4], decoder_channels))

        self.pre_conv = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                      ConvBNReLU(decoder_channels, decoder_channels),)

        self.head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  ConvBNReLU(decoder_channels // 2, decoder_channels // 2),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),)

        self.out = Conv(decoder_channels // 2, num_classes, kernel_size=1)

        if last_act != None:
            self.last_act = nn.Sigmoid()
        self.apply(self._init_weights)

    def forward(self, input):
        x1, x2, x3, x4 = input
        x4 = self.b4(x4)

        x3r = self.up_conv3(x4)
        x3  = self.b3(x3)
        x3  = x3 + x3r

        x2r = self.up_conv2(x3)
        x2  = self.b2(x2)
        x2  = x2r + x2

        x1r = self.up_conv1(x2)
        x1  = self.b1(x1)
        x1  = x1 + x1r

        x = self.pre_conv(x1)
        x = self.head(x)
        x_out = self.out(x)

        if self.last_act:
            x_out = self.last_act(x_out)

        #return x1, x2, x3, x, x_out
        return x, x_out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class RSMTMamba(nn.Module):
    def __init__(self,
                 backbone_name='resnet34d.ra2_in1k',   #resnet18.fb_swsl_ig1b_ft_in1k #resnet34d.ra2_in1k
                 pretrained=True,
                 num_classes=6,
                 decoder_channels=112,
                 last_feat_size=16,  # last_feat_size=input_img_size // 32
                 img_size=512
                 ):
        super().__init__()
        cf_channel = 32
        # self.firstconv = nn.Sequential(ConvBNReLU(in_channels=4, out_channels=cf_channel, kernel_size=1),
        #                               ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel),
        #                               ConvBNReLU(in_channels=cf_channel, out_channels=3, kernel_size=1))

        self.backbone = timm.create_model(backbone_name, features_only=True, #img_size = 512, #output_stride=32,
                                          out_indices=(0, 1, 2, 3, 4), pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder_seg = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=num_classes, last_feat_size=last_feat_size)
        self.decoder_dsm = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=1,
                                   last_feat_size=last_feat_size, last_act='sigmoid')
        self.decoder_boundary = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels, num_classes=1,
                                   last_feat_size=last_feat_size, last_act='sigmoid')
        #cf_channel = decoder_channels // 2
        channel_decoder = decoder_channels // 2
        cf_channel = 32

        self.down_fusion    = nn.Sequential(ConvBNReLU(in_channels=channel_decoder*3, out_channels=cf_channel, kernel_size=1),
                                            ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel, stride=2),
                                            ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel),
                                            ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel, stride=2),
                                            ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel),
                                            ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel, stride=2),
                                            ConvBNReLU(in_channels=cf_channel, out_channels=cf_channel),
                                           )

        self.convx2  = ConvBNReLU(in_channels=encoder_channels[-3], out_channels=cf_channel, kernel_size=1)

        self.channelMamba = BlockCFLChannel(in_chs=cf_channel, dim = cf_channel, out_ch = cf_channel)
        self.spatialMamba = BlockCFLSpatial(in_chs=cf_channel, dim = cf_channel, out_ch = cf_channel)

        self.refineDeSeg = refineDecoderM2(encoder_channels=[encoder_channels[-2], encoder_channels[-4], encoder_channels[-3], cf_channel, channel_decoder], decoder_channels=decoder_channels,
                                         num_classes=num_classes, last_feat_size=last_feat_size)
        self.refineDeDsm = refineDecoderM2(encoder_channels=[encoder_channels[-2], encoder_channels[-4], encoder_channels[-3], cf_channel, channel_decoder], decoder_channels=decoder_channels,
                                         num_classes=1, last_feat_size=last_feat_size, last_act='sigmoid')
        self.refineDeBou = refineDecoderM2(encoder_channels=[encoder_channels[-2], encoder_channels[-4], encoder_channels[-3], cf_channel, channel_decoder], decoder_channels=decoder_channels,
                                         num_classes=1, last_feat_size=last_feat_size, last_act='sigmoid')



    def forward(self, x):
        #x = self.firstconv(x)
        encoder= self.backbone(x)
        x0, x1, x2, x3, x4 = encoder # 128 64 32 16
        xinput = x1, x2, x3, x4
        f_segd, x_seg = self.decoder_seg(xinput)
        f_dsmd, x_dsm = self.decoder_dsm(xinput)
        f_boud, x_bou = self.decoder_boundary(xinput)


        down_feature = self.down_fusion(torch.cat((f_segd, f_dsmd, f_boud), dim=1))
        encoder2     = self.convx2(x2)

        down_feature = down_feature + encoder2
        fmamba       = self.channelMamba(down_feature, down_feature)
        fmamba       = self.spatialMamba(fmamba, fmamba)

#
        inputs =  [x3, x1, x2, fmamba, f_segd]
        inputd =  [x3, x1, x2, fmamba, f_dsmd]
        inputb =  [x3, x1, x2, fmamba, f_boud]

        out_seg, c_s      = self.refineDeSeg(inputs)
        out_dsm, c_d      = self.refineDeDsm(inputd)
        out_bou, c_b      = self.refineDeBou(inputb)

        return x_seg, x_dsm, x_bou, out_seg, out_dsm, out_bou, #c_s, c_d, c_b, fmamba, x2



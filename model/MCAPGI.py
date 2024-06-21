# from torchvision.models import resnet34
import torch
import torch.nn as nn


from modules import ADown, CBFuse, CBLinear, Conv, CrossAttention, RepNCSPELAN4, TransformerBlock, l2_normalization, Multi_CrossAttention


class MCAPGI(nn.Module):
    def __init__(self, num_class=2, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0.1, drop_rate=0.):
        super(MCAPGI, self).__init__()

        self.transformer1 = TransformerBlock(
            dim=128, num_heads=4,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path_rate)
        self.transformer2 = TransformerBlock(
            dim=256, num_heads=4,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path_rate)
        self.transformer3 = TransformerBlock(
            dim=512, num_heads=4,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop_path=drop_path_rate)
        self.cross_attention = CrossAttention(
            dim=512, num_heads=1,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.)

        self.pos_drops = nn.Dropout(p=drop_rate)

        self.conv1 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1)

        self.avg1 = nn.AvgPool2d(4, 4)
        self.avg2 = nn.AvgPool2d(2, 2)

        self.fc = nn.Linear(512, num_class)
        self.fcm = nn.Linear(512, num_class)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.my_conv1 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.my_conv2 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            Conv(3, 64, 3, 2),
            Conv(64, 128, 3, 2),
            RepNCSPELAN4(128, 256, 128, 64, 1),
            ADown(256, 256),
            RepNCSPELAN4(256, 512, 256, 128, 1)
        )
        self.layer2 = nn.Sequential(
            ADown(512, 512),
            RepNCSPELAN4(512, 512, 512, 256, 1)
        )
        self.layer3 = nn.Sequential(
            ADown(512, 512),
            RepNCSPELAN4(512, 512, 512, 256, 1)
        )
        self.silence_layer1 = nn.Sequential(
            Conv(3, 64, 3, 2),
            Conv(64, 128, 3, 2),
            RepNCSPELAN4(128, 256, 128, 64, 1),
            ADown(256, 256)
        )
        self.cblinear1 = CBLinear(512, [256])
        self.cblinear2 = CBLinear(512, [256, 512])
        self.cblinear3 = CBLinear(512, [256, 512, 512])

        self.cbfuse1 = CBFuse([0, 0, 0])
        self.gelan1 = RepNCSPELAN4(256, 512, 256, 128, 1)
        self.adown1 = ADown(512, 512)
        self.cbfuse2 = CBFuse([1, 1])
        self.gelan2 = RepNCSPELAN4(512, 512, 512, 256, 1)
        self.adown2 = ADown(512, 512)
        self.cbfuse3 = CBFuse([2])
        self.gelan3 = RepNCSPELAN4(512, 512, 512, 256, 1)
        self.gelan4 = RepNCSPELAN4(512, 128, 128, 64, 1)
        self.gelan5 = RepNCSPELAN4(512, 256, 256, 128, 1)
        self.mult_att = Multi_CrossAttention(
            hidden_size=512, all_head_size=512)

    def forward(self, x):
        # print(x.shape)
        silence = x

        x = self.layer1(x)

        cat1 = x

        x = self.layer2(x)

        cat2 = x

        x = self.layer3(x)

        cat3 = x

        cbl1 = self.cblinear1(cat1)

        cbl2 = self.cblinear2(cat2)

        cbl3 = self.cblinear3(cat3)

        silence = self.silence_layer1(silence)

        silence = self.cbfuse1([cbl1, cbl2, cbl3, silence])

        silence = self.gelan1(silence)

        f1_0 = self.gelan4(silence)

        f1_b, f1_c, f1_h, f1_w = f1_0.shape  # f1_b=1,f1_c = 128,f1_h=28,f1_w:28
        f1 = f1_0.flatten(2).transpose(1, 2)  # f1 (1,784,128)

        f1 = self.transformer1(f1)
        f1 = f1.transpose(1, 2)
        f1 = f1.reshape(f1_b, f1_c, f1_h, f1_w)
        f1 = f1 + f1_0  # shape (1,128,28,28)

        silence = self.adown1(silence)

        silence = self.cbfuse2([cbl2, cbl3, silence])

        silence = self.gelan2(silence)

        f2_0 = self.gelan5(silence)

        f2_b, f2_c, f2_h, f2_w = f2_0.shape
        f2 = f2_0.flatten(2).transpose(1, 2)
        f2 = self.transformer2(f2)
        f2 = f2.transpose(1, 2)
        f2 = f2.reshape(f2_b, f2_c, f2_h, f2_w)
        f2 = f2 + f2_0  # shape (1,256,14,14)

        silence = self.adown2(silence)

        silence = self.cbfuse3([cbl3, silence])

        silence = self.gelan3(silence)

        f3_0 = silence
        f3_b, f3_c, f3_h, f3_w = x.shape
        f3 = x.flatten(2).transpose(1, 2)
        f3 = self.transformer3(f3)
        f3 = f3.transpose(1, 2)
        f3 = f3.reshape(f3_b, f3_c, f3_h, f3_w)
        f3 = f3 + f3_0  # shape (1,512,7,7)
        # print("f3 shape", f3.shape)

        f1 = self.conv1(f1)  # shape(1,512,28,28)
        f1 = self.avg1(f1)

        f2 = self.conv2(f2)
        f2 = self.avg2(f2)

        fm_cat = self.cross_attention(f1, f2, f3)  # shape(1,512,7,7)

        fm_cat = self.my_conv2(fm_cat)  # NPM
        fm = self.avgpool(fm_cat).flatten(1)  # NPM

        fm = l2_normalization(fm)

        # print("silence shape", silence.shape)
        silence = self.my_conv1(silence)  # NPM
        # print('silence before avgpool', silence.shape)

        silence = self.avgpool(silence)  # NPM

        silence = silence.view(silence.size(0), -1)

        fm = self.mult_att(fm, silence)  # MCA

        fm = self.fcm(fm)

        return fm


if __name__ == "__main__":

    model = MCAPGI(num_class=2)
    input = torch.randn(1, 3, 224, 224)
    output = model(input)
    print(output.shape)

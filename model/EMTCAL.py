import torch
import torch.nn as nn
from torchvision.models import resnet34
from modules import CrossAttention, TransformerBlock, l2_normalization


# This code is based on https://github.com/TangXu-Group/Remote-Sensing-Images-Classification
class EMTCAL(nn.Module):
    def __init__(self, num_class=21, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0.1, drop_rate=0.):
        super(EMTCAL, self).__init__()
        self.backbone = resnet34(pretrained=True)
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

    def forward(self, x):
        x = self.backbone.conv1(x)  # (1,64,112,112)
        x = self.backbone.bn1(x)  # (1,64,112,112)
        x = self.backbone.relu(x)  # (1,64,112,112)
        x = self.backbone.maxpool(x)  # (1,64,56,56)

        x = self.backbone.layer1(x)  # (1,64,56,56)

        x = self.backbone.layer2(x)  # (1, 128, 28, 28)
        f1_0 = x  # (1,128,28,28)
        f1_b, f1_c, f1_h, f1_w = x.shape  # f1_b=1,f1_c = 128,f1_h=28,f1_w:28
        f1 = x.flatten(2).transpose(1, 2)  # f1 (1,784,128)

        f1 = self.transformer1(f1)
        f1 = f1.transpose(1, 2)
        f1 = f1.reshape(f1_b, f1_c, f1_h, f1_w)
        f1 = f1 + f1_0  # (1,128,28,28)

        x = self.backbone.layer3(x)  # (1, 256, 14, 14)
        # print(x.shape)
        f2_0 = x
        f2_b, f2_c, f2_h, f2_w = x.shape
        f2 = x.flatten(2).transpose(1, 2)
        f2 = self.transformer2(f2)
        f2 = f2.transpose(1, 2)
        f2 = f2.reshape(f2_b, f2_c, f2_h, f2_w)
        f2 = f2 + f2_0  # (1,256,14,14)

        x = self.backbone.layer4(x)  # (1,512,7,7)

        f3_0 = x
        f3_b, f3_c, f3_h, f3_w = x.shape
        f3 = x.flatten(2).transpose(1, 2)
        f3 = self.transformer3(f3)
        f3 = f3.transpose(1, 2)
        f3 = f3.reshape(f3_b, f3_c, f3_h, f3_w)
        f3 = f3 + f3_0  # (1,512,7,7)

        f1 = self.conv1(f1)  # (1,512,28,28)
        f1 = self.avg1(f1)   # (1,512,7,7)

        f2 = self.conv2(f2)
        f2 = self.avg2(f2)  # (1,512,7,7)

        fm_cat = self.cross_attention(f1, f2, f3)  # (1,512,7,7)

        fm = self.avgpool(fm_cat).flatten(1)
        fm = l2_normalization(fm)
        fm_out = self.fcm(fm)
        # print("x shape",x.shape)
        x = self.backbone.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # fm = x + fm_out

        fm = (x + fm_out)/2.

        return fm


if __name__ == "__main__":

    input = torch.randn(1, 3, 224, 224)
    model = EMTCAL(num_class=2)
    output = model(input)
    print(output.shape)

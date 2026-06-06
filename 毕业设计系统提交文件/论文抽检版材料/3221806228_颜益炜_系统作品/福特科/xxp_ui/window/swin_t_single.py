import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinFusion_single(nn.Module):
    def __init__(
            self,
            pretrained=True,
            freeze_backbone=False,
            feature_dim=768,  # swin_tiny 默认特征维度
    ):
        super().__init__()

        # 加载预训练Swin-Tiny模型
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)  # 使用新版API

        # 特征维度设置（swin_tiny最后输出768维）
        self.feature_dim = feature_dim

        # 移除原始分类头
        self.feature_extractor = nn.Sequential(
            *list(self.backbone.children())[:-3]  # 保留特征提取部分
        )

        # 冻结参数
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # 添加自适应池化（Swin原始输出为7x7窗口特征）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 融合层（3个图像的特征拼接）
        # self.fusion = nn.Sequential(
        #     nn.Linear(3 * self.feature_dim, 1024),
        #     nn.GELU(),
        #     nn.Dropout(0.5)
        # )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """
        输入形状: (batch_size, 3_images, 3_channels, 224, 224)
        输出形状: (batch_size, 1)
        """
        img = x[:, 0]
        batch_size = x.size(0)
        feat = self.feature_extractor(img)  # [batch, 768, 7, 7]
        feat = self.avgpool(feat)  # [batch, 768, 1, 1]
        feat = feat.view(batch_size, -1)
        return self.regressor(feat)


# 测试示例
if __name__ == "__main__":
    model = SwinFusion()
    dummy_input = torch.randn(2, 3, 3, 224, 224)  # batch_size=2, 3个图像
    output = model(dummy_input)
    print(output.shape)  # 应输出 torch.Size([2, 1])
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import transforms


class SwinFusionContrastive3(nn.Module):
    def __init__(
            self,
            pretrained=True,
            freeze_backbone=False,
            feature_dim=768,
            proj_dim=256,
            init_temperature=0.5,
            lambda_cl=0.5,
            use_attention=True
    ):
        super().__init__()
        self.lambda_cl = lambda_cl
        self.use_attention = use_attention

        # 可学习的温度参数
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))

        # 骨干网络初始化
        weights = Swin_T_Weights.DEFAULT if pretrained else None
        self.backbone = swin_t(weights=weights)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])

        # 部分解冻策略
        if freeze_backbone:
            for name, param in self.feature_extractor.named_parameters():
                if 'layers.3' in name or 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 增强版投影头
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

        # 注意力融合模块
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1, bias=False)
            )

        # 回归任务分支
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim if use_attention else 3 * feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 1)
        )

        # EMA损失平衡初始化
        self.register_buffer('ema_ratio', torch.tensor(1.0))

    def forward(self, x):
        batch_size = x.size(0)
        features = []
        projections = []

        # 处理三张图像
        for i in range(3):
            img = x[:, i]
            feat = self.feature_extractor(img)
            feat = self.avgpool(feat).view(batch_size, -1)

            # 获取对比学习投影
            proj = self.projection_head(feat)
            projections.append(proj)
            features.append(feat)

        # 注意力加权融合或拼接融合
        if self.use_attention:
            features = torch.stack(features, dim=1)
            attn_weights = F.softmax(self.attention(features), dim=1)
            fused = (features * attn_weights).sum(dim=1)
        else:
            fused = torch.cat(features, dim=1)

        # 回归预测
        fused = self.fusion(fused)
        pred = self.regressor(fused)

        return pred, projections

    def contrastive_loss(self, projections, labels):
        # 向量化实现
        proj_all = torch.stack(projections, dim=1).view(-1, projections[0].size(1))
        B, num_views = len(projections[0]), 3
        total_views = B * num_views

        # 计算余弦相似度矩阵
        temperature = torch.exp(self.log_temperature.clamp(max=5.0))
        sim_mat = F.cosine_similarity(proj_all.unsqueeze(1), proj_all.unsqueeze(0), dim=2) / temperature

        # 构建样本索引和标签差异矩阵
        i_indices = torch.arange(B).repeat_interleave(num_views).to(proj_all.device)
        diff_mat = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0)) > 0.5

        # 构建正负样本掩码
        view_diff_mask = diff_mat[i_indices, i_indices.unsqueeze(0)]
        same_sample = (i_indices.unsqueeze(1) == i_indices.unsqueeze(0))
        pos_mask = same_sample & ~torch.eye(total_views, dtype=torch.bool, device=proj_all.device)
        neg_mask = view_diff_mask & ~same_sample

        # 稳定化logsumexp计算
        pos_logits = sim_mat.masked_fill(~pos_mask, -float('inf'))
        den_logits = sim_mat.masked_fill(~(pos_mask | neg_mask), -float('inf'))

        log_exp_pos = torch.logsumexp(pos_logits, dim=1)
        log_exp_den = torch.logsumexp(den_logits, dim=1)

        loss = -(log_exp_pos - log_exp_den).mean()
        return loss

    def total_loss(self, pred, targets, projections):
        mse_loss = F.mse_loss(pred, targets)
        cl_loss = self.contrastive_loss(projections, targets)

        # EMA动态平衡
        current_ratio = mse_loss.detach() / (cl_loss.detach() + 1e-8)
        self.ema_ratio = 0.9 * self.ema_ratio + 0.1 * current_ratio
        adjusted_lambda = self.lambda_cl * self.ema_ratio.clamp(0.5, 2.0)

        total_loss = mse_loss + adjusted_lambda * cl_loss
        return total_loss, self.ema_ratio.item()

    def get_temperature(self):
        return torch.exp(self.log_temperature).item()


import torch
import numpy as np
from torchvision import transforms


def preprocess_numpy_for_gpu(img_np, device):
    """
    将NumPy数组图像预处理并传输到GPU

    参数:
        img_np: NumPy数组, 形状为(H,W,3)或(H,W), 值范围通常为0-255(uint8)或0-1(float)
        device: torch设备对象(如'cuda:0')

    返回:
        torch.Tensor: 预处理后的图像张量, 形状为(1, C, H, W), 在指定设备上
    """
    # 确保输入是NumPy数组
    if not isinstance(img_np, np.ndarray):
        raise TypeError("输入必须是NumPy数组")

    # 转换图像为float32并归一化到[0,1]范围
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    elif img_np.dtype == np.float64:
        img_np = img_np.astype(np.float32)

    # 处理灰度图像(如果是单通道)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np] * 3, axis=-1)  # 转换为3通道

    # 转换为CHW格式 (HWC -> CHW)
    img_np = np.transpose(img_np, (2, 0, 1))  # 从HWC转为CHW

    # 转换为PyTorch张量(仍在CPU上)
    img_tensor = torch.from_numpy(img_np)

    # 调整大小和裁剪(在CPU上完成)
    resize = transforms.Resize(256)
    crop = transforms.CenterCrop(224)
    img_tensor = resize(img_tensor)
    img_tensor = crop(img_tensor)

    # 移动到GPU
    img_tensor = img_tensor.to(device)

    # GPU上的标准化(更快)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    # 添加批次维度并返回
    return img_tensor.unsqueeze(0).unsqueeze(0)

def predict(im1,im2,im3,model,device):
    img1 = preprocess_numpy_for_gpu(im1, device)
    img2 = preprocess_numpy_for_gpu(im2, device)
    img3 = preprocess_numpy_for_gpu(im3, device)
    input = torch.cat((img1, img2, img3), dim=1)
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(input)
        print(output)
        return round(output.cpu().numpy()[0][0],4)



# 测试示例
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im1 = cv2.imread(r'E:\ftkpic\auto_focus_sample\5\cam1\0.0.jpg')
    im2 = cv2.imread(r'E:\ftkpic\auto_focus_sample\5\cam1\0.0.jpg')
    im3 = cv2.imread(r'E:\ftkpic\auto_focus_sample\5\cam1\0.0.jpg')
    model = SwinFusionContrastive3().to(device)
    model.load_state_dict(torch.load('./best_model.pth'))
    res = predict(im1,im2,im3,model,device)
    print(res)
    # loaded_model.eval()  # 设置为评估模式
    # with torch.no_grad():
    #     output,_ = loaded_model(input)
    #     result = round(output.cpu().numpy()[0][0],4)
    #     print(result)
"""
Scale-Aware Improvements to DD3D Architecture
Chi tiết các cải tiến kỹ thuật cho small object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

class ScaleAwareDD3D(nn.Module):
    """
    DD3D với các cải tiến scale-aware cho aviation domain
    """
    
    def __init__(self, original_dd3d):
        super().__init__()
        self.backbone = original_dd3d.backbone
        
        # CẢI TIẾN 1: Multi-Scale Feature Pyramid Network
        self.scale_aware_fpn = ScaleAwareFPN(
            in_channels=[256, 512, 1024, 2048],  # DLA-34 channels
            out_channels=256,
            num_scales=6  # Tăng từ 5 lên 6 scales
        )
        
        # CẢI TIẾN 2: Scale-Aware Detection Head
        self.scale_aware_head = ScaleAwareDetectionHead(
            in_channels=256,
            num_classes=1,  # aircraft only
            scale_ranges=[(0, 32), (16, 64), (32, 128), (64, 256), (128, 512), (256, 1024)]
        )
        
        # CẢI TIẾN 3: Adaptive Receptive Field Module
        self.adaptive_rf = AdaptiveReceptiveField(
            in_channels=256,
            kernel_sizes=[3, 5, 7, 9]  # Multiple kernel sizes
        )
        
        # CẢI TIẾN 4: Scale-Aware Depth Head
        self.scale_aware_depth = ScaleAwareDepthHead(
            in_channels=256,
            depth_ranges=[(50, 200), (200, 500), (500, 1000), (1000, 1500)]
        )

class ScaleAwareFPN(nn.Module):
    """
    CẢI TIẾN 1: Multi-Scale Feature Pyramid Network
    
    Vấn đề: FPN gốc không đủ scales cho small objects
    Giải pháp: 
    - Thêm extra scales cho small objects (P6, P7)
    - Scale-specific feature enhancement
    - Cross-scale feature fusion
    """
    
    def __init__(self, in_channels, out_channels, num_scales=6):
        super().__init__()
        
        # Lateral connections (như FPN gốc)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels
        ])
        
        # Extra scales cho small objects
        self.extra_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(num_scales - len(in_channels))
        ])
        
        # Scale-specific enhancement
        self.scale_enhancers = nn.ModuleList([
            ScaleSpecificEnhancer(out_channels, scale_idx)
            for scale_idx in range(num_scales)
        ])
        
        # Cross-scale attention
        self.cross_scale_attention = CrossScaleAttention(out_channels, num_scales)
    
    def forward(self, features):
        """
        Args:
            features: List of backbone features [C2, C3, C4, C5]
        Returns:
            enhanced_features: List of enhanced features [P2, P3, P4, P5, P6, P7]
        """
        # Standard FPN forward pass
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # Top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], scale_factor=2, mode='nearest'
            )
        
        # Add extra scales
        extra_features = []
        last_feat = laterals[-1]
        for extra_conv in self.extra_convs:
            last_feat = extra_conv(last_feat)
            extra_features.append(last_feat)
        
        all_features = laterals + extra_features
        
        # Scale-specific enhancement
        enhanced_features = []
        for i, (feat, enhancer) in enumerate(zip(all_features, self.scale_enhancers)):
            enhanced_feat = enhancer(feat)
            enhanced_features.append(enhanced_feat)
        
        # Cross-scale attention
        enhanced_features = self.cross_scale_attention(enhanced_features)
        
        return enhanced_features

class ScaleSpecificEnhancer(nn.Module):
    """
    Enhancement module cho từng scale cụ thể
    
    Ý tưởng: Mỗi scale cần different enhancement strategies
    - Small scales (P6, P7): Focus on noise reduction, feature amplification
    - Large scales (P2, P3): Focus on context aggregation
    """
    
    def __init__(self, channels, scale_idx):
        super().__init__()
        self.scale_idx = scale_idx
        
        if scale_idx >= 4:  # Small object scales (P6, P7)
            # Denoising + amplification cho small objects
            self.enhancer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
                # Feature amplification
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid()
            )
        else:  # Large object scales
            # Context aggregation
            self.enhancer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        if self.scale_idx >= 4:
            # Amplification for small objects
            attention = self.enhancer(x)
            return x * attention + x
        else:
            return self.enhancer(x) + x

class AdaptiveReceptiveField(nn.Module):
    """
    CẢI TIẾN 3: Adaptive Receptive Field Module
    
    Vấn đề: Fixed receptive field không phù hợp cho multi-scale objects
    Giải pháp: Dynamic receptive field adaptation dựa trên object scale
    """
    
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        
        # Multiple branches với different kernel sizes
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // len(kernel_sizes), 
                         kernel_size=k, padding=k//2),
                nn.GroupNorm(8, in_channels // len(kernel_sizes)),
                nn.ReLU(inplace=True)
            ) for k in kernel_sizes
        ])
        
        # Adaptive weighting network
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(kernel_sizes), 1),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_conv = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, x):
        # Multiple receptive field branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Adaptive weighting
        weights = self.weight_net(x)  # [B, num_branches, 1, 1]
        
        # Weighted combination
        weighted_features = []
        for i, feat in enumerate(branch_outputs):
            weight = weights[:, i:i+1, :, :]  # [B, 1, 1, 1]
            weighted_features.append(feat * weight)
        
        # Concatenate and project
        combined = torch.cat(weighted_features, dim=1)
        output = self.output_conv(combined)
        
        return output + x  # Residual connection

class ScaleAwareDetectionHead(nn.Module):
    """
    CẢI TIẾN 2: Scale-Aware Detection Head
    
    Vấn đề: Single detection head không optimal cho different scales
    Giải pháp: Scale-specific detection heads với different strategies
    """
    
    def __init__(self, in_channels, num_classes, scale_ranges):
        super().__init__()
        self.scale_ranges = scale_ranges
        self.num_scales = len(scale_ranges)
        
        # Scale-specific classification heads
        self.cls_heads = nn.ModuleList([
            self._make_cls_head(in_channels, num_classes, scale_idx)
            for scale_idx in range(self.num_scales)
        ])
        
        # Scale-specific regression heads
        self.reg_heads = nn.ModuleList([
            self._make_reg_head(in_channels, scale_idx)
            for scale_idx in range(self.num_scales)
        ])
        
        # Scale-specific centerness heads
        self.centerness_heads = nn.ModuleList([
            self._make_centerness_head(in_channels, scale_idx)
            for scale_idx in range(self.num_scales)
        ])
    
    def _make_cls_head(self, in_channels, num_classes, scale_idx):
        """
        Tạo classification head cho scale cụ thể
        Small scales cần more sophisticated processing
        """
        if scale_idx >= 4:  # Small object scales
            # More layers for small objects
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, num_classes, 3, padding=1)
            )
        else:
            # Standard head for larger objects
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, num_classes, 3, padding=1)
            )

class ScaleAwareLoss(nn.Module):
    """
    CẢI TIẾN 4: Scale-Aware Loss Function
    
    Vấn đề: Standard loss không account for scale imbalance
    Giải pháp: Scale-aware weighting + hard negative mining
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, scale_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        # Scale-specific weights (higher weight for smaller objects)
        if scale_weights is None:
            self.scale_weights = [1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
        else:
            self.scale_weights = scale_weights
    
    def forward(self, predictions, targets, scale_indices):
        """
        Args:
            predictions: Model predictions
            targets: Ground truth
            scale_indices: Which scale each prediction belongs to
        """
        # Standard focal loss
        focal_loss = self.focal_loss(predictions, targets)
        
        # Apply scale-aware weighting
        scale_weights = torch.tensor(self.scale_weights, device=predictions.device)
        weights = scale_weights[scale_indices]
        
        weighted_loss = focal_loss * weights
        
        return weighted_loss.mean()

# USAGE EXAMPLE
def create_scale_aware_dd3d(original_dd3d_config):
    """
    Tạo DD3D với scale-aware improvements
    """
    model = ScaleAwareDD3D(original_dd3d_config)
    
    # Custom loss function
    criterion = ScaleAwareLoss(
        alpha=0.25, 
        gamma=2.0,
        scale_weights=[1.0, 1.2, 1.5, 2.0, 3.0, 4.0]  # Higher weights for smaller scales
    )
    
    return model, criterion
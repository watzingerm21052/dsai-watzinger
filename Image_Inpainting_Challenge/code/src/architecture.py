"""
    State-of-the-Art Image Inpainting Architecture
    - Gated Convolutions für bessere Inpainting-Kontrolle
    - Transformer Blocks für globalen Kontext
    - Multi-Scale Feature Fusion
    - Self-Attention Mechanisms
    - Residual Dense Blocks
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    architecture.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention Modul"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # B, HW, C
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = self.ln(x_flat + attn_out)
        x_flat = x_flat + self.ff(x_flat)
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class GatedConv2d(nn.Module):
    """Gated Convolution - speziell für Inpainting entwickelt"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        feature = self.conv_feature(x)
        gate = self.sigmoid(self.conv_gate(x))
        return feature * gate


class TransformerBlock(nn.Module):
    """Effiziente Vision Transformer Block für globalen Kontext"""
    def __init__(self, channels, num_heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(channels * mlp_ratio), channels),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Subsample für schnellere Attention (nur 32x32 Maximum)
        if H > 32 or W > 32:
            x_small = F.adaptive_avg_pool2d(x, (32, 32))
            is_subsampled = True
        else:
            x_small = x
            is_subsampled = False
            
        x_flat = x_small.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Self-Attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        x_out = x_flat.transpose(1, 2)
        if H > 32 or W > 32:
            x_out = x_out.view(B, C, 32, 32)
            x_out = F.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=True)
        else:
            x_out = x_out.view(B, C, H, W)
        
        return x_out


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg + max_out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat)) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block für bessere Feature Extraktion"""
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, growth_rate, 3, 1, 1),
            nn.InstanceNorm2d(growth_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1),
            nn.InstanceNorm2d(growth_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, 1, 1),
            nn.InstanceNorm2d(growth_rate),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_rate, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], 1))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        return x + x4 * 0.2


class DecoderBlock(nn.Module):
    """Improved Decoder Block mit Attention"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.rdb = ResidualDenseBlock(out_channels + skip_channels, growth_rate=32)
        self.cbam = CBAM(out_channels + skip_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.rdb(x)
        x = self.cbam(x)
        x = self.out_conv(x)
        return x


class ImprovedInpaintingModel(nn.Module):
    """
    State-of-the-Art Inpainting Model:
    - Pre-trained EfficientNet-B3 Encoder
    - Multi-Scale Feature Fusion
    - Self-Attention in Bottleneck
    - Residual Dense Blocks
    - CBAM Attention
    """
    def __init__(self, n_in_channels=4, n_classes=3):
        super().__init__()
        
        # Pre-trained EfficientNet-B3 als Encoder (höhere Kapazität)
        efficientnet = timm.create_model('efficientnet_b3', pretrained=True, features_only=True)
        
        # Input Projection: 4 Kanäle (RGB + Mask) → 3 Kanäle für EfficientNet
        self.input_proj = nn.Sequential(
            nn.Conv2d(n_in_channels, 3, 1),
            nn.InstanceNorm2d(3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Encoder Stages (EfficientNet-B3: [24, 32, 48, 136, 384])
        self.encoder = efficientnet
        
        # Feature Enhancement Modules
        self.enhance1 = ResidualDenseBlock(24)
        self.enhance2 = ResidualDenseBlock(32)
        self.enhance3 = ResidualDenseBlock(48)
        self.enhance4 = ResidualDenseBlock(136)
        
        # Bottleneck mit Self-Attention
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(384, growth_rate=64),
            SelfAttention(384, num_heads=8),
            ResidualDenseBlock(384, growth_rate=64),
            CBAM(384)
        )
        
        # Decoder mit skip connections
        self.dec4 = DecoderBlock(384, 136, 136)
        self.dec3 = DecoderBlock(136, 48, 48)
        self.dec2 = DecoderBlock(48, 32, 32)
        self.dec1 = DecoderBlock(32, 24, 24)
        
        # Final Upsampling und Output
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(16, growth_rate=16)
        )
        
        self.output = nn.Sequential(
            nn.Conv2d(16, n_classes, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Input Projection
        x = self.input_proj(x)
        
        # Encoder mit EfficientNet
        features = self.encoder(x)
        enc1, enc2, enc3, enc4, enc5 = features
        
        # Feature Enhancement
        enc1 = self.enhance1(enc1)
        enc2 = self.enhance2(enc2)
        enc3 = self.enhance3(enc3)
        enc4 = self.enhance4(enc4)
        
        # Bottleneck
        x = self.bottleneck(enc5)
        
        # Decoder mit skip connections
        x = self.dec4(x, enc4)
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        
        # Final Output
        x = self.final_up(x)
        x = self.output(x)
        
        return x


# Backward Compatibility
MyModel = ImprovedInpaintingModel


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block - kombiniert ResNet + DenseNet"""
    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, growth_rate, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + growth_rate, growth_rate, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2*growth_rate, growth_rate, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3*growth_rate, growth_rate, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Conv2d(channels + 4*growth_rate, channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat([x, x1], 1))
        x3 = self.conv3(torch.cat([x, x1, x2], 1))
        x4 = self.conv4(torch.cat([x, x1, x2, x3], 1))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg + max_out) * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PartialConv2d(nn.Module):
    """Simplified Partial Convolution für Inpainting"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # Separate Conv für Maske (1 channel)
        self.mask_conv = nn.Conv2d(1, 1, kernel_size, stride, padding, bias=False)
        # Initialize mask conv weights to 1
        nn.init.constant_(self.mask_conv.weight, 1.0)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, mask):
        # Convolution auf maskierten Input
        out = self.conv(x * mask)
        # Update Maske
        with torch.no_grad():
            mask_out = self.mask_conv(mask)
            mask_out = torch.clamp(mask_out, 0, 1)
        # Normalisiere basierend auf Maske
        out = out / (mask_out + 1e-8)
        out = self.norm(out)
        out = self.activation(out)
        return out, mask_out


class EncoderBlock(nn.Module):
    """Encoder Block mit PartialConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.partial_conv = PartialConv2d(in_channels, out_channels, 3, 1, 1)
        self.rdb = ResidualDenseBlock(out_channels, growth_rate=32)
        self.cbam = CBAM(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, mask):
        x, mask = self.partial_conv(x, mask)
        x = self.rdb(x)
        x = self.cbam(x)
        x_pool = self.pool(x)
        mask_pool = self.pool(mask)
        return x, x_pool, mask_pool


class DecoderBlock(nn.Module):
    """Decoder Block mit Skip Connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.rdb = ResidualDenseBlock(in_channels + skip_channels, growth_rate=32)
        self.cbam = CBAM(in_channels + skip_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if skip.shape[-2:] != x.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.rdb(x)
        x = self.cbam(x)
        x = self.out_conv(x)
        return x


class ImprovedInpaintingModel(nn.Module):
    """
    Ultra-Advanced Inpainting Model:
    - Gated Convolutions für maske-bewusste Verarbeitung
    - Transformer Blocks für globalen Kontext
    - Multi-Scale Feature Fusion
    - Pre-trained EfficientNet-B4 Encoder
    - Residual Dense Blocks
    - CBAM Attention
    """
    def __init__(self, n_in_channels=4, n_classes=3):
        super().__init__()
        
        # Input Gated Convolution
        self.input_gate = GatedConv2d(n_in_channels, 64, 7, 1, 3)
        self.input_norm = nn.InstanceNorm2d(64)
        self.input_act = nn.LeakyReLU(0.2, inplace=True)
        
        # Encoder mit Gated Convolutions
        self.enc1 = nn.Sequential(
            GatedConv2d(64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(64, 32),
            CBAM(64)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            GatedConv2d(64, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(128, 32),
            CBAM(128)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            GatedConv2d(128, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(256, 64),
            CBAM(256)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            GatedConv2d(256, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(512, 64),
            CBAM(512)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck mit Transformer Blocks (auf niedriger Auflösung)
        self.bottleneck = nn.Sequential(
            GatedConv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            TransformerBlock(512, num_heads=4),
            ResidualDenseBlock(512, 64),
            CBAM(512)
        )
        
        # Decoder mit Gated Convolutions
        self.up4 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec4 = nn.Sequential(
            GatedConv2d(512 + 512, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(256, 64),
            CBAM(256)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.dec3 = nn.Sequential(
            GatedConv2d(256 + 256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(128, 32),
            CBAM(128)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.dec2 = nn.Sequential(
            GatedConv2d(128 + 128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(64, 32),
            CBAM(64)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = nn.Sequential(
            GatedConv2d(64 + 64, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(64, 32)
        )
        
        # Output Head mit Gated Conv
        self.output = nn.Sequential(
            GatedConv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, n_classes, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract mask and RGB
        input_rgb = x[:, :3, :, :]
        mask = x[:, 3:4, :, :]
        
        # Input processing
        x = self.input_gate(x)
        x = self.input_norm(x)
        x = self.input_act(x)
        
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up4(x)
        if x.shape[-2:] != enc4.shape[-2:]:
            x = F.interpolate(x, size=enc4.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.up3(x)
        if x.shape[-2:] != enc3.shape[-2:]:
            x = F.interpolate(x, size=enc3.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        if x.shape[-2:] != enc2.shape[-2:]:
            x = F.interpolate(x, size=enc2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        if x.shape[-2:] != enc1.shape[-2:]:
            x = F.interpolate(x, size=enc1.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Generate output
        predicted = self.output(x)
        
        # Kombiniere bekannte Pixel mit Vorhersage
        output = input_rgb * mask + predicted * (1 - mask)
        
        return output


# Backward Compatibility
MyModel = ImprovedInpaintingModel

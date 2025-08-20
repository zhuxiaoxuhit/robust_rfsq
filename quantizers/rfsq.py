"""
Residual Finite Scalar Quantization (RFSQ) 实现
基于FSQ的多层残差量化模块，支持两种不同的策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .fsq import FSQ
from typing import List, Tuple, Optional


class InvertibleLayerNorm(nn.Module):
    """可逆的LayerNorm模块，专门处理图像特征图(B, C, H, W)格式"""
    
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        
        # 可学习参数，针对通道维度
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
        # 用于存储当前批次的统计信息，实现精确逆变换
        self.register_buffer('current_mean', None, persistent=False)
        self.register_buffer('current_std', None, persistent=False)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        执行LayerNorm并保存统计信息用于后续逆变换
        输入: x (B, C, H, W)
        输出: normalized (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 对每个通道计算均值和标准差 (在H, W维度上)
        self.current_mean = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        variance = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # (B, C, 1, 1)
        self.current_std = torch.sqrt(variance + self.eps)
        
        # 执行标准化
        normalized = (x - self.current_mean) / self.current_std
        
        # 应用可学习参数 (广播到 (B, C, H, W))
        weight = self.weight.view(1, C, 1, 1)  # (1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)      # (1, C, 1, 1)
        
        return weight * normalized + bias
    
    def inverse(self, normalized_x: Tensor) -> Tensor:
        """
        使用保存的统计信息执行精确的逆变换
        输入: normalized_x (B, C, H, W)
        输出: original (B, C, H, W)
        """
        if self.current_mean is None or self.current_std is None:
            raise RuntimeError("必须先调用forward方法以保存统计信息")
        
        B, C, H, W = normalized_x.shape
        
        # 获取可学习参数并调整形状
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        # 逆向操作：先减去bias，除以weight，再反标准化
        denormalized = (normalized_x - bias) / weight
        return denormalized * self.current_std + self.current_mean


class LearnableScalingRFSQStage(nn.Module):
    """使用可学习缩放因子的RFSQ层
    
    通过学习一个缩放因子来调整输入的动态范围，
    使FSQ能够更好地适应不同尺度的残差信号
    """
    
    def __init__(self, fsq_levels: List[int], initial_scale: float = 1.0):
        super().__init__()
        self.fsq = FSQ(levels=fsq_levels)
        
        # 使用log空间的参数，通过softplus确保缩放因子始终为正
        self.log_scale = nn.Parameter(torch.log(torch.tensor(initial_scale)))
    
    def forward(self, residual_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            residual_in: 输入的残差向量
        Returns:
            quantized_true: 在原始尺度下的量化向量
            residual_out: 更新后的残差，用于下一层
            indices: FSQ量化索引
        """
        # 获取正的缩放因子
        scale = F.softplus(self.log_scale)
        
        # 对输入应用缩放变换
        scaled_input = residual_in * scale
        
        # 使用FSQ进行量化
        quantized_scaled, indices = self.fsq(scaled_input)
        
        # 关键步骤：将量化结果逆变换回原始尺度
        quantized_true = quantized_scaled / scale
        
        # 计算新的残差：原始输入减去量化结果
        residual_out = residual_in - quantized_true
        
        return quantized_true, residual_out, indices


class BasicRFSQStage(nn.Module):
    """基础RFSQ层 - 不做任何预处理变换
    
    直接对输入残差使用FSQ量化，作为对比基准
    """
    
    def __init__(self, fsq_levels: List[int]):
        super().__init__()
        self.fsq = FSQ(levels=fsq_levels)
    
    def forward(self, residual_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            residual_in: 输入的残差向量
        Returns:
            quantized_true: 量化向量
            residual_out: 更新后的残差，用于下一层
            indices: FSQ量化索引
        """
        # 直接使用FSQ量化，不做任何预处理
        quantized_true, indices = self.fsq(residual_in)
        
        # 计算新的残差：原始输入减去量化结果
        residual_out = residual_in - quantized_true
        
        return quantized_true, residual_out, indices


class LayerNormRFSQStage(nn.Module):
    """使用可逆LayerNorm的RFSQ层
    
    通过LayerNorm标准化输入，使FSQ在标准化空间中工作，
    然后通过逆变换将结果映射回原始空间
    """
    
    def __init__(self, fsq_levels: List[int], num_channels):
        super().__init__()
        self.fsq = FSQ(levels=fsq_levels)
        self.layernorm = InvertibleLayerNorm(num_channels)
    
    def forward(self, residual_in: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            residual_in: 输入的残差向量 (B, C, H, W)
        Returns:
            quantized_true: 在原始尺度下的量化向量
            residual_out: 更新后的残差，用于下一层
            indices: FSQ量化索引
        """
        # 执行LayerNorm正向变换
        normalized_input = self.layernorm(residual_in)
        
        # 在标准化空间中使用FSQ量化
        quantized_normalized, indices = self.fsq(normalized_input)
        
        # 关键步骤：使用逆变换将量化结果映射回原始空间
        quantized_true = self.layernorm.inverse(quantized_normalized)
        
        # 计算新的残差
        residual_out = residual_in - quantized_true
        
        return quantized_true, residual_out, indices


class RFSQ(nn.Module):
    """Residual Finite Scalar Quantization 主模块
    
    通过多层残差量化实现更精细的向量量化，
    每一层量化当前的残差并将结果传递给下一层
    """
    
    def __init__(
        self,
        num_stages: int,
        strategy: str,
        fsq_levels: List[int],
        dim: Optional[int] = None,
        initial_scale: float = 1.0,
        **kwargs
    ):
        super().__init__()
        
        if strategy not in ['scale', 'layernorm', 'none']:
            raise ValueError("strategy必须是'scale'、'layernorm'或'none'")
        
        self.num_stages = num_stages
        self.strategy = strategy
        
        # 创建多个RFSQ层
        self.stages = nn.ModuleList()
        
        for i in range(num_stages):
            if strategy == 'scale':
                # 使用可学习缩放因子策略
                stage = LearnableScalingRFSQStage(fsq_levels, initial_scale)
            elif strategy == 'layernorm':
                if dim is None:
                    raise ValueError("使用layernorm策略时必须指定dim参数")
                # 使用可逆LayerNorm策略，dim表示通道数
                stage = LayerNormRFSQStage(fsq_levels, dim)
            else:  # none
                # 使用基础RFSQ策略，不做任何预处理变换
                stage = BasicRFSQStage(fsq_levels)
            
            self.stages.append(stage)
    
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: 原始输入向量 (B, C, H, W)
        Returns:
            z_reconstructed: 通过所有层重建的向量
            indices: 所有层的量化索引，新维度拼接
        """
        # 存储每层的量化结果和索引
        all_quantized_vectors = []
        all_indices = []
        
        # 初始残差就是原始输入
        residual = z
        
        # 逐层处理残差
        for i, stage in enumerate(self.stages):
            quantized_true, residual, indices = stage(residual)
            
            # 收集每层的结果
            all_quantized_vectors.append(quantized_true)
            all_indices.append(indices)
        
        # 最终重建：将所有层的量化向量相加
        z_reconstructed = sum(all_quantized_vectors)
        
        # 在新维度上拼接所有层的索引
        indices_tensor = torch.stack(all_indices, dim=-1)
        
        return z_reconstructed, indices_tensor

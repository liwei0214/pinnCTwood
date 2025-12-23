"""
步骤3：PINN特征提取 (NumPy 2.x兼容版本 - 完全修复版)
从扩充后的样本中提取16个物理特征
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ========== 物理常数 ==========
WOOD_PHYSICS = {
    'cell_wall_density': 1.53,  # 细胞壁密度 g/cm³
    'water_density': 1.0,
    'air_density': 0.0012,
}


class WoodDataset(Dataset):
    """木材CT数据集 (NumPy 2.x兼容版本)"""
    
    def __init__(self, ct_data, sample_name, sample_size=1000):
        self.sample_name = sample_name
        self.shape = ct_data.shape
        self.sample_size = sample_size
        
        # 确保输入数据类型正确
        ct_data = np.asarray(ct_data, dtype=np.float32)
        
        # CT统计特征
        self.ct_mean = float(np.mean(ct_data))
        self.ct_std = float(np.std(ct_data))
        self.ct_skewness = float(stats.skew(ct_data.flatten()))
        self.ct_kurtosis = float(stats.kurtosis(ct_data.flatten()))
        
        # 创建坐标网格
        z, y, x = np.meshgrid(
            np.arange(self.shape[0], dtype=np.float32),
            np.arange(self.shape[1], dtype=np.float32),
            np.arange(self.shape[2], dtype=np.float32),
            indexing='ij'
        )
        
        # 归一化坐标到[-1, 1]
        epsilon = 1e-8
        self.x_coords = np.asarray(
            (x.flatten() / (self.shape[2] + epsilon) * 2 - 1), 
            dtype=np.float32
        )
        self.y_coords = np.asarray(
            (y.flatten() / (self.shape[1] + epsilon) * 2 - 1), 
            dtype=np.float32
        )
        self.z_coords = np.asarray(
            (z.flatten() / (self.shape[0] + epsilon) * 2 - 1), 
            dtype=np.float32
        )
        
        # CT值处理
        self.ct_values = np.asarray(ct_data.flatten(), dtype=np.float32)
        self.ct_min = float(self.ct_values.min())
        self.ct_max = float(self.ct_values.max())
        
        # 归一化
        self.ct_normalized = np.asarray(
            (self.ct_values - self.ct_min) / (self.ct_max - self.ct_min + epsilon),
            dtype=np.float32
        )
        
        # 计算梯度
        self.gradients = self._compute_gradients(ct_data)
        
        self.total_voxels = int(ct_data.size)
    
    def _compute_gradients(self, ct_data):
        """计算CT梯度"""
        grad_z = np.gradient(ct_data, axis=0)
        grad_y = np.gradient(ct_data, axis=1)
        grad_x = np.gradient(ct_data, axis=2)
        grad_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        return np.asarray(grad_magnitude.flatten(), dtype=np.float32)
    
    def __len__(self):
        return max(1, self.total_voxels // self.sample_size)
    
    def __getitem__(self, idx):
        indices = np.random.choice(self.total_voxels, self.sample_size, replace=False)
        
        # 返回numpy数组，确保类型正确
        return {
            'x': np.asarray(self.x_coords[indices].reshape(-1, 1), dtype=np.float32),
            'y': np.asarray(self.y_coords[indices].reshape(-1, 1), dtype=np.float32),
            'z': np.asarray(self.z_coords[indices].reshape(-1, 1), dtype=np.float32),
            'ct_norm': np.asarray(self.ct_normalized[indices], dtype=np.float32),
            'ct_raw': np.asarray(self.ct_values[indices], dtype=np.float32),
            'gradients': np.asarray(self.gradients[indices], dtype=np.float32),
        }


class WoodPINN(nn.Module):
    """物理信息神经网络"""
    
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        
        # 编码器
        layers = []
        input_dim = 3  # x, y, z
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(input_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            input_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出头
        final_dim = hidden_dim // 2
        self.density_head = nn.Linear(final_dim, 1)
        self.moisture_head = nn.Linear(final_dim, 1)
        self.porosity_head = nn.Linear(final_dim, 1)
        self.anisotropy_head = nn.Linear(final_dim, 3)
        self.heartwood_head = nn.Linear(final_dim, 1)
        self.homogeneity_head = nn.Linear(final_dim, 1)
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x, y, z):
        # 处理输入维度
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        if len(z.shape) == 1:
            z = z.unsqueeze(1)
        
        coords = torch.cat([x, y, z], dim=1)
        
        # 编码
        features = self.encoder(coords)
        
        # 提取物理属性（带约束）
        density = torch.sigmoid(self.density_head(features)) * 0.6 + 0.75  # [0.75, 1.35]
        moisture = torch.sigmoid(self.moisture_head(features)) * 0.04 + 0.08  # [0.08, 0.12]
        porosity = torch.sigmoid(self.porosity_head(features)) * 0.3  # [0, 0.3]
        anisotropy = torch.tanh(self.anisotropy_head(features))
        heartwood = torch.sigmoid(self.heartwood_head(features))
        homogeneity = torch.sigmoid(self.homogeneity_head(features))
        
        return {
            'density': density,
            'moisture': moisture,
            'porosity': porosity,
            'anisotropy': anisotropy,
            'heartwood': heartwood,
            'homogeneity': homogeneity
        }


def physics_loss(outputs, targets):
    """物理约束损失"""
    
    density = outputs['density']
    moisture = outputs['moisture']
    porosity = outputs['porosity']
    ct_norm = targets['ct_norm']
    
    # 数据拟合损失
    expected_density = 0.8 + 0.3 * ct_norm  # 简化关系
    density_loss = torch.mean((density - expected_density) ** 2)
    
    # 物理约束：密度-孔隙度-含水率关系
    rho_cell = WOOD_PHYSICS['cell_wall_density']
    physics_density = rho_cell * (1 - porosity) * (1 + moisture)
    physics_loss_val = torch.mean((density - physics_density) ** 2)
    
    # 含水率约束
    moisture_constraint = torch.mean(
        torch.relu(0.08 - moisture) + torch.relu(moisture - 0.12)
    )
    
    # 孔隙度约束
    porosity_constraint = torch.mean(
        torch.relu(-porosity) + torch.relu(porosity - 0.3)
    )
    
    # 总损失
    total_loss = (
        1.0 * density_loss +
        0.3 * physics_loss_val +
        0.2 * moisture_constraint +
        0.2 * porosity_constraint
    )
    
    return total_loss, {
        'density': density_loss.item(),
        'physics': physics_loss_val.item(),
        'moisture_constraint': moisture_constraint.item(),
        'porosity_constraint': porosity_constraint.item(),
    }


def train_pinn(dataloader, sample_name, epochs=25, lr=1e-3):
    """训练PINN模型 (完全修复版 - 包含维度处理)"""
    
    print(f"\n训练 {sample_name}")
    
    model = WoodPINN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch in dataloader:
            model.train()
            
            # ===== 修复1：DataLoader自动将numpy数组转换为Tensor =====
            if isinstance(batch['x'], torch.Tensor):
                x = batch['x'].float().to(device)
                y = batch['y'].float().to(device)
                z = batch['z'].float().to(device)
            else:
                x = torch.from_numpy(batch['x']).float().to(device)
                y = torch.from_numpy(batch['y']).float().to(device)
                z = torch.from_numpy(batch['z']).float().to(device)
            
            # ===== 修复2：处理DataLoader批处理后的维度 =====
            if len(x.shape) == 3:
                batch_size = x.shape[0]
                sample_size = x.shape[1]
                x = x.reshape(batch_size * sample_size, -1)
                y = y.reshape(batch_size * sample_size, -1)
                z = z.reshape(batch_size * sample_size, -1)
            
            # 处理targets
            if isinstance(batch['ct_norm'], torch.Tensor):
                ct_norm = batch['ct_norm'].float().to(device)
                ct_raw = batch['ct_raw'].float().to(device)
            else:
                ct_norm = torch.from_numpy(batch['ct_norm']).float().to(device)
                ct_raw = torch.from_numpy(batch['ct_raw']).float().to(device)
            
            # 展平targets的维度
            if len(ct_norm.shape) == 2:
                ct_norm = ct_norm.reshape(-1)
                ct_raw = ct_raw.reshape(-1)
            
            # 确保维度正确
            if len(ct_norm.shape) == 1:
                ct_norm = ct_norm.unsqueeze(1)
            if len(ct_raw.shape) == 1:
                ct_raw = ct_raw.unsqueeze(1)
            
            targets = {
                'ct_norm': ct_norm,
                'ct_raw': ct_raw,
            }
            # ===== 修复结束 =====
            
            outputs = model(x, y, z)
            loss, loss_dict = physics_loss(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        
        avg_loss = float(np.mean(epoch_losses))
        history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return model, history


def extract_features(model, sample_shape, n_points=30):
    """提取16个物理特征 (NumPy 2.x兼容版本)"""
    
    model.eval()
    
    # 均匀采样 - 显式指定dtype
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, n_points, dtype=np.float32),
        np.linspace(-1, 1, n_points, dtype=np.float32),
        np.linspace(-1, 1, n_points, dtype=np.float32),
        indexing='ij'
    )
    
    with torch.no_grad():
        # 使用torch.from_numpy代替torch.FloatTensor
        x_flat = torch.from_numpy(np.asarray(x.flatten(), dtype=np.float32)).unsqueeze(1).to(device)
        y_flat = torch.from_numpy(np.asarray(y.flatten(), dtype=np.float32)).unsqueeze(1).to(device)
        z_flat = torch.from_numpy(np.asarray(z.flatten(), dtype=np.float32)).unsqueeze(1).to(device)
        
        outputs = model(x_flat, y_flat, z_flat)
        
        # 提取16个特征
        features = {
            # 1-5: 密度特征
            'density_mean': float(outputs['density'].mean().item()),
            'density_std': float(outputs['density'].std().item()),
            'density_min': float(outputs['density'].min().item()),
            'density_max': float(outputs['density'].max().item()),
            'density_cv': float((outputs['density'].std() / (outputs['density'].mean() + 1e-8)).item()),
            
            # 6-7: 含水率特征
            'moisture_mean': float(outputs['moisture'].mean().item()),
            'moisture_std': float(outputs['moisture'].std().item()),
            
            # 8-9: 孔隙率特征
            'porosity_mean': float(outputs['porosity'].mean().item()),
            'porosity_std': float(outputs['porosity'].std().item()),
            
            # 10-13: 各向异性特征
            'anisotropy_x': float(outputs['anisotropy'][:, 0].mean().item()),
            'anisotropy_y': float(outputs['anisotropy'][:, 1].mean().item()),
            'anisotropy_z': float(outputs['anisotropy'][:, 2].mean().item()),
            'anisotropy_magnitude': float(torch.norm(outputs['anisotropy'], dim=1).mean().item()),
            
            # 14-15: 心材特征
            'heartwood_ratio': float(outputs['heartwood'].mean().item()),
            'heartwood_uniformity': float((1 - outputs['heartwood'].std()).item()),
            
            # 16: 均匀性
            'homogeneity': float(outputs['homogeneity'].mean().item()),
        }
    
    return features


def process_all_samples(augmentation_info_path='./step2_augmented_data/augmentation_info.json',
                       output_dir='./step3_features'):
    """处理所有扩充的样本 (NumPy 2.x兼容版本)"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载扩充信息
    with open(augmentation_info_path, 'r', encoding='utf-8') as f:
        aug_info = json.load(f)
    
    samples = aug_info['samples']
    print(f"共 {len(samples)} 个样本需要处理")
    
    all_features = {}
    
    for idx, sample_info in enumerate(tqdm(samples, desc="提取特征"), 1):
        sample_id = sample_info['sample_id']
        npy_file = sample_info['npy_file']
        
        try:
            # 加载数据 - 显式转换为float32
            ct_data = np.load(npy_file)
            ct_data = np.asarray(ct_data, dtype=np.float32)
            
            # 创建数据集
            dataset = WoodDataset(ct_data, sample_id, sample_size=800)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            
            # 训练PINN
            model, history = train_pinn(dataloader, sample_id, epochs=20, lr=1e-3)
            
            # 提取特征
            features = extract_features(model, ct_data.shape)
            
            # 添加元数据
            features['sample_id'] = sample_id
            features['wood_name'] = sample_id.split('_')[0] + '_' + sample_id.split('_')[1]
            features['augmentation_method'] = sample_info['method']
            
            # 添加CT统计
            features['data_info'] = {
                'ct_mean': float(sample_info['ct_mean']),
                'ct_std': float(sample_info['ct_std']),
                'ct_min': int(sample_info['ct_min']),
                'ct_max': int(sample_info['ct_max']),
                'shape': sample_info['shape'],
                'non_zero_ratio': float(np.count_nonzero(ct_data) / ct_data.size)
            }
            
            all_features[sample_id] = features
            
        except Exception as e:
            print(f"\n处理 {sample_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存特征
    features_path = output_dir / 'extracted_features.json'
    with open(features_path, 'w', encoding='utf-8') as f:
        json.dump(all_features, f, indent=2, ensure_ascii=False)
    
    print(f"\n特征提取完成！")
    print(f"共提取 {len(all_features)} 个样本的特征")
    print(f"保存路径: {features_path}")
    
    return all_features


def main():
    """主函数"""
    print("=" * 70)
    print("PINN特征提取 - 步骤3")
    print("=" * 70)
    
    # 检查版本
    print(f"\nNumPy版本: {np.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 提取所有样本的特征
    all_features = process_all_samples(
        augmentation_info_path='./step2_augmented_data/augmentation_info.json',
        output_dir='./step3_features'
    )
    
    # 统计
    if len(all_features) > 0:
        print("\n特征提取统计:")
        wood_counts = {}
        for sample_id, features in all_features.items():
            wood_name = features['wood_name']
            wood_counts[wood_name] = wood_counts.get(wood_name, 0) + 1
        
        for wood_name, count in sorted(wood_counts.items()):
            print(f"  {wood_name}: {count} 个样本")
        
        print("\n提取的16个特征:")
        sample_features = list(all_features.values())[0]
        feature_names = [k for k in sample_features.keys() if k not in ['sample_id', 'wood_name', 'augmentation_method', 'data_info']]
        for i, name in enumerate(feature_names, 1):
            print(f"  {i:2d}. {name}")
    
    print("\n" + "=" * 70)
    print("特征提取完成！")
    print("=" * 70)
    print("\n输出:")
    print("  - 特征文件: ./step3_features/extracted_features.json")
    
    print("\n下一步:")
    print("  运行 step4_quality_scoring.py 进行质量评分")


if __name__ == "__main__":
    main()
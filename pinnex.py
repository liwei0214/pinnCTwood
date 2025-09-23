"""
Physics-Informed Neural Network for Wood CT Physical Feature Extraction
Based on GB/T 10759-2012 Standard
Template version - configure values before use
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Sample configuration template - Replace None with actual values
WOOD_SAMPLES = {
    'Sample_1': {
        'y_bounds': (None, None),  # Y-axis boundaries in voxels
        'position': 'top',  # Position identifier
        'reference_density': None,  # Reference density in g/cm³
        'moisture_content': None,  # Moisture content (0-1 scale)
        'category': 'Type_A',  # Wood category
        'heartwood_ratio': None,  # Heartwood proportion (0-1)
        'color': '#000000'  # Visualization color
    },
    'Sample_2': {
        'y_bounds': (None, None),
        'position': 'middle',
        'reference_density': None,
        'moisture_content': None,
        'category': 'Type_B',
        'heartwood_ratio': None,
        'color': '#000000'
    },
    'Sample_3': {
        'y_bounds': (None, None),
        'position': 'bottom',
        'reference_density': None,
        'moisture_content': None,
        'category': 'Type_C',
        'heartwood_ratio': None,
        'color': '#000000'
    }
}

# Physical constants template - Replace None with actual values
WOOD_PHYSICS = {
    'cell_wall_density': None,  # Cell wall density g/cm³
    'water_density': None,  # Water density g/cm³
    'air_density': None,  # Air density g/cm³
    'oil_density': None,  # Wood extractives density g/cm³
    'ct_air': None,  # CT value for air (typically negative)
    'ct_water': None,  # CT value for water (reference)
    'ct_bone': None,  # CT reference value for calibration
}

# Default ranges if not specified (can be modified)
DEFAULT_RANGES = {
    'density_min': None,  # Minimum wood density g/cm³
    'density_max': None,  # Maximum wood density g/cm³
    'moisture_min': None,  # Minimum moisture content
    'moisture_max': None,  # Maximum moisture content
    'porosity_max': None,  # Maximum porosity
}


def validate_configuration():
    """Validate that required configuration values are set"""
    required_fields = ['reference_density', 'moisture_content', 'heartwood_ratio']
    physics_fields = ['cell_wall_density', 'water_density', 'ct_air', 'ct_water']

    for sample_name, sample_info in WOOD_SAMPLES.items():
        for field in required_fields:
            if sample_info.get(field) is None:
                raise ValueError(f"Missing required field '{field}' in {sample_name}")

    for field in physics_fields:
        if WOOD_PHYSICS.get(field) is None:
            raise ValueError(f"Missing required physics constant '{field}'")


class EnhancedWoodDataset(Dataset):
    """Enhanced wood dataset with physical features"""

    def __init__(self, ct_data, sample_name, sample_size=None, extract_gradients=True):
        self.sample_name = sample_name
        self.shape = ct_data.shape
        self.sample_size = sample_size or 1000  # Default if None
        self.sample_info = WOOD_SAMPLES[sample_name]

        # Calculate CT statistical features
        self.ct_mean = np.mean(ct_data)
        self.ct_std = np.std(ct_data)
        self.ct_skewness = stats.skew(ct_data.flatten())
        self.ct_kurtosis = stats.kurtosis(ct_data.flatten())

        # Create coordinate grid
        z, y, x = np.meshgrid(
            np.arange(self.shape[0]),
            np.arange(self.shape[1]),
            np.arange(self.shape[2]),
            indexing='ij'
        )

        # Normalize coordinates to [-1, 1]
        epsilon = 1e-8
        self.x_coords = (x.flatten() / (self.shape[2] + epsilon) * 2 - 1).astype(np.float32)
        self.y_coords = (y.flatten() / (self.shape[1] + epsilon) * 2 - 1).astype(np.float32)
        self.z_coords = (z.flatten() / (self.shape[0] + epsilon) * 2 - 1).astype(np.float32)

        # CT value processing
        self.ct_values = ct_data.flatten().astype(np.float32)
        self.ct_min = self.ct_values.min()
        self.ct_max = self.ct_values.max()

        # Multiple normalization methods
        self.ct_normalized = (self.ct_values - self.ct_min) / (self.ct_max - self.ct_min + epsilon)
        self.ct_standardized = (self.ct_values - self.ct_mean) / (self.ct_std + epsilon)

        # Compute gradient features
        if extract_gradients:
            self.gradients = self._compute_gradients(ct_data)
        else:
            self.gradients = np.zeros_like(self.ct_values)

        self.total_voxels = ct_data.size

        print(f"  {sample_name}:")
        print(f"    CT range: [{self.ct_min:.0f}, {self.ct_max:.0f}]")
        print(f"    CT mean±std: {self.ct_mean:.0f}±{self.ct_std:.0f}")
        print(f"    Skewness: {self.ct_skewness:.2f}, Kurtosis: {self.ct_kurtosis:.2f}")

    def _compute_gradients(self, ct_data):
        """Compute CT gradients (reflecting material changes)"""
        grad_z = np.gradient(ct_data, axis=0)
        grad_y = np.gradient(ct_data, axis=1)
        grad_x = np.gradient(ct_data, axis=2)
        grad_magnitude = np.sqrt(grad_z ** 2 + grad_y ** 2 + grad_x ** 2)
        return grad_magnitude.flatten().astype(np.float32)

    def __len__(self):
        return max(1, self.total_voxels // self.sample_size)

    def __getitem__(self, idx):
        indices = np.random.choice(self.total_voxels, self.sample_size, replace=False)

        return {
            'x': self.x_coords[indices].reshape(-1, 1),
            'y': self.y_coords[indices].reshape(-1, 1),
            'z': self.z_coords[indices].reshape(-1, 1),
            'ct_norm': self.ct_normalized[indices],
            'ct_std': self.ct_standardized[indices],
            'ct_raw': self.ct_values[indices],
            'gradients': self.gradients[indices],
            'moisture_ref': np.full(self.sample_size,
                                    self.sample_info['moisture_content'] or 0.1,
                                    dtype=np.float32)
        }


class WoodPINN(nn.Module):
    """Physics-Informed Neural Network for Wood Feature Extraction"""

    def __init__(self, sample_info, hidden_dim=None, num_layers=None):
        super().__init__()

        self.sample_info = sample_info
        self.ref_density = sample_info.get('reference_density', 1.0)
        self.ref_moisture = sample_info.get('moisture_content', 0.1)
        self.heartwood_ratio = sample_info.get('heartwood_ratio', 0.5)

        # Network architecture parameters
        hidden_dim = hidden_dim or 128
        num_layers = num_layers or 3

        # Build encoder layers
        layers = []
        input_dim = 3
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(input_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
            ])
            input_dim = out_dim

        self.encoder = nn.Sequential(*layers)

        # Multiple specialized output heads
        final_dim = hidden_dim // 2
        self.density_head = nn.Linear(final_dim, 1)
        self.moisture_head = nn.Linear(final_dim, 1)
        self.porosity_head = nn.Linear(final_dim, 1)
        self.anisotropy_head = nn.Linear(final_dim, 3)
        self.heartwood_head = nn.Linear(final_dim, 1)
        self.homogeneity_head = nn.Linear(final_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            gain = 0.5  # Default gain value
            nn.init.xavier_uniform_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y, z):
        # Handle input dimensions
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        if len(z.shape) == 1:
            z = z.unsqueeze(1)

        if len(x.shape) == 3:
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            z = z.reshape(-1, 1)

        coords = torch.cat([x, y, z], dim=1)

        # Encode features
        features = self.encoder(coords)

        # Extract physical properties
        # Use configurable ranges or defaults
        density_range = DEFAULT_RANGES.get('density_max', 1.5) - DEFAULT_RANGES.get('density_min', 0.5)
        density_min = DEFAULT_RANGES.get('density_min', 0.5)
        density = torch.sigmoid(self.density_head(features)) * density_range + density_min

        moisture_range = DEFAULT_RANGES.get('moisture_max', 0.2) - DEFAULT_RANGES.get('moisture_min', 0.05)
        moisture_min = DEFAULT_RANGES.get('moisture_min', 0.05)
        moisture = torch.sigmoid(self.moisture_head(features)) * moisture_range + moisture_min

        porosity_max = DEFAULT_RANGES.get('porosity_max', 0.5)
        porosity = torch.sigmoid(self.porosity_head(features)) * porosity_max

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


def physics_informed_loss(outputs, targets, sample_info):
    """Physics-informed loss function based on material constraints"""

    density = outputs['density']
    moisture = outputs['moisture']
    porosity = outputs['porosity']
    heartwood = outputs['heartwood']

    ct_norm = targets['ct_norm']
    moisture_ref = targets['moisture_ref']

    # Loss weight parameters (can be configured)
    weights = {
        'density': 1.0,
        'moisture': 0.5,
        'moisture_range': 0.3,
        'physics': 0.2,
        'heartwood': 0.1,
        'regularization': 0.001
    }

    # 1. Data fitting loss
    ref_density = sample_info.get('reference_density', 1.0)
    density_scale = 0.2  # Scaling factor for CT influence
    expected_density = ref_density - 0.1 + density_scale * ct_norm
    density_loss = torch.mean((density - expected_density) ** 2)

    # 2. Moisture constraint
    moisture_loss = torch.mean((moisture - moisture_ref) ** 2)

    # Moisture range constraint (if bounds are specified)
    moisture_min = DEFAULT_RANGES.get('moisture_min', 0.05)
    moisture_max = DEFAULT_RANGES.get('moisture_max', 0.2)
    moisture_range_loss = torch.mean(
        torch.relu(moisture_min - moisture) +
        torch.relu(moisture - moisture_max)
    )

    # 3. Physical constraint: density relationship
    if WOOD_PHYSICS['cell_wall_density'] is not None:
        rho_cell = WOOD_PHYSICS['cell_wall_density']
        physics_density = rho_cell * (1 - porosity) * (1 + moisture)
        physics_loss = torch.mean((density - physics_density) ** 2)
    else:
        physics_loss = torch.tensor(0.0, device=density.device)

    # 4. Heartwood ratio constraint
    target_heartwood = sample_info.get('heartwood_ratio', 0.5)
    heartwood_loss = torch.mean((heartwood - target_heartwood) ** 2)

    # 5. Regularization loss
    reg_loss = torch.mean(density ** 2) + torch.mean(porosity ** 2)

    # Combined loss
    total_loss = (
            weights['density'] * density_loss +
            weights['moisture'] * moisture_loss +
            weights['moisture_range'] * moisture_range_loss +
            weights['physics'] * physics_loss +
            weights['heartwood'] * heartwood_loss +
            weights['regularization'] * reg_loss
    )

    losses = {
        'total': total_loss,
        'density': density_loss,
        'moisture': moisture_loss,
        'physics': physics_loss,
        'heartwood': heartwood_loss
    }

    return losses


def train_pinn_model(dataloader, sample_name, sample_info,
                     epochs=None, learning_rate=None, weight_decay=None):
    """Train PINN model"""

    # Default training parameters
    epochs = epochs or 30
    learning_rate = learning_rate or 1e-3
    weight_decay = weight_decay or 1e-4

    print(f"\n{'=' * 60}")
    print(f"Training {sample_name} model")
    print(f"Category: {sample_info['category']}")

    if sample_info.get('reference_density'):
        print(f"Reference density: {sample_info['reference_density']} g/cm³")
    if sample_info.get('moisture_content'):
        print(f"Standard moisture: {sample_info['moisture_content'] * 100:.1f}%")

    model = WoodPINN(sample_info).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'total_loss': [],
        'density_loss': [],
        'moisture_loss': [],
        'physics_loss': []
    }

    for epoch in range(epochs):
        epoch_losses = {k: [] for k in history.keys()}

        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            model.train()

            # Prepare data
            x = torch.FloatTensor(batch['x']).to(device)
            y = torch.FloatTensor(batch['y']).to(device)
            z = torch.FloatTensor(batch['z']).to(device)

            targets = {
                'ct_norm': torch.FloatTensor(batch['ct_norm']).to(device).unsqueeze(1),
                'ct_std': torch.FloatTensor(batch['ct_std']).to(device).unsqueeze(1),
                'gradients': torch.FloatTensor(batch['gradients']).to(device).unsqueeze(1),
                'moisture_ref': torch.FloatTensor(batch['moisture_ref']).to(device).unsqueeze(1)
            }

            # Forward pass
            outputs = model(x, y, z)

            # Calculate loss
            losses = physics_informed_loss(outputs, targets, sample_info)

            # Optimize
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Record losses
            for key in epoch_losses.keys():
                if key in losses:
                    epoch_losses[key].append(losses[key.replace('_loss', '')].item())

        scheduler.step()

        # Record history
        for key in history.keys():
            if epoch_losses[key]:
                history[key].append(np.mean(epoch_losses[key]))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {history['total_loss'][-1]:.6f}")

    return model, history


def extract_comprehensive_features(model, sample_name, sample_shape, n_points=None):
    """Extract comprehensive physical features"""

    model.eval()
    n_points = n_points or 30  # Default sampling points

    # Uniform sampling in sample space
    z, y, x = np.meshgrid(
        np.linspace(-1, 1, n_points),
        np.linspace(-1, 1, n_points),
        np.linspace(-1, 1, n_points),
        indexing='ij'
    )

    with torch.no_grad():
        x_flat = torch.FloatTensor(x.flatten()).unsqueeze(1).to(device)
        y_flat = torch.FloatTensor(y.flatten()).unsqueeze(1).to(device)
        z_flat = torch.FloatTensor(z.flatten()).unsqueeze(1).to(device)

        outputs = model(x_flat, y_flat, z_flat)

        # Extract statistical features
        features = {
            'sample_name': sample_name,
            'category': WOOD_SAMPLES[sample_name]['category'],

            # Density features (g/cm³)
            'density_mean': float(outputs['density'].mean()),
            'density_std': float(outputs['density'].std()),
            'density_min': float(outputs['density'].min()),
            'density_max': float(outputs['density'].max()),
            'density_cv': float(outputs['density'].std() / (outputs['density'].mean() + 1e-8)),

            # Moisture features (%)
            'moisture_mean': float(outputs['moisture'].mean()) * 100,
            'moisture_std': float(outputs['moisture'].std()) * 100,

            # Porosity features
            'porosity_mean': float(outputs['porosity'].mean()),
            'porosity_std': float(outputs['porosity'].std()),

            # Anisotropy features
            'anisotropy_x': float(outputs['anisotropy'][:, 0].mean()),
            'anisotropy_y': float(outputs['anisotropy'][:, 1].mean()),
            'anisotropy_z': float(outputs['anisotropy'][:, 2].mean()),
            'anisotropy_magnitude': float(torch.norm(outputs['anisotropy'], dim=1).mean()),

            # Heartwood features
            'heartwood_ratio': float(outputs['heartwood'].mean()),
            'heartwood_uniformity': float(1 - outputs['heartwood'].std()),

            # Material homogeneity
            'homogeneity': float(outputs['homogeneity'].mean()),
        }

    return features


def analyze_ct_samples(ct_data, shape=None, output_dir='./results'):
    """Main analysis function for CT samples"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}
    all_models = {}

    # Process each sample
    for sample_name, sample_info in WOOD_SAMPLES.items():
        # Skip if required values are None
        if sample_info.get('y_bounds')[0] is None:
            print(f"Skipping {sample_name}: y_bounds not configured")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing {sample_name}")

        # Extract sample data
        y_start, y_end = sample_info['y_bounds']

        # Configurable sampling strategy
        z_range = slice(None, None, 2)  # Every 2nd slice in Z
        x_range = slice(None, None, 2)  # Every 2nd slice in X
        y_range = slice(y_start, y_end, 5)  # Every 5th slice in Y range

        sample_data = ct_data[z_range, y_range, x_range].copy()
        print(f"Sample shape: {sample_data.shape}")

        # Create dataset
        dataset = EnhancedWoodDataset(
            sample_data,
            sample_name,
            sample_size=800,  # Can be configured
            extract_gradients=True
        )

        dataloader = DataLoader(
            dataset,
            batch_size=8,  # Can be configured
            shuffle=True,
            num_workers=0
        )

        # Train model
        model, history = train_pinn_model(
            dataloader,
            sample_name,
            sample_info,
            epochs=25  # Can be configured
        )

        # Extract features
        features = extract_comprehensive_features(model, sample_name, sample_data.shape)

        all_results[sample_name] = features
        all_models[sample_name] = model

    # Save results
    results_path = os.path.join(output_dir, 'pinn_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAnalysis results saved to {results_path}")

    # Save models
    for sample_name, model in all_models.items():
        model_path = os.path.join(output_dir, f'pinn_model_{sample_name}.pth')
        torch.save(model.state_dict(), model_path)

    print("Models saved")

    return all_models, all_results


def load_configuration(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    global WOOD_SAMPLES, WOOD_PHYSICS, DEFAULT_RANGES
    WOOD_SAMPLES = config.get('samples', WOOD_SAMPLES)
    WOOD_PHYSICS = config.get('physics', WOOD_PHYSICS)
    DEFAULT_RANGES = config.get('ranges', DEFAULT_RANGES)

    return config


# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("PINN-based CT Feature Extraction System")
    print("=" * 60)
    print("Configuration Status:")
    print("  - All numerical values are set to None")
    print("  - Please configure WOOD_SAMPLES and WOOD_PHYSICS before use")
    print("  - Or load configuration from JSON file")
    print("=" * 60)

    # Example: Load configuration from file
    # config = load_configuration('config.json')

    # Example: Load CT data
    # ct_data = np.load('your_ct_data.npy')
    # or
    # ct_data = np.memmap('your_ct_file.raw', dtype=np.uint16, mode='r', shape=(z, y, x))

    # Example: Run analysis
    # models, results = analyze_ct_samples(ct_data)

    print("\nTo use this system:")
    print("1. Configure WOOD_SAMPLES with your sample parameters")
    print("2. Configure WOOD_PHYSICS with material constants")
    print("3. Load your CT data")
    print("4. Call analyze_ct_samples(ct_data)")
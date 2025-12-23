"""
Step 6 - Improved: Ablation Study & Robustness Testing with Proper Cross-Validation
Implements stratified 5-fold cross-validation with 3 repetitions for rigorous statistical validation
Compatible with NumPy 2.x
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from scipy import stats as sp_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AblationAndRobustnessImproved:
    """
    Improved ablation and robustness testing with proper cross-validation
    
    Key improvements over original:
    1. Stratified 5-fold cross-validation with 3 repetitions (15 total evaluations)
    2. Paired t-tests for statistical significance testing
    3. Cohen's kappa for inter-rater agreement
    4. VIF analysis for multicollinearity detection
    5. Proper mean ± std reporting across all metrics
    """
    
    def __init__(self, scores_path='./step4_quality_scores/quality_scores.csv'):
        """Initialize with quality scores"""
        self.scores_path = Path(scores_path)
        
        if not self.scores_path.exists():
            raise FileNotFoundError(f"Quality scores file not found: {self.scores_path}")
        
        self.df = pd.read_csv(scores_path)
        
        if len(self.df) == 0:
            raise ValueError("Quality scores file is empty")
        
        # Feature groups by quality dimension (paper Eq. 4)
        self.feature_groups = {
            'Density-Strength': [
                'density_mean', 'density_std', 'density_min', 
                'density_max', 'density_cv'
            ],
            'Structural Stability': [
                'homogeneity', 'anisotropy_x', 'anisotropy_y', 
                'anisotropy_z', 'anisotropy_magnitude'
            ],
            'Material Purity': [
                'heartwood_ratio', 'heartwood_uniformity', 
                'porosity_mean', 'porosity_std'
            ],
            'Processing Adaptability': [
                'moisture_mean', 'moisture_std'
            ]
        }
        
        print(f"Ablation & Robustness Testing Initialization")
        print(f"Loaded {len(self.df)} samples")
        
        # Store all metrics from cross-validation
        self.cv_results = {}
    
    def prepare_baseline(self):
        """Prepare baseline data for all analyses"""
        
        # Collect all features
        all_features = []
        for features in self.feature_groups.values():
            all_features.extend(features)
        
        # Check feature availability
        available_features = [f for f in all_features if f in self.df.columns]
        if len(available_features) < len(all_features):
            missing = set(all_features) - set(available_features)
            print(f"\nWarning: Missing features: {missing}")
            all_features = available_features
        
        self.all_features = all_features
        self.X = self.df[all_features].values.astype(np.float64)
        
        # Label mapping
        label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.y = np.array([label_map[label] for label in self.df['quality_grade'].values], 
                         dtype=np.int32)
        
        print(f"\nBaseline preparation complete")
        print(f"Features: {len(all_features)}")
        print(f"Samples: {len(self.y)}")
        print(f"Classes: {np.unique(self.y)}")
        
        return self.X, self.y
    
    def cross_validate_with_repetition(self, X, y, n_splits=5, n_repeats=3, 
                                       features_to_use=None, clf_type='linear'):
        """
        Stratified K-fold cross-validation with repetition
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of folds (default 5)
            n_repeats: Number of repetitions (default 3)
            features_to_use: Feature indices to use (None = all)
            clf_type: 'linear' or 'rbf' for SVM
            
        Returns:
            Dictionary with scores, kappas, and other metrics across all folds
        """
        
        scores = []
        kappas = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for repeat in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                                 random_state=42 + repeat)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Feature selection
                if features_to_use is not None:
                    X_train = X_train[:, features_to_use]
                    X_test = X_test[:, features_to_use]
                
                # Standardization
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Training
                kernel = 'linear' if clf_type == 'linear' else 'rbf'
                clf = SVC(kernel=kernel, random_state=42, C=1.0)
                clf.fit(X_train_scaled, y_train)
                
                # Evaluation
                y_pred = clf.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, y_pred)
                kappa = cohen_kappa_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted'
                )
                
                scores.append(acc)
                kappas.append(kappa)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        
        return {
            'scores': np.array(scores),
            'kappas': np.array(kappas),
            'precisions': np.array(precisions),
            'recalls': np.array(recalls),
            'f1_scores': np.array(f1_scores),
            'accuracy_mean': float(np.mean(scores)),
            'accuracy_std': float(np.std(scores)),
            'kappa_mean': float(np.mean(kappas)),
            'kappa_std': float(np.std(kappas)),
            'n_folds': n_splits * n_repeats
        }
    
    def paired_ttest(self, scores_alg1, scores_alg2, alg1_name, alg2_name):
        """
        Paired t-test comparing two algorithms across cross-validation folds
        
        Args:
            scores_alg1: Array of scores from algorithm 1 (15 folds)
            scores_alg2: Array of scores from algorithm 2 (15 folds)
            alg1_name: Name of algorithm 1
            alg2_name: Name of algorithm 2
            
        Returns:
            Dictionary with t-statistic, p-value, and interpretation
        """
        
        t_stat, p_value = sp_stats.ttest_rel(scores_alg1, scores_alg2)
        mean_diff = float(np.mean(scores_alg1) - np.mean(scores_alg2))
        std_diff = float(np.std(scores_alg1 - scores_alg2))
        
        # Significance marking
        if p_value < 0.001:
            sig_mark = "***"
            sig_text = "p < 0.001"
        elif p_value < 0.01:
            sig_mark = "**"
            sig_text = "p < 0.01"
        elif p_value < 0.05:
            sig_mark = "*"
            sig_text = "p < 0.05"
        else:
            sig_mark = "ns"
            sig_text = "p >= 0.05 (not significant)"
        
        print(f"\nPaired t-test: {alg1_name} vs {alg2_name}")
        print(f"  Mean difference: {mean_diff*100:+.2f}%")
        print(f"  t-statistic: {t_stat:.4f}, {sig_text} {sig_mark}")
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'significance': sig_mark
        }
    
    def calculate_vif(self, X, feature_names):
        """
        Calculate Variance Inflation Factor for multicollinearity analysis
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            DataFrame with VIF values
        """
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feature_names
        
        try:
            vif_data["VIF"] = [variance_inflation_factor(X, i) 
                              for i in range(X.shape[1])]
        except Exception as e:
            print(f"Warning: VIF calculation failed: {e}")
            return None
        
        # Sort by VIF descending
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        print("\n" + "="*70)
        print("Variance Inflation Factor (VIF) Analysis")
        print("="*70)
        print(vif_data.to_string(index=False))
        
        # Severity classification
        print("\nInterpretation:")
        for idx, row in vif_data.iterrows():
            vif_val = row['VIF']
            if vif_val < 5:
                severity = "Low (acceptable)"
            elif vif_val < 10:
                severity = "Moderate multicollinearity"
            else:
                severity = "Severe multicollinearity"
            print(f"  {row['Feature']:30s}: VIF={vif_val:7.2f} ({severity})")
        
        return vif_data
    
    def ablation_study(self):
        """
        Ablation study: test contribution of each quality dimension
        Uses proper cross-validation with statistical testing
        """
        
        print("\n" + "="*70)
        print("Ablation Study: Dimension Contributions (5-fold CV, 3 repeats)")
        print("="*70)
        
        # Baseline with all features
        baseline_results = self.cross_validate_with_repetition(
            self.X, self.y, clf_type='linear'
        )
        
        print(f"\nBaseline (all features):")
        print(f"  Accuracy: {baseline_results['accuracy_mean']*100:.2f}% ± {baseline_results['accuracy_std']*100:.2f}%")
        print(f"  Cohen's κ: {baseline_results['kappa_mean']:.3f} ± {baseline_results['kappa_std']:.3f}")
        
        baseline_scores = baseline_results['scores']
        
        ablation_results = {}
        
        # Test each dimension removal
        for dimension, features_to_remove in self.feature_groups.items():
            features_to_remove = [f for f in features_to_remove if f in self.all_features]
            
            if not features_to_remove:
                print(f"\nSkipping dimension: {dimension} (no available features)")
                continue
            
            print(f"\nRemoving dimension: {dimension}")
            print(f"  Removed features: {features_to_remove}")
            
            # Remaining features
            remaining_features = [f for f in self.all_features if f not in features_to_remove]
            
            if not remaining_features:
                print(f"  Warning: no remaining features")
                continue
            
            # Feature indices
            feature_indices = [self.all_features.index(f) for f in remaining_features]
            
            # Cross-validation on remaining features
            results = self.cross_validate_with_repetition(
                self.X, self.y, features_to_use=feature_indices, clf_type='linear'
            )
            
            ablated_scores = results['scores']
            
            # Paired t-test
            test_result = self.paired_ttest(
                baseline_scores, ablated_scores,
                "Baseline", f"Without {dimension}"
            )
            
            impact = baseline_results['accuracy_mean'] - results['accuracy_mean']
            impact_pct = impact * 100
            
            ablation_results[dimension] = {
                'baseline_accuracy': float(baseline_results['accuracy_mean']),
                'ablated_accuracy': float(results['accuracy_mean']),
                'impact': float(impact),
                'impact_percentage': float(impact_pct),
                'baseline_kappa': float(baseline_results['kappa_mean']),
                'ablated_kappa': float(results['kappa_mean']),
                't_statistic': test_result['t_statistic'],
                'p_value': test_result['p_value'],
                'significance': test_result['significance']
            }
            
            print(f"  Ablated accuracy: {results['accuracy_mean']*100:.2f}% ± {results['accuracy_std']*100:.2f}%")
            print(f"  Impact: {impact_pct:+.2f}%")
        
        self.ablation_results = ablation_results
        return ablation_results
    
    def noise_robustness_test(self, noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30]):
        """
        Test robustness to measurement noise using cross-validation
        """
        
        print("\n" + "="*70)
        print("Noise Robustness Test (5-fold CV, 3 repeats)")
        print("="*70)
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            print(f"\nNoise level: σ={noise_level}")
            
            if noise_level == 0.0:
                # Use clean data
                results = self.cross_validate_with_repetition(
                    self.X, self.y, clf_type='linear'
                )
                robustness_results[float(noise_level)] = {
                    'accuracy_mean': float(results['accuracy_mean']),
                    'accuracy_std': float(results['accuracy_std']),
                    'kappa_mean': float(results['kappa_mean']),
                    'kappa_std': float(results['kappa_std'])
                }
            else:
                # Add noise to test features
                X_noisy_list = []
                
                for repeat in range(3):
                    for fold in range(5):
                        np.random.seed(42 + repeat * 100 + fold)
                        noise = np.random.normal(0, noise_level, self.X.shape)
                        X_noisy = self.X + noise
                        X_noisy_list.append(X_noisy)
                
                # Average performance across noisy versions
                accuracies = []
                kappas = []
                
                for X_noisy in X_noisy_list:
                    results = self.cross_validate_with_repetition(
                        X_noisy, self.y, clf_type='linear'
                    )
                    accuracies.append(results['accuracy_mean'])
                    kappas.append(results['kappa_mean'])
                
                robustness_results[float(noise_level)] = {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'kappa_mean': float(np.mean(kappas)),
                    'kappa_std': float(np.std(kappas))
                }
            
            acc_mean = robustness_results[noise_level]['accuracy_mean']
            acc_std = robustness_results[noise_level]['accuracy_std']
            print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
        
        self.robustness_results = robustness_results
        return robustness_results
    
    def sample_efficiency_test(self, train_ratios=[0.3, 0.5, 0.7, 0.9, 1.0]):
        """
        Test performance with limited training data
        """
        
        print("\n" + "="*70)
        print("Sample Efficiency Test (5-fold CV, 3 repeats)")
        print("="*70)
        
        efficiency_results = {}
        
        for ratio in train_ratios:
            print(f"\nTraining set ratio: {ratio*100:.0f}%")
            
            if ratio == 1.0:
                results = self.cross_validate_with_repetition(
                    self.X, self.y, clf_type='linear'
                )
                efficiency_results[float(ratio)] = {
                    'train_samples': int(len(self.y) * 0.8),
                    'accuracy_mean': float(results['accuracy_mean']),
                    'accuracy_std': float(results['accuracy_std']),
                    'kappa_mean': float(results['kappa_mean']),
                    'kappa_std': float(results['kappa_std'])
                }
            else:
                # Subsample training data
                accuracies = []
                kappas = []
                train_sample_counts = []
                
                for repeat in range(3):
                    skf = StratifiedKFold(n_splits=5, shuffle=True, 
                                         random_state=42 + repeat)
                    
                    for train_idx, test_idx in skf.split(self.X, self.y):
                        X_train, X_test = self.X[train_idx], self.X[test_idx]
                        y_train, y_test = self.y[train_idx], self.y[test_idx]
                        
                        # Subsample training data
                        np.random.seed(42 + repeat)
                        n_samples = int(len(y_train) * ratio)
                        subsample_idx = np.random.choice(
                            len(y_train), n_samples, replace=False
                        )
                        X_train_sub = X_train[subsample_idx]
                        y_train_sub = y_train[subsample_idx]
                        
                        # Standardization
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_sub)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Training and evaluation
                        clf = SVC(kernel='linear', random_state=42)
                        clf.fit(X_train_scaled, y_train_sub)
                        
                        y_pred = clf.predict(X_test_scaled)
                        acc = accuracy_score(y_test, y_pred)
                        kappa = cohen_kappa_score(y_test, y_pred)
                        
                        accuracies.append(acc)
                        kappas.append(kappa)
                        train_sample_counts.append(n_samples)
                
                efficiency_results[float(ratio)] = {
                    'train_samples': int(np.mean(train_sample_counts)),
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'kappa_mean': float(np.mean(kappas)),
                    'kappa_std': float(np.std(kappas))
                }
            
            acc_mean = efficiency_results[ratio]['accuracy_mean']
            acc_std = efficiency_results[ratio]['accuracy_std']
            train_samples = efficiency_results[ratio]['train_samples']
            print(f"  Training samples: {train_samples}")
            print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
        
        self.efficiency_results = efficiency_results
        return efficiency_results
    
    def save_results(self, output_dir='./step6_ablation_robustness'):
        """Save comprehensive results"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Prepare report
        report = {
            'methodology': {
                'cross_validation': 'Stratified 5-fold with 3 repetitions (15 folds total)',
                'classifier': 'SVM (linear kernel)',
                'n_folds': 15
            },
            'baseline': {},
            'ablation_study': {},
            'noise_robustness': {},
            'sample_efficiency': {}
        }
        
        # Add ablation results
        for dimension, metrics in self.ablation_results.items():
            report['ablation_study'][dimension] = {
                'ablated_accuracy': float(metrics['ablated_accuracy']),
                'impact_percentage': float(metrics['impact_percentage']),
                'p_value': float(metrics['p_value']),
                'significance': metrics['significance']
            }
        
        # Add robustness results
        for noise, metrics in self.robustness_results.items():
            report['noise_robustness'][f'sigma_{float(noise)}'] = {
                'accuracy_mean': float(metrics['accuracy_mean']),
                'accuracy_std': float(metrics['accuracy_std'])
            }
        
        # Add efficiency results
        for ratio, metrics in self.efficiency_results.items():
            report['sample_efficiency'][f'{int(ratio*100)}%'] = {
                'train_samples': int(metrics['train_samples']),
                'accuracy_mean': float(metrics['accuracy_mean']),
                'accuracy_std': float(metrics['accuracy_std'])
            }
        
        # Save JSON
        json_path = output_dir / 'ablation_robustness_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nJSON report saved: {json_path}")
        
        self.visualize_results(output_dir)
    
    def visualize_results(self, output_dir):
        """Visualize comprehensive results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Ablation results
        ax1 = axes[0, 0]
        dimensions = list(self.ablation_results.keys())
        impacts = [self.ablation_results[d]['impact_percentage'] for d in dimensions]
        
        colors = ['red' if imp < 0 else 'green' if imp == 0 else 'orange' 
                  for imp in impacts]
        bars = ax1.barh(dimensions, impacts, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_xlabel('Accuracy Impact (%)')
        ax1.set_title('(a) Ablation Study: Dimension Contributions\n(Negative = Performance Drop)')
        ax1.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, imp) in enumerate(zip(bars, impacts)):
            sig = self.ablation_results[dimensions[i]]['significance']
            x_pos = imp + (0.5 if imp > 0 else -0.5)
            ax1.text(x_pos, i, f'{imp:.2f}% {sig}', va='center', 
                    ha='left' if imp > 0 else 'right', fontsize=9)
        
        # 2. Noise robustness
        ax2 = axes[0, 1]
        noise_levels = sorted(self.robustness_results.keys())
        accuracies = [self.robustness_results[n]['accuracy_mean'] * 100 
                      for n in noise_levels]
        stds = [self.robustness_results[n]['accuracy_std'] * 100 
               for n in noise_levels]
        
        ax2.errorbar(noise_levels, accuracies, yerr=stds, fmt='o-', 
                    linewidth=2, markersize=8, capsize=5, capthick=2)
        ax2.set_xlabel('Noise Level (σ)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('(b) Noise Robustness Test')
        ax2.set_ylim([70, 105])
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample efficiency
        ax3 = axes[1, 0]
        ratios = sorted(self.efficiency_results.keys())
        train_samples = [self.efficiency_results[r]['train_samples'] for r in ratios]
        eff_accuracies = [self.efficiency_results[r]['accuracy_mean'] * 100 for r in ratios]
        eff_stds = [self.efficiency_results[r]['accuracy_std'] * 100 for r in ratios]
        
        ax3.errorbar(train_samples, eff_accuracies, yerr=eff_stds, fmt='s-', 
                    linewidth=2, markersize=8, capsize=5, capthick=2)
        ax3.set_xlabel('Training Samples')
        ax3.set_ylabel('Accuracy (%)')
        ax3.set_title('(c) Sample Efficiency Test')
        ax3.set_ylim([70, 105])
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Find most important dimension
        max_impact_dim = max(self.ablation_results.items(),
                           key=lambda x: abs(x[1]['impact_percentage']))
        
        summary_data = [
            ['Metric', 'Value'],
            ['', ''],
            ['Ablation Study', ''],
            ['Most Critical Dimension', max_impact_dim[0]],
            ['Max Impact Value', f"{max_impact_dim[1]['impact_percentage']:.2f}%"],
            ['Significance', max_impact_dim[1]['significance']],
            ['', ''],
            ['Robustness (σ=0.10)', f"{self.robustness_results[0.10]['accuracy_mean']*100:.2f}%"],
            ['', ''],
            ['Sample Efficiency', ''],
            ['Accuracy @ 30% Data', f"{self.efficiency_results[0.3]['accuracy_mean']*100:.2f}%"],
            ['Accuracy @ 70% Data', f"{self.efficiency_results[0.7]['accuracy_mean']*100:.2f}%"],
        ]
        
        table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('(d) Experiment Summary', fontsize=11, fontweight='bold', pad=20)
        
        plt.suptitle('Ablation Study & Robustness Testing Results\n(5-fold CV, 3 repeats, 15 evaluations)', 
                    fontsize=13, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(output_dir / 'ablation_robustness.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Result figure saved: {output_dir / 'ablation_robustness.png'}")


def main():
    """Main function"""
    print("="*70)
    print("Ablation Study & Robustness Testing - Step 6 (Improved)")
    print("With Proper Cross-Validation & Statistical Validation")
    print("="*70)
    
    try:
        exp = AblationAndRobustnessImproved(
            scores_path='./step4_quality_scores/quality_scores.csv'
        )
        
        # Prepare data
        X, y = exp.prepare_baseline()
        
        # VIF analysis for multicollinearity
        print("\nPerforming VIF analysis...")
        vif_df = exp.calculate_vif(X, exp.all_features)
        
        # Ablation study
        ablation_results = exp.ablation_study()
        
        # Noise robustness
        robustness_results = exp.noise_robustness_test(
            noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30]
        )
        
        # Sample efficiency
        efficiency_results = exp.sample_efficiency_test(
            train_ratios=[0.3, 0.5, 0.7, 0.9, 1.0]
        )
        
        # Save results
        exp.save_results(output_dir='./step6_ablation_robustness')
        
        # Print summary
        print("\n" + "="*70)
        print("Ablation & Robustness Testing Complete!")
        print("="*70)
        
        print("\nKey Findings:")
        
        if ablation_results:
            max_impact_dim, max_impact_val = max(
                ablation_results.items(),
                key=lambda x: abs(x[1]['impact_percentage'])
            )
            print(f"  Most critical dimension: {max_impact_dim}")
            print(f"  Impact on removal: {max_impact_val['impact_percentage']:.2f}% {max_impact_val['significance']}")
        
        if 0.10 in robustness_results:
            print(f"  Noise robustness (σ=0.10): {robustness_results[0.10]['accuracy_mean']*100:.2f}%")
        
        if 0.3 in efficiency_results:
            print(f"  Sample efficiency (30% data): {efficiency_results[0.3]['accuracy_mean']*100:.2f}%")
        
        print("\nOutputs:")
        print("  - Results JSON: ./step6_ablation_robustness/ablation_robustness_results.json")
        print("  - Results figure: ./step6_ablation_robustness/ablation_robustness.png")
        
        print("\n✓ All experiments completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

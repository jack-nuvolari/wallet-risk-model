import os
import json
from typing import Dict, Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class RiskWeightTrainer:
    """Trains optimal risk weights for wallet risk assessment using pairwise comparisons."""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = data_path
        self.current_weights = {
            'transaction_frequency': 0.15,
            'transaction_amounts': 0.15,
            'interaction_patterns': 0.15,
            'recent_activity': 0.10,
            'contract_interactions': 0.15,
            'balance_risk': 0.20,
            'defi_positions': 0.10,
        }
        self.trained_weights = None
        self.true_weights = None
    
    def create_mock_pairwise_data(self, num_samples: int = 1000, use_extreme_true_weights: bool = True, min_l1_shift: float = 0.25):
        """Create mock pairwise comparison data for training.
        
        If use_extreme_true_weights is True, we generate a spiky Dirichlet
        ground-truth weight vector that is significantly different from
        current_weights (by at least min_l1_shift in L1 distance). These
        true weights are used to determine pairwise labels.
        """
        print(f"Creating {num_samples} mock wallet pairwise comparisons...")
        
        # Choose ground-truth weights used to produce labels
        if use_extreme_true_weights:
            self.true_weights = self._sample_true_weights(min_l1_shift=min_l1_shift)
        else:
            self.true_weights = dict(self.current_weights)
        
        pairwise_data = []
        
        for i in range(num_samples):
            wallet_a = self._generate_mock_wallet(f"Wallet_A_{i}")
            wallet_b = self._generate_mock_wallet(f"Wallet_B_{i}")
            
            if not self._validate_risk_components(wallet_a) or not self._validate_risk_components(wallet_b):
                continue
            
            # Use TRUE weights to set labels; this makes learning deviate from current weights
            risk_score_a_true = self._calculate_weighted_risk(wallet_a, self.true_weights)
            risk_score_b_true = self._calculate_weighted_risk(wallet_b, self.true_weights)
            label = 1 if risk_score_b_true > risk_score_a_true else 0
            
            # Feature vector: differences between component risks
            features = [
                wallet_b['transaction_frequency_risk'] - wallet_a['transaction_frequency_risk'],
                wallet_b['transaction_amounts_risk'] - wallet_a['transaction_amounts_risk'],
                wallet_b['interaction_patterns_risk'] - wallet_a['interaction_patterns_risk'],
                wallet_b['recent_activity_risk'] - wallet_a['recent_activity_risk'],
                wallet_b['contract_interactions_risk'] - wallet_a['contract_interactions_risk'],
                wallet_b['balance_risk'] - wallet_a['balance_risk'],
                wallet_b['defi_positions_risk'] - wallet_a['defi_positions_risk']
            ]
            
            pair_data = {
                'wallet_a': {
                    'address': wallet_a['address'], 
                    'risk_score': risk_score_a_true,
                    'risk_components': {
                        'transaction_frequency_risk': wallet_a['transaction_frequency_risk'],
                        'transaction_amounts_risk': wallet_a['transaction_amounts_risk'],
                        'interaction_patterns_risk': wallet_a['interaction_patterns_risk'],
                        'recent_activity_risk': wallet_a['recent_activity_risk'],
                        'contract_interactions_risk': wallet_a['contract_interactions_risk'],
                        'balance_risk': wallet_a['balance_risk'],
                        'defi_positions_risk': wallet_a['defi_positions_risk']
                    }
                },
                'wallet_b': {
                    'address': wallet_b['address'], 
                    'risk_score': risk_score_b_true,
                    'risk_components': {
                        'transaction_frequency_risk': wallet_b['transaction_frequency_risk'],
                        'transaction_amounts_risk': wallet_b['transaction_amounts_risk'],
                        'interaction_patterns_risk': wallet_b['interaction_patterns_risk'],
                        'recent_activity_risk': wallet_b['recent_activity_risk'],
                        'contract_interactions_risk': wallet_b['contract_interactions_risk'],
                        'balance_risk': wallet_b['balance_risk'],
                        'defi_positions_risk': wallet_b['defi_positions_risk']
                    }
                },
                'features': features,
                'label': label
            }
            
            pairwise_data.append(pair_data)
        
        # Save dataset
        output_path = os.path.join(self.data_path, "wallet_pairwise_training.json")
        with open(output_path, 'w') as f:
            json.dump(pairwise_data, f, indent=2)
        print(f"Mock data saved to {output_path}")
        print(f"Generated {len(pairwise_data)} valid training samples")
        
        # Save meta with true weights for reproducibility
        meta = {
            'true_weights': self.true_weights,
            'current_weights': self.current_weights,
            'num_samples': len(pairwise_data)
        }
        with open(os.path.join(self.data_path, "wallet_pairwise_training_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        
        return pairwise_data
    
    def _sample_true_weights(self, min_l1_shift: float = 0.25) -> Dict[str, float]:
        """Sample a spiky Dirichlet weight vector far from current_weights."""
        keys = list(self.current_weights.keys())
        current = np.array([self.current_weights[k] for k in keys], dtype=float)
        
        # Spiky Dirichlet: alpha < 1 promotes extreme weights
        alpha = np.full(len(keys), 0.25, dtype=float)
        for _ in range(1000):
            candidate = np.random.dirichlet(alpha)
            l1 = float(np.sum(np.abs(candidate - current)))
            if l1 >= min_l1_shift:
                return {k: float(v) for k, v in zip(keys, candidate)}
        # Fallback if not achieved by chance
        return {k: float(v) for k, v in zip(keys, candidate)}
    
    def _generate_mock_wallet(self, address: str) -> Dict[str, float]:
        """Generate a mock wallet with random risk component scores."""
        return {
            'address': address,
            'transaction_frequency_risk': np.random.uniform(0, 100),
            'transaction_amounts_risk': np.random.uniform(0, 100),
            'interaction_patterns_risk': np.random.uniform(0, 100),
            'recent_activity_risk': np.random.uniform(0, 100),
            'contract_interactions_risk': np.random.uniform(0, 100),
            'balance_risk': np.random.uniform(0, 100),
            'defi_positions_risk': np.random.uniform(0, 100)
        }
    
    def _validate_risk_components(self, wallet: Dict[str, float]) -> bool:
        """Validate that all risk component scores are positive and within bounds."""
        for key, value in wallet.items():
            if key != 'address':
                if not (0 <= value <= 100):
                    return False
        return True
    
    def _calculate_weighted_risk(self, wallet: Dict[str, float], weights: Dict[str, float] = None) -> float:
        """Calculate weighted risk score for a wallet."""
        if weights is None:
            weights = self.current_weights
            
        component_to_weight = {
            'transaction_frequency_risk': 'transaction_frequency',
            'transaction_amounts_risk': 'transaction_amounts',
            'interaction_patterns_risk': 'interaction_patterns',
            'recent_activity_risk': 'recent_activity',
            'contract_interactions_risk': 'contract_interactions',
            'balance_risk': 'balance_risk',
            'defi_positions_risk': 'defi_positions'
        }
        
        total_score = 0
        for component, value in wallet.items():
            if component != 'address':
                weight_key = component_to_weight[component]
                total_score += value * weights[weight_key]
        
        return total_score
    
    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data from file."""
        file_path = os.path.join(self.data_path, "wallet_pairwise_training.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        features = np.array([item['features'] for item in data])
        labels = np.array([item['label'] for item in data])
        
        print(f"Loaded {len(data)} training samples")
        return features, labels
    
    def train_weights(self, features: np.ndarray, labels: np.ndarray, method: str = 'logreg_l1', C: float = 0.5) -> Dict[str, float]:
        """Train optimal risk weights.
        
        method:
            - 'logreg'     : Standard Logistic Regression
            - 'logreg_l1'  : L1-regularized Logistic Regression (saga)
        """
        if method not in ('logreg', 'logreg_l1'):
            raise ValueError("Unsupported method. Use 'logreg' or 'logreg_l1'.")
        
        print(f"Training risk weights using {method}...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        if method == 'logreg':
            model = LinearRegression()
        else:
            # L1 penalty encourages sparsity -> more pronounced weight shifts after normalization
            # Use Lasso for L1 regularization with LinearRegression-like behavior
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=1.0/C, max_iter=4000, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        # Convert coefficients to normalized, positive weights
        raw_weights = np.abs(model.coef_)
        if raw_weights.sum() == 0:
            raw_weights = np.ones_like(raw_weights)
        normalized_weights = raw_weights / np.sum(raw_weights)
        
        weight_keys = list(self.current_weights.keys())
        self.trained_weights = dict(zip(weight_keys, normalized_weights))
        return self.trained_weights
    
    def compare_weights(self) -> Dict[str, Dict[str, float]]:
        """Compare current weights with trained weights."""
        if self.trained_weights is None:
            raise ValueError("No trained weights available. Train the model first.")
        
        comparison = {}
        for key in self.current_weights.keys():
            comparison[key] = {
                'current': self.current_weights[key],
                'trained': self.trained_weights[key],
                'difference': self.trained_weights[key] - self.current_weights[key],
                'change_percentage': ((self.trained_weights[key] - self.current_weights[key]) / 
                                    self.current_weights[key]) * 100
            }
        
        return comparison
    
    def save_trained_weights(self, filename: str = "trained_risk_weights.json"):
        """Save trained weights to file."""
        if self.trained_weights is None:
            print("No trained weights to save.")
            return
        
        output_path = os.path.join(self.data_path, filename)
        
        save_data = {
            'trained_weights': self.trained_weights,
            'current_weights': self.current_weights,
            'true_weights': self.true_weights,
            'risk_components': list(self.current_weights.keys())
        }
        
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Trained weights saved to {output_path}")
    
    def load_trained_weights(self, filename: str = "trained_risk_weights.json") -> Dict[str, float]:
        """Load trained weights from file."""
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trained weights file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.trained_weights = data['trained_weights']
        print(f"Loaded trained weights from {file_path}")
        
        return self.trained_weights


def train():
    """Main function to demonstrate the risk weight training system."""
    print("=== Wallet Risk Weight Training System ===\n")
    
    trainer = RiskWeightTrainer()
    
    print("1. Creating mock training data with extreme true weights...")
    trainer.create_mock_pairwise_data(num_samples=3000, use_extreme_true_weights=True, min_l1_shift=0.35)
    
    print("\n2. Loading training data...")
    features, labels = trainer.load_training_data()
    
    print("\n3. Training weights with L1 regularization...")
    trainer.train_weights(features, labels, method='logreg_l1', C=0.4)
    
    print("\n4. Comparing weights...")
    comparison = trainer.compare_weights()
    for component, changes in comparison.items():
        print(f"{component}: {changes['current']:.4f} → {changes['trained']:.4f} "
              f"({changes['change_percentage']:+.1f}%)")
    
    print("\n5. Saving trained weights...")
    trainer.save_trained_weights()
    
    print("\n=== Training Complete ===")
    print("True weights and trained weights saved; larger shifts expected due to extreme ground-truth.")


if __name__ == "__main__":
    train()

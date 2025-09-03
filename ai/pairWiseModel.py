import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

class PairwiseCoinRiskModel:
    """
    A pairwise learning approach for predicting coin risk scores.
    Uses relative comparisons between coins to train regression models.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the pairwise risk model.
        
        Args:
            model_type: Type of regression model ('random_forest', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.training_features = None
        self.training_risk_scores = None
        
    def _create_model(self):
        """Create the specified regression model."""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _extract_features(self, coin_data):
        """
        Extract numerical features from coin data.
        
        Args:
            coin_data: List of coin dictionaries
            
        Returns:
            DataFrame with features and risk scores
        """
        features = []
        risk_scores = []
        
        for coin in coin_data:
            # Extract numerical features
            feature_vector = [
                coin.get('market_cap', 0),
                coin.get('total_supply', 0),
                coin.get('circulating_supply', 0),
                coin.get('current_price', 0),
                coin.get('price_change_24h', 0),
                coin.get('total_volume', 0)
            ]
            
            # Add derived features
            if coin.get('total_supply', 0) > 0:
                supply_utilization = coin.get('circulating_supply', 0) / coin.get('total_supply', 1)
                feature_vector.append(supply_utilization)
            else:
                feature_vector.append(0)
            
            # Price volatility (normalized by current price)
            if coin.get('current_price', 0) > 0:
                volatility = abs(coin.get('price_change_24h', 0)) / coin.get('current_price', 1)
                feature_vector.append(volatility)
            else:
                feature_vector.append(0)
            
            # Market cap to volume ratio
            if coin.get('total_volume', 0) > 0:
                cap_volume_ratio = coin.get('market_cap', 0) / coin.get('total_volume', 1)
                feature_vector.append(cap_volume_ratio)
            else:
                feature_vector.append(0)
            
            features.append(feature_vector)
            risk_scores.append(float(coin.get('risk', 0)))
        
        # Define feature names
        self.feature_names = [
            'market_cap', 'total_supply', 'circulating_supply', 'current_price',
            'price_change_24h', 'total_volume', 'supply_utilization',
            'price_volatility', 'cap_volume_ratio'
        ]
        
        return pd.DataFrame(features, columns=self.feature_names), np.array(risk_scores)
    
    def _create_pairwise_data(self, features, risk_scores):
        """
        Create pairwise comparison data for training.
        
        Args:
            features: Feature matrix
            risk_scores: Risk score array
            
        Returns:
            Tuple of (pairwise_features, pairwise_labels)
        """
        n_samples = len(features)
        pairwise_features = []
        pairwise_labels = []
        
        # Create all possible pairs
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Feature difference between coins
                feature_diff = features.iloc[i].values - features.iloc[j].values
                
                # Label: 1 if coin i has higher risk than coin j, 0 otherwise
                label = 1 if risk_scores[i] > risk_scores[j] else 0
                
                pairwise_features.append(feature_diff)
                pairwise_labels.append(label)
        
        return np.array(pairwise_features), np.array(pairwise_labels)
    
    def _train_with_ranking_loss(self, features, risk_scores):
        """
        Train using ranking loss approach.
        """
        # Store training data for later use
        self.training_features = features
        self.training_risk_scores = risk_scores
        
        # Create pairwise data
        pairwise_features, pairwise_labels = self._create_pairwise_data(features, risk_scores)
        
        # Split pairwise data
        X_train, X_test, y_train, y_test = train_test_split(
            pairwise_features, pairwise_labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model on pairwise data
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate pairwise accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        print(f"Pairwise Training Accuracy: {train_acc:.4f}")
        print(f"Pairwise Test Accuracy: {test_acc:.4f}")
        
        return train_acc, test_acc
   
    def _predict_ranking_score(self, coin_features):
        """
        Predict a ranking score using the pairwise model.
        This score represents how the coin ranks relative to training data.
        """
        # Extract features
        feature_vector = self._extract_single_features(coin_features)
        
        # Scale features
        features_scaled = self.scaler.transform([feature_vector])
        
        # Get pairwise predictions against all training samples
        ranking_scores = []
        
        for i, train_features in enumerate(self.training_features.values):
            # Create feature difference (coin vs training sample)
            feature_diff = feature_vector - train_features
            feature_diff_scaled = self.scaler.transform([feature_diff])
            
            # Predict probability that our coin has higher risk
            if hasattr(self.model, 'predict_proba'):
                prob_higher_risk = self.model.predict_proba(feature_diff_scaled)[0][1]
            else:
                # For models without predict_proba, use decision function
                prob_higher_risk = self.model.predict(feature_diff_scaled)[0]
                # Convert to probability-like score
                prob_higher_risk = 1 / (1 + np.exp(-prob_higher_risk))
            
            ranking_scores.append(prob_higher_risk)
        
        # Convert ranking scores to a single risk score
        # Higher ranking score means higher risk
        avg_ranking_score = np.mean(ranking_scores)
        
        # Map to risk score range (0-1) based on training data distribution
        risk_percentile = np.percentile(self.training_risk_scores, avg_ranking_score * 100)
        
        return risk_percentile
    
    def _extract_single_features(self, coin_features):
        """Extract features for a single coin."""
        feature_vector = [
            coin_features.get('market_cap', 0),
            coin_features.get('total_supply', 0),
            coin_features.get('circulating_supply', 0),
            coin_features.get('current_price', 0),
            coin_features.get('price_change_24h', 0),
            coin_features.get('total_volume', 0)
        ]
        
        # Add derived features
        if coin_features.get('total_supply', 0) > 0:
            supply_utilization = coin_features.get('circulating_supply', 0) / coin_features.get('total_supply', 1)
            feature_vector.append(supply_utilization)
        else:
            feature_vector.append(0)
        
        if coin_features.get('current_price', 0) > 0:
            volatility = abs(coin_features.get('price_change_24h', 0)) / coin_features.get('current_price', 1)
            feature_vector.append(volatility)
        else:
            feature_vector.append(0)
        
        if coin_features.get('total_volume', 0) > 0:
            cap_volume_ratio = coin_features.get('market_cap', 0) / coin_features.get('total_volume', 1)
            feature_vector.append(cap_volume_ratio)
        else:
            feature_vector.append(0)
        
        return np.array(feature_vector)
    
    def train(self, coin_data):
        """
        Train the risk prediction model.
        
        Args:
            coin_data: List of coin dictionaries with features and risk scores
        """
        print(f"Training {self.model_type} model...")
        print(f"Dataset size: {len(coin_data)} coins")
        
        # Extract features and risk scores
        features, risk_scores = self._extract_features(coin_data)
        print(f"Features: {self.feature_names}")
        
        # Check if we have valid risk scores
        valid_risk_scores = [r for r in risk_scores if r > 0]
        print(f"Valid risk scores: {len(valid_risk_scores)}/{len(risk_scores)}")
        
        # Use pairwise learning approach
        print("Using pairwise learning approach...")
        train_acc, test_acc = self._train_with_ranking_loss(features, risk_scores)
        self.is_trained = True
        return train_acc, test_acc

    
    def predict_risk(self, coin_features):
        """
        Predict risk score for a single coin.
        
        Args:
            coin_features: Dictionary with coin features
            
        Returns:
            Predicted risk score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Use ranking-based prediction
        return self._predict_ranking_score(coin_features)

    
    def predict_risk_batch(self, coin_data_list):
        """
        Predict risk scores for multiple coins.
        
        Args:
            coin_data_list: List of coin feature dictionaries
            
        Returns:
            List of predicted risk scores
        """
        predictions = []
        for coin in coin_data_list:
            pred = self.predict_risk(coin)
            predictions.append(pred)
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance if the model supports it."""
        if not self.is_trained:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return None
    
    def save_model(self, filepath):
        """Save the trained model and scaler."""
        import pickle
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'training_features': self.training_features,
            'training_risk_scores': self.training_risk_scores
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.training_features = model_data.get('training_features')
        self.training_risk_scores = model_data.get('training_risk_scores')
        print(f"Model loaded from {filepath}")


def train():
    """Example usage of the PairwiseCoinRiskModel."""
    
    # Load coin data
    with open('data/coin_data.json', 'r') as f:
        coin_data = json.load(f)
    
    print("=== Coin Risk Prediction using Pairwise Learning ===")
    print(f"Loaded {len(coin_data)} coins")
    
    # Initialize model
    model = PairwiseCoinRiskModel(model_type='random_forest')
    
    # Train model
    print("\n--- Training Phase ---")
    results = model.train(coin_data)
    
    # Show feature importance
    print("\n--- Feature Importance ---")
    importance = model.get_feature_importance()
    if importance:
        for feature, imp in importance.items():
            print(f"{feature}: {imp:.4f}")
    
    # Test predictions
    print("\n--- Prediction Examples ---")
    test_coins = coin_data[:5]  # Test on first 5 coins
    
    for coin in test_coins:
        predicted_risk = model.predict_risk(coin)
        actual_risk = float(coin.get('risk', 0))
        print(f"{coin['name']} ({coin['symbol']}):")
        print(f"  Actual Risk: {actual_risk:.3f}")
        print(f"  Predicted Risk: {predicted_risk:.3f}")
        print(f"  Difference: {abs(predicted_risk - actual_risk):.3f}")
        print()
    
    # Save model
    model.save_model('ai/models/pairwise_coin_risk_model.pkl')
    
    print("=== Training Complete ===")


if __name__ == "__main__":
    train()


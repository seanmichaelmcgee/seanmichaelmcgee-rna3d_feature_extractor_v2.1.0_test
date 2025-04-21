
"""
Tests for the FeatureExtractor class.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from src.features.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    """Tests for the FeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_extractor = FeatureExtractor()
    
    def test_init(self):
        """Test initialization of FeatureExtractor."""
        self.assertIsNotNone(self.feature_extractor)
    
    @patch('src.features.feature_extractor.extract_thermodynamic_features')
    def test_extract_thermodynamic_features(self, mock_extract):
        """Test extraction of thermodynamic features."""
        # Configure mock
        mock_extract.return_value = {
            'struct.mfe': -10.5,
            'struct.ensemble_energy': -10.0,
            'struct.pairing_probs': np.eye(10)
        }
        
        # Call the method
        sequence = 'GAUCGAUCGA'
        features = self.feature_extractor.extract_thermodynamic_features(sequence)
        
        # Verify the mock was called
        mock_extract.assert_called_once_with(sequence)
        
        # Verify the result
        self.assertIn('struct.mfe', features)
        self.assertIn('struct.ensemble_energy', features)
        self.assertIn('struct.pairing_probs', features)
    
    @patch('src.features.feature_extractor.calculate_mutual_information')
    def test_extract_mi_features(self, mock_calculate):
        """Test extraction of mutual information features."""
        # Configure mock
        mock_calculate.return_value = {
            'mi.scores': np.array([0.1, 0.2, 0.3]),
            'mi.coupling_matrix': np.ones((3, 3)) * 0.1
        }
        
        # Call the method
        msa_sequences = ['AAA', 'AAA', 'AAA']
        features = self.feature_extractor.extract_mi_features(msa_sequences)
        
        # Verify the mock was called
        mock_calculate.assert_called_once()
        
        # Verify the result
        self.assertIn('mi.scores', features)
        self.assertIn('mi.coupling_matrix', features)

if __name__ == '__main__':
    unittest.main()
"""


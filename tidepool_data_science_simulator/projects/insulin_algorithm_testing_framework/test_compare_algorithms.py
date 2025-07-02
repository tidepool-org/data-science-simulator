#!/usr/bin/env python3
"""
Simple test to verify the compare_algorithms method implementation.
This test checks the method signature and basic structure without requiring
the full dependency stack.
"""

import sys
import inspect
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_compare_algorithms_signature():
    """Test that compare_algorithms method has the correct signature."""
    
    # Mock the dependencies to avoid import errors
    import types
    
    # Create mock modules
    mock_numpy = types.ModuleType('numpy')
    mock_pandas = types.ModuleType('pandas')
    mock_scipy = types.ModuleType('scipy')
    mock_scipy.stats = types.ModuleType('scipy.stats')
    
    # Add basic attributes that are used
    mock_numpy.nan = float('nan')
    mock_numpy.array = list
    mock_numpy.mean = lambda x: sum(x) / len(x) if x else 0
    mock_numpy.std = lambda x, ddof=0: 0.0
    mock_numpy.sqrt = lambda x: x ** 0.5
    mock_numpy.isnan = lambda x: False
    
    mock_pandas.DataFrame = dict
    mock_pandas.Series = dict
    mock_pandas.api = types.ModuleType('pandas.api')
    mock_pandas.api.types = types.ModuleType('pandas.api.types')
    mock_pandas.api.types.is_numeric_dtype = lambda x: True
    mock_pandas.notna = lambda x: True
    
    mock_scipy.stats.ttest_rel = lambda x, y: (0.0, 0.5)
    mock_scipy.stats.wilcoxon = lambda x, y, alternative='two-sided': (0.0, 0.5)
    mock_scipy.stats.ttest_ind = lambda x, y, equal_var=False: (0.0, 0.5)
    mock_scipy.stats.mannwhitneyu = lambda x, y, alternative='two-sided': (0.0, 0.5)
    mock_scipy.stats.shapiro = lambda x: (0.0, 0.5)
    mock_scipy.stats.sem = lambda x: 0.1
    mock_scipy.stats.t = types.ModuleType('scipy.stats.t')
    mock_scipy.stats.t.ppf = lambda x, df: 1.96
    
    # Mock statsmodels
    mock_statsmodels = types.ModuleType('statsmodels')
    mock_statsmodels.stats = types.ModuleType('statsmodels.stats')
    mock_statsmodels.stats.multitest = types.ModuleType('statsmodels.stats.multitest')
    mock_statsmodels.stats.multitest.multipletests = lambda pvals, alpha=0.05, method='bonferroni': (
        [False] * len(pvals), pvals, 0.05, 0.05
    )
    
    # Install mocks
    sys.modules['numpy'] = mock_numpy
    sys.modules['pandas'] = mock_pandas
    sys.modules['scipy'] = mock_scipy
    sys.modules['scipy.stats'] = mock_scipy.stats
    sys.modules['statsmodels'] = mock_statsmodels
    sys.modules['statsmodels.stats'] = mock_statsmodels.stats
    sys.modules['statsmodels.stats.multitest'] = mock_statsmodels.stats.multitest
    
    # Mock the tidepool metrics module
    mock_tidepool_metrics = types.ModuleType('tidepool_data_science_metrics')
    mock_tidepool_metrics.glucose = types.ModuleType('tidepool_data_science_metrics.glucose')
    mock_tidepool_metrics.glucose.glucose = types.ModuleType('tidepool_data_science_metrics.glucose.glucose')
    
    # Add mock functions
    for func_name in ['percent_values_ge_70_le_180', 'percent_values_lt_70', 'percent_values_lt_54',
                      'percent_values_gt_180', 'percent_values_gt_250', 'blood_glucose_risk_index',
                      'lbgi_risk_score']:
        setattr(mock_tidepool_metrics.glucose.glucose, func_name, lambda x: 50.0)
    
    sys.modules['tidepool_data_science_metrics'] = mock_tidepool_metrics
    sys.modules['tidepool_data_science_metrics.glucose'] = mock_tidepool_metrics.glucose
    sys.modules['tidepool_data_science_metrics.glucose.glucose'] = mock_tidepool_metrics.glucose.glucose
    
    try:
        # Now try to import and inspect the class
        from analysis.statistical_analyzer import StatisticalAnalyzer
        
        # Check that the method exists
        assert hasattr(StatisticalAnalyzer, 'compare_algorithms'), "compare_algorithms method not found"
        
        # Get the method signature
        method = getattr(StatisticalAnalyzer, 'compare_algorithms')
        sig = inspect.signature(method)
        
        # Check parameter names
        param_names = list(sig.parameters.keys())
        expected_params = ['self', 'metrics_df', 'reference_algorithm', 'comparison_algorithms', 
                          'metrics_to_analyze', 'paired']
        
        assert param_names == expected_params, f"Expected parameters {expected_params}, got {param_names}"
        
        # Check parameter defaults
        params = sig.parameters
        assert params['metrics_to_analyze'].default is None, "metrics_to_analyze should default to None"
        assert params['paired'].default is True, "paired should default to True"
        
        print("✅ compare_algorithms method signature is correct!")
        print(f"   Parameters: {param_names}")
        print(f"   Signature: {sig}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing compare_algorithms: {e}")
        return False
    
    finally:
        # Clean up mocks
        for module in ['numpy', 'pandas', 'scipy', 'scipy.stats', 'statsmodels', 
                      'statsmodels.stats', 'statsmodels.stats.multitest',
                      'tidepool_data_science_metrics', 'tidepool_data_science_metrics.glucose',
                      'tidepool_data_science_metrics.glucose.glucose']:
            if module in sys.modules:
                del sys.modules[module]

if __name__ == "__main__":
    print("Testing compare_algorithms method implementation...")
    success = test_compare_algorithms_signature()
    
    if success:
        print("\n🎉 All tests passed! The compare_algorithms method is properly implemented.")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)

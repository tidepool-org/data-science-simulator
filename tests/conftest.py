import os
import sys

# Add the post_processing directory to sys.path so that bare-name imports
# within algorithm_risk_comparison.py (e.g. 'from jira_risk_probabilities import ...')
# resolve correctly when pytest is invoked from the project root.
_POST_PROCESSING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'post_processing',
)
if _POST_PROCESSING_DIR not in sys.path:
    sys.path.insert(0, _POST_PROCESSING_DIR)

import sys
import os

# Prefer a sibling checkout of data-science-models (if present) over the
# installed package. Override with the DATA_SCIENCE_MODELS_PATH env var.
_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
local_models_path = os.environ.get(
    "DATA_SCIENCE_MODELS_PATH",
    os.path.abspath(os.path.join(_repo_root, "..", "..", "data-science-models")),
)
if os.path.isdir(local_models_path) and local_models_path not in sys.path:
    sys.path.insert(0, local_models_path)

# Now import
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


def test_insulin_model_parameters():
    """Test that different insulin types create different tau parameters"""

    model1 = SimpleMetabolismModel(
        insulin_sensitivity_factor=150,
        carb_insulin_ratio=20,
        patient_insulin_type="rapid_acting_adult"
    )

    model2 = SimpleMetabolismModel(
        insulin_sensitivity_factor=150,
        carb_insulin_ratio=20,
        patient_insulin_type="glargine"
    )

    print(f"Rapid model tau1={model1.insulin_model._tau1}, tau2={model1.insulin_model._tau2}")
    print(f"Glargine model tau1={model2.insulin_model._tau1}, tau2={model2.insulin_model._tau2}")

    # Add assertions to verify the models are different
    assert model1.insulin_model._tau1 != model2.insulin_model._tau1, "Tau1 values should be different"
    assert model1.insulin_model._tau2 != model2.insulin_model._tau2, "Tau2 values should be different"

    # Verify expected values
    assert model1.insulin_model._tau1 == 55, "Rapid acting should have tau1=55"
    assert model2.insulin_model._tau1 == 300, "Glargine should have tau1=300"


if __name__ == "__main__":
    # This allows you to run the file directly without pytest
    test_insulin_model_parameters()
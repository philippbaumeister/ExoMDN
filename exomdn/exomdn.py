import json
from pathlib import Path

from exomdn.widgets import PredictionWidget, LoadModelWidget


class ExoMDN:
    def __init__(self) -> None:
        self.model = None

        self.model_path = Path("./models")
        self.input_prompt = None
        self.prediction = None
        self.mixture_components = None

    @property
    def prediction_widget(self):
        if self.model is None:
            raise ValueError("No model loaded. Run 'load_model_widget' first.")
        return PredictionWidget(self)

    @property
    def load_model_widget(self):
        return LoadModelWidget(self, self.model_path)
    
    @property
    def rf(self):
        return self.model.rf_layers
    
    @property
    def rf_log_ratios(self):
        return self.model.rf_outputs
    
    @property
    def mf(self):
        return self.model.mf_layers

    @property
    def mf_logratios(self):
        return self.model.mf_outputs

    def load_dummy_model(self):
        self.model = DummyModel()


class DummyModel:
    def __init__(self) -> None:
        self.inputs = ["planet_mass", "planet_radius"]
        self.input_properties = {
            "planet_mass": {
                "description": "Planet mass",
                "min_value": 0.1,
                "max_value": 25,
                "default_value": 1,
                "label": "$M_P$",
                "unit": "$M_\\oplus$",
                "step": 0.1},
            "planet_radius": {
                "description": "Planet radius",
                "min_value": 0,
                "max_value": 10,
                "default_value": 1,
                "label": "$R_P$",
                "unit": "$R_\\oplus$",
                "step": 0.1}
            }

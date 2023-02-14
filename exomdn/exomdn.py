import json
from pathlib import Path

from exomdn.widgets import PredictionWidget, LoadModelWidget


class ExoMDN:
    def __init__(self, model_path="../models", data_path="../data") -> None:
        self.model = None

        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
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
    def rf_logratios(self):
        return self.model.rf_outputs
    
    @property
    def mf(self):
        return self.model.mf_layers

    @property
    def mf_logratios(self):
        return self.model.mf_outputs

    @property
    def logratios(self):
        return self.model.outputs


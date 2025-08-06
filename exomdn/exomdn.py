import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from exomdn.mdn_model import Model

from exomdn.widgets import PredictionWidget, LoadModelWidget


class ExoMDN:
    def __init__(self, model_path="./models", data_path="./data") -> None:
        self.model = None

        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.input_prompt = None
        self.prediction = None
        self.mixture_components = None

    def load_model(self, model_path):
        """Load a model from the specified path."""
        self.model = Model(model_path)

    def predict(self, x: List[List], samples: int = 10000, **keras_kws) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Predict the output for given input data.

        Args:
            x (List[List]): Input data for prediction.
            samples (int): Number of samples to generate from the model's mixture.
            **keras_kws: Additional keyword arguments for Keras prediction.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - DataFrame with predictions.
                - DataFrame with mixture components.
                - DataFrame with input prompts.

        Example:
            exomdn = ExoMDN()
            exomdn.load_model('path/to/model')
            predictions, components, input_prompt = exomdn.predict([[1.0, 1.0], [2.0, 2.0]], samples=1000)
        """
        df, df_components, input_prompt = self.model.predict(x, samples, **keras_kws)
        self.input_prompt = input_prompt
        self.prediction = df
        self.mixture_components = df_components
        return df, df_components, input_prompt

    def predict_with_error(self, x: List[List], errors: List[List], samples: Tuple[int, int] = (1000, 100),
                           extra_query="") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Predict the interior structure posterior distribution given the input parameters and their uncertainties.
        The list and order of inputs for a model can be accessed through the 'inputs' attribute.

        To predict the interior including measurement uncertainties, the model predicts multiple samples from the
        uncertainty distribution first, and then samples from each respective distribution. Therefore, the samples
        parameter is a tuple with the number of samples to draw first from the uncertainty distribution and then from
        each respective distribution (total samples = samples[0] * samples[1]).

        Args:
            x: a list of input values
            errors: a list of uncertainties corresponding to each input value
            samples: a tuple with the number of samples to draw first from the uncertainty distribution and then from
            each respective distribution (total samples = samples[0] * samples[1])
            extra_query: an additional query to filter the input samples

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - DataFrame with predictions.
                - DataFrame with mixture components.
                - DataFrame with input prompts.

        """
        df, df_components, input_prompt = self.model.predict_with_error(x, errors, samples, extra_query)
        self.input_prompt = input_prompt
        self.prediction = df
        self.mixture_components = df_components
        return df, df_components, input_prompt


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


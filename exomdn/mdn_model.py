import json
from pathlib import Path
import pandas as pd
import numpy as np

import joblib
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import exomdn.mdn_layer as mdn
import exomdn.log_ratio as logratio

from numpy.polynomial import Polynomial
from typing import List, Tuple, Union


class Model:
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)

        self.rng = np.random.default_rng()

        with open(self.model_path / "setup_parameters.json", "r") as f:
            self.parameters = json.load(f)
        self.model_name = self.parameters.get("model_id", self.model_path.name)
        self.components = self.parameters["architecture"]["components"]
        self.inputs = self.parameters["inputs"]
        self.outputs = self.parameters["outputs"]
        self.input_properties = self.parameters.get("input_properties", {})
        self.output_dim = len(self.outputs)

        self.preprocessor = joblib.load(self.model_path / "preprocessor.pkl")
        self.keras_model = tf.keras.models.load_model(self.model_path / f"model",
                                                      custom_objects={"MDN": mdn.MDN,
                                                                      "mdn_loss_func": mdn.get_mixture_loss_func(
                                                                          self.output_dim, self.components)})

        print(f"Loaded model '{self.model_name}'")
        print("=" * 65)
        print("Model architecture:\n")
        self.keras_model.summary()

        self.base = "core"
        self.layers = [f"{x}_{y}" for y in ["rf", "mf"] for x in ["core", "mantle", "water", "atmosphere"]]
        self.rf_base = f"{self.base}_rf"
        self.rf_outputs = self.outputs[:3]
        self.rf_layers = ["core_rf", "mantle_rf", "water_rf", "atmosphere_rf"]
        self._rf_layers_nb = sorted(set(self.rf_layers).difference({f"{self.base}_rf"}), key=self.rf_layers.index)
        self.mf_base = f"{self.base}_mf"
        self.mf_outputs = self.outputs[3:]
        self.mf_layers = ["core_mf", "mantle_mf", "water_mf", "atmosphere_mf"]
        self._mf_layers_nb = sorted(set(self.mf_layers).difference({f"{self.base}_mf"}), key=self.mf_layers.index)
        self._current_data = None
        self._current_components = None

    def construct_mixture(self, prediction: np.ndarray) -> tfd.Mixture:
        """

        Args:
            prediction:

        Returns:

        """
        out_mu, out_sigma, out_pi = tf.split(prediction,
                                             num_or_size_splits=[self.components * self.output_dim,
                                                                 self.components * self.output_dim,
                                                                 self.components],
                                             axis=-1, name="mdn_coef_split")
        # Construct the mixture models
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [self.output_dim] * self.components
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(mus, sigs)]
        mixture = tfd.Mixture(cat=cat, components=coll)
        return mixture

    def sample_from_mixture(self, mixture: tfd.Mixture, samples: int) -> pd.DataFrame:
        """

        Args:
            mixture:
            samples:

        Returns:

        """
        tf_sample = np.swapaxes(mixture.sample(samples).numpy(), 0, 1)
        sample = pd.DataFrame(tf_sample.reshape(-1, self.output_dim), columns=self.outputs)
        sample["prediction"] = np.repeat(np.arange(tf_sample.shape[0]), tf_sample.shape[1])
        sample = self.logratio_transform(sample, inverse=True)
        return sample

    def get_mixture_components(self, mixture: tfd.Mixture) -> pd.DataFrame:
        """

        Args:
            mixture:

        Returns:

        """
        means = np.swapaxes(np.array([component.mean().numpy() for component in mixture.components]), 0, 1)
        var = np.swapaxes(np.array([component.variance().numpy() for component in mixture.components]), 0, 1)
        logits = mixture.cat.logits.numpy()
        probs = np.array([mdn.softmax(logit) for logit in logits])
        df_mean = pd.DataFrame(means.reshape(-1, self.output_dim), columns=self.outputs)
        df_var = pd.DataFrame(var.reshape(-1, self.output_dim), columns=["var_" + out for out in self.outputs])

        df = pd.concat([df_mean, df_var], axis=1)
        df["weight"] = probs.reshape(-1)
        df["prediction"] = np.repeat(np.arange(means.shape[0]), means.shape[1])
        df = self.logratio_transform(df, inverse=True)
        return df

    def logratio_transform(self, df: pd.DataFrame, inverse: bool = False) -> pd.DataFrame:
        """

        Args:
            df:
            inverse:

        Returns:

        """
        if inverse:
            df = logratio.inv_transform_df(data=df,
                                           base=self.rf_base,
                                           columns=self.rf_outputs,
                                           new_columns=self._rf_layers_nb)
            df = logratio.inv_transform_df(data=df,
                                           base=self.mf_base,
                                           columns=self.mf_outputs,
                                           new_columns=self._mf_layers_nb)
        else:
            df = logratio.transform_df(data=df,
                                       base=self.rf_base,
                                       columns=self.rf_outputs,
                                       new_columns=self._rf_layers_nb)
            df = logratio.transform_df(data=df,
                                       base=self.mf_base,
                                       columns=self.mf_outputs,
                                       new_columns=self._mf_layers_nb)
        return df

    def predict(self, x: List[List], samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """

        Args:
            x:
            samples:

        Returns:

        """
        print(f"Running prediction (n={len(x)})")
        prediction = self.keras_model.predict(self.preprocessor.transform(np.array(x)))
        mixture = self.construct_mixture(prediction)
        print(f"Sampling from mixture ({len(x)}x{samples} samples)")
        df = self.sample_from_mixture(mixture, samples)
        df_components = self.get_mixture_components(mixture)
        input_prompt = pd.DataFrame(x, columns=self.inputs)
        input_prompt.rename_axis("prediction", inplace=True)
        return df, df_components, input_prompt

    def predict_with_error(self, x: List, errors: List, samples: Tuple[int, int] = (1000, 100), extra_query="") \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """

        Args:
            x:
            errors:
            samples:

        Returns:

        """
        uncertainty_samples = pd.DataFrame(self.rng.multivariate_normal(mean=x, cov=np.diag(np.power(errors, 2)),
                                                                        size=samples[0]),
                                           columns=self.inputs)
        uncertainty_samples.rename_axis("prediction", inplace=True)
        initial_length = len(uncertainty_samples)

        if "planet_radius" in self.inputs and "planet_mass" in self.inputs:
            min_radius = self.min_planet_radius(uncertainty_samples["planet_mass"])
            uncertainty_samples = uncertainty_samples[uncertainty_samples["planet_radius"] > min_radius]

        for value in self.inputs:
            if value in self.input_properties:
                props = self.input_properties[value]
                min_value = props.get("min_value", uncertainty_samples[value].min())
                max_value = props.get("max_value", uncertainty_samples[value].max())
                uncertainty_samples = uncertainty_samples.query(f"{value}.between({min_value}, {max_value})")
        
        if extra_query:
            uncertainty_samples = uncertainty_samples.query(extra_query)

        dropped_samples = initial_length - len(uncertainty_samples)
        if dropped_samples > 0:
            print(f"{dropped_samples} points are outside the parameter limits and will not be used in the prediction. "
                  f"Current number of samples: {len(uncertainty_samples) * samples[1]}")
        print(f"Running prediction (n={len(uncertainty_samples)}):")
        prediction = self.keras_model.predict(self.preprocessor.transform(uncertainty_samples.to_numpy()))
        mixture = self.construct_mixture(prediction)
        print(f"Sampling from mixture ({len(uncertainty_samples)}x{samples[1]} samples):")
        df = self.sample_from_mixture(mixture, samples[1])
        df_components = self.get_mixture_components(mixture)
        return df, df_components, uncertainty_samples

    @staticmethod
    def min_planet_radius(mass):
        # Power law fit to 100% iron
        # Returns radius in Re
        coef = [-0.22381715,  0.27825493, -0.00753378]
        p = Polynomial(coef)
        return np.exp(p(np.log(mass)))

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import ipywidgets as widgets

if TYPE_CHECKING:
    from exomdn import ExoMDN

from exomdn.mdn_model import Model


class LoadModelWidget(widgets.VBox):
    def __init__(self, parent: "ExoMDN", models_path: Path):
        super().__init__()
        self.parent = parent
        self.models_path = models_path

        label = widgets.Label("Select model:")
        self.select_model = widgets.Select(layout=widgets.Layout(width="50%"))
        self.update_model_list()
        self.load_model_button = widgets.Button(description="Load model")
        self.update_list_button = widgets.Button(description="Update list")
        self.output = widgets.Output(layout=widgets.Layout(width="100%"))

        self.children = [label,
                         self.select_model,
                         widgets.HBox([self.load_model_button, self.update_list_button]),
                         self.output]

        self.update_list_button.on_click(self.on_update_list_button_clicked)
        self.load_model_button.on_click(self.on_load_model_button_clicked)

    def update_model_list(self):
        available_models = [f.name for f in os.scandir(self.models_path) if (f.is_dir() and not f.name.startswith("_"))]
        self.select_model.options = sorted(available_models)

    def on_load_model_button_clicked(self, b):
        self.output.clear_output()
        if self.select_model.value is None:
            with self.output:
                print("No model selected!")
            return
        with self.output:
            self.parent.model = Model(self.models_path / self.select_model.value)
            print(f"MDN inputs: {self.parent.model.inputs}")
            print(f"MDN outputs: {self.parent.model.outputs}")

    def on_update_list_button_clicked(self, b):
        self.output.clear_output(wait=True)
        self.update_model_list()
        with self.output:
            print(f"Reloaded model list. Found {len(self.select_model.options)} models.")


class PredictionWidget(widgets.VBox):
    def __init__(self, parent: "ExoMDN"):
        super().__init__()
        self.parent = parent

        try:
            with open(self.parent.data_path / "exoplanets.json", "r") as f:
                self.exoplanet_data = json.load(f)
        except FileNotFoundError:
            self.exoplanet_data = {}

        # Tab 'planet parameters'
        self.use_error = widgets.Checkbox(value=False, description="Include uncertainties?", indent=True)

        self.parameter_inputs = {}

        for parameter in self.parent.model.inputs:
            config = self.get_parameter_info(parameter)
            input_widget = ParameterWithErrorInput(label=config["label"] + " = ",
                                                   si_unit=config["unit"],
                                                   tooltip=config["description"],
                                                   value=config["default_value"],
                                                   min_value=config["min_value"],
                                                   max_value=config["max_value"],
                                                   step=config["step"])
            input_widget.error.disabled = True
            self.parameter_inputs[parameter] = input_widget

        tab1 = widgets.VBox([self.use_error] + list(self.parameter_inputs.values()))

        # tab 'options'
        style = {"description_width": "150px"}
        self.samples_input = widgets.BoundedIntText(value=10000, min=1, max=1e6, description="Samples",
                                                    tooltip="Total number of samples per prediction",
                                                    style=style,
                                                    layout=widgets.Layout(width="30%"))
        self.subsamples_input = widgets.BoundedIntText(value=1000, min=1, max=1e6, description="Uncertainty samples",
                                                       tooltip="How often to sample from within the uncertainties?",
                                                       disabled=True,
                                                       style=style,
                                                       layout=widgets.Layout(width="30%"))

        tab2 = widgets.VBox([self.samples_input, self.subsamples_input])

        # load planet data tab
        self.planet_selection = widgets.Select(layout=widgets.Layout(width="30%", height="200px"))
        self.planet_selection.options = self.exoplanet_data.keys()
        self.planet_info = widgets.Output(layout=widgets.Layout(width="60%", height="200px"))
        self.load_planet_data_button = widgets.Button(description="Load selected planet",
                                                      layout=widgets.Layout(width="30%"))
        if len(self.exoplanet_data) == 0:
            with self.planet_info:
                print(f"No file 'exoplanets.json' found in datapath {self.parent.data_path}")
        tab3 = widgets.VBox([widgets.HBox([self.planet_selection, self.planet_info]), self.load_planet_data_button])

        self.tab = widgets.Accordion()
        self.tab.children = [tab3, tab1, tab2]
        self.tab.titles = ["Load exoplanet data", "Planet parameters", "Options"]
        self.tab.selected_index = 1

        self.predict_button = widgets.Button(description="Predict interior", layout=widgets.Layout(width="30%"))

        # Output
        self.output = widgets.Output(layout=widgets.Layout(width="100%"))

        self.children = [self.tab, self.predict_button, self.output]

        # setup callbacks
        self.use_error.observe(self.use_error_checkbox_change, names="value")
        self.planet_selection.observe(self.on_select_planet_data_change, names="value")
        self.predict_button.on_click(self.on_predict_button_clicked)
        self.load_planet_data_button.on_click(self.on_load_planet_button_clicked)

    def use_error_checkbox_change(self, change):
        for widget in self.parameter_inputs.values():
            widget.error.disabled = not self.use_error.value
        self.subsamples_input.disabled = not self.use_error.value

    def get_parameter_info(self, parameter):
        config = self.parent.model.input_properties
        if parameter in config:
            return config[parameter]
        config = {
            "label": parameter,
            "unit": "",
            "description": "",
            "min_value": None,
            "max_value": None,
            "default_value": 1,
            "step": 1
            }
        return config

    def on_predict_button_clicked(self, b):
        use_uncertainty = self.use_error.value
        values = [inp.parameter.value for inp in self.parameter_inputs.values()]
        samples = self.samples_input.value
        errors = None
        subsamples = None
        if use_uncertainty:
            errors = [inp.error.value for inp in self.parameter_inputs.values()]
            subsamples = self.subsamples_input.value
            samples = round(samples / subsamples)

        with self.output:
            self.output.clear_output(wait=True)
            if use_uncertainty:
                self.parent.prediction, self.parent.mixture_components, self.parent.input_prompt = \
                    self.parent.model.predict_with_error(x=values, errors=errors, samples=(subsamples, samples))
            else:
                self.parent.prediction, self.parent.mixture_components, self.parent.input_prompt = \
                    self.parent.model.predict(x=[values], samples=samples)
                self.parent.uncertainty_samples = None
            print("== Done! ==")

    def on_load_planet_button_clicked(self, b):
        with self.planet_info:
            selected = self.exoplanet_data[self.planet_selection.value]
            for key in self.parent.model.inputs:
                try:
                    value = selected[key]
                    value_err = selected[f"{key}_err"]
                except KeyError:
                    value = self.parent.model.input_properties[key]["default_value"]
                    value_err = 0
                self.parameter_inputs[key].parameter.value = value
                self.parameter_inputs[key].error.value = value_err
            print("--- Updated planet parameters ---")

    def on_select_planet_data_change(self, change):
        self.planet_info.clear_output()
        no_data = []
        with self.planet_info:
            planet = change["new"]
            selected = self.exoplanet_data[planet]
            print(f"== {planet} ==")
            for key in self.parent.model.inputs:
                try:
                    value = selected[key]
                    value_err = selected.get(f"{key}_err", 0)
                    print(f"{key} = {value} ± {value_err}")
                except KeyError:
                    no_data.append(key)
                    continue
            print(f"Reference: {selected['ref_name']}")
            print(f"Reference (T_eq): {selected['eqt_refname']}")
            if len(no_data) > 0:
                print(f"No values found for {'; '.join(no_data)}. Using default values.")


class ParameterWithErrorInput(widgets.HBox):
    def __init__(self, label: str, si_unit: str = "", value: float = None, min_value: float = 0, max_value: float = 1,
                 step: float = 1, tooltip: str = ""):
        super().__init__()
        value = min_value if value is None else value
        if min_value is None or max_value is None:
            self.parameter = widgets.FloatText(value=value,
                                               step=step,
                                               description=label,
                                               tooltip=tooltip,
                                               layout=widgets.Layout(flex="8 1 0%", width="auto"))
        else:
            self.parameter = widgets.BoundedFloatText(value=value,
                                                      min=min_value,
                                                      max=max_value,
                                                      step=step,
                                                      description=label,
                                                      tooltip=tooltip,
                                                      layout=widgets.Layout(flex="8 1 0%", width="auto"))
        si_label = widgets.Label(si_unit,
                                 layout=widgets.Layout(flex="1 1 0%", width="auto"))
        self.error = widgets.BoundedFloatText(value=0,
                                              min=0,
                                              max=max_value,
                                              step=step,
                                              description="$\pm$",
                                              tooltip="Uncertainty",
                                              layout=widgets.Layout(flex="8 1 0%", width="auto"))
        self.children = [self.parameter, si_label, self.error]
        self.layout = widgets.Layout(width="75%")

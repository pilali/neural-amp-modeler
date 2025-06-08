## Extending the NAM Training Framework

The Neural Amp Modeler (NAM) training framework is designed to be extensible, allowing users to integrate custom neural network architectures and dataset loaders beyond the built-in options. This enables experimentation with novel model designs and sophisticated data handling, such as for parametric modeling.

### Registering Custom Network Architectures

To use a custom neural network architecture with the `nam_full` training script, you need to register an initializer for it.

*   **Function:** `nam.train.lightning_module.LightningModule.register_net_initializer(name: str, constructor: Callable)`
*   **Parameters:**
    *   `name` (string): A unique string identifier for your custom architecture. This name will be used in the `model_config.json` file under the `net.name` key to select your custom network.
    *   `constructor` (Callable): A callable function or method that instantiates your custom network. This constructor will receive the `net.config` object (the dictionary associated with the `config` key under `net` in `model_config.json`) as its argument. It should return an instance of your custom network.
        *   Typically, this is a class method like `YourCustomNet.init_from_config(cls, net_config)` defined on your custom network class.
*   **Custom Network Interface:**
    Your custom network class should ideally inherit from `nam.models.base.BaseNet`. If not directly inheriting, it should implement a similar interface to ensure compatibility with the training framework and export process. Key methods and attributes include:
    *   `receptive_field` (property or attribute): An integer indicating the total number of input samples the network needs to produce one output sample.
    *   `export_config(self)` (method): Returns a dictionary containing any configuration needed for the exported model (e.g., for the NAM plugin).
    *   `export_weights(self)` (method): Returns a NumPy array of the model's weights in the format expected by the NAM plugin.
    *   `forward(self, x, ...)`: The standard PyTorch forward pass.

Example registration:
```python
from nam.train.lightning_module import LightningModule
from nam.models.base import BaseNet

class MyCustomNet(BaseNet):
    def __init__(self, input_channels, hidden_features, receptive_field_val):
        super().__init__()
        self._receptive_field = receptive_field_val
        # ... define your layers ...

    @property
    def receptive_field(self):
        return self._receptive_field

    @classmethod
    def init_from_config(cls, config_dict):
        # config_dict is model_config["net"]["config"]
        return cls(
            input_channels=config_dict.get("input_channels", 1),
            hidden_features=config_dict["hidden_features"],
            receptive_field_val=config_dict["receptive_field"]
        )

    # ... other required methods ...

# Register before initializing LightningModule
LightningModule.register_net_initializer("MyCustomNet", MyCustomNet.init_from_config)
```
Then, in your `model_config.json`:
```json
{
  "net": {
    "name": "MyCustomNet",
    "config": {
      "hidden_features": 128,
      "receptive_field": 2049
      // other custom params
    }
  }
  // ... other model configs ...
}
```

### Registering Custom Dataset Loaders

Similarly, you can register custom dataset loaders to handle specific data formats or preprocessing needs.

*   **Function:** `nam.data.register_dataset_initializer(name: str, constructor: Callable)`
*   **Parameters:**
    *   `name` (string): A unique string identifier for your custom dataset. This name will be used in the `data_config.json` file, typically under a `type` key within the `train` or `validation` dataset configurations.
    *   `constructor` (Callable): A callable function or method that instantiates your custom dataset. This constructor will receive the specific dataset configuration object from `data_config.json` (e.g., the dictionary for one of the datasets in the `train` list if using `ConcatDataset`, or the `train` object itself). It should return an instance of your custom dataset.
        *   Typically, this is a class method like `YourCustomDataset.init_from_config(cls, dataset_config)` defined on your custom dataset class.
*   **Custom Dataset Interface:**
    Your custom dataset class should ideally inherit from `torch.utils.data.Dataset` and, for consistency, could follow the pattern of `nam.data.AbstractDataset` or `nam.data.Dataset`. At a minimum, it must implement:
    *   `__init__(self, ...)`: To initialize with parameters from the config.
    *   `__len__(self)`: Returns the total number of items in the dataset.
    *   `__getitem__(self, idx)`: Returns a single data sample (e.g., a tuple of input tensor `x` and target tensor `y`).

Example registration:
```python
from torch.utils.data import Dataset
from nam.data import register_dataset_initializer

class MyCustomDataset(Dataset):
    def __init__(self, audio_path, params_path, segment_length):
        super().__init__()
        # ... load audio and params ...
        self.audio_path = audio_path
        self.params_path = params_path
        self.segment_length = segment_length
        # ... more init logic ...

    @classmethod
    def init_from_config(cls, config_dict):
        # config_dict is the specific dataset entry from data_config.json
        return cls(
            audio_path=config_dict["audio_path"],
            params_path=config_dict["params_path"],
            segment_length=config_dict.get("segment_length", 4096)
        )

    def __len__(self):
        # ... return length ...
        return 0 # Placeholder

    def __getitem__(self, idx):
        # ... load segment and corresponding params ...
        # return x_audio_segment, (y_audio_segment, params_for_segment)
        # or however your custom net expects it.
        return None, None # Placeholder


# Register before init_dataset is called
register_dataset_initializer("MyCustomDataset", MyCustomDataset.init_from_config)
```
Then, in your `data_config.json` (e.g., for a single training dataset):
```json
{
  "train": {
    "type": "MyCustomDataset", // Selects your custom dataset
    "audio_path": "path/to/audio.wav",
    "params_path": "path/to/params.csv",
    "segment_length": 8192
    // other custom params
  },
  "common": {
    // ...
  }
}
```

### Use Case: Advanced Parametric Modeling

Imagine you want to model an amplifier where the model explicitly takes 'gain' and 'tone' knob settings as inputs, in addition to the audio.

1.  **`CustomParameterizedNet`:**
    *   Design a neural network (e.g., a modified WaveNet or LSTM) that has dedicated input paths or embedding layers for the 'gain' and 'tone' parameters. These parameters could be concatenated to the audio input at some stage, or used to modulate layer behaviors.
    *   The `forward` method might look like `forward(self, audio_input, gain_param, tone_param)`.
    *   Implement `YourNet.init_from_config` to parse any specific architecture details from `model_config.json`.

2.  **`CustomParameterizedDataset`:**
    *   This dataset would load the primary audio (`x_path`, `y_path`) and also a corresponding file (e.g., CSV) containing time-aligned 'gain' and 'tone' values.
    *   Its `__getitem__` method would return a tuple like `(audio_input_segment, (audio_target_segment, gain_value, tone_value))`. The batch collation would then need to handle this structure. The `LightningModule`'s `training_step` would also need to be adapted to unpack these parameters and pass them to the network.

3.  **Registration:**
    *   Call `LightningModule.register_net_initializer("ParametricNAM", CustomParameterizedNet.init_from_config)`.
    *   Call `nam.data.register_dataset_initializer("ParametricDataset", CustomParameterizedDataset.init_from_config)`.

4.  **Configuration:**
    *   `model_config.json`:
        ```json
        {
          "net": {
            "name": "ParametricNAM",
            "config": { /* parameters for CustomParameterizedNet */ }
          }
          // ...
        }
        ```
    *   `data_config.json`:
        ```json
        {
          "train": {
            "type": "ParametricDataset",
            "x_path": "...", "y_path": "...",
            "knob_data_path": "path/to/knobs.csv"
            // other custom params
          },
          // ...
        }
        ```

### How to Use Extensibility Features

*   **Timing of Registration:** The registration functions (`register_net_initializer`, `register_dataset_initializer`) must be called early in your training script's execution. This needs to happen *before* the `LightningModule` is instantiated (for custom nets) or `init_dataset` is called (for custom datasets), which in practice means before `Trainer.fit()` is invoked by the main `nam/train/full.py` script.

*   **`~/.neural-amp-modeler/extensions` Directory:**
    A convenient way to manage and automatically load your custom components is to place your Python files containing the custom class definitions and their registration calls into the `~/.neural-amp-modeler/extensions` directory (create it if it doesn't exist). The NAM framework attempts to load Python files from this directory at startup. This allows your extensions to be available without modifying the core NAM scripts directly.
    *   Ensure your custom component files in this directory are self-contained or manage their dependencies appropriately.
    *   The registration calls should be at the global level within these files to ensure they execute when the file is imported.

By leveraging these extensibility points, users can significantly tailor the NAM training framework to their specific research and modeling needs.

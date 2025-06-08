# Neural Amp Modeler (`nam`) Full Training Documentation

## Introduction

This document provides comprehensive documentation for the `nam_full` training script and associated components within the Neural Amp Modeler (NAM) ecosystem. Its purpose is to detail:
- The command-line arguments for the `nam_full` script.
- The structure and parameters of the JSON configuration files used for defining data, learning processes, and model architectures.
- The end-to-end model training flow, from data input to model export.
- The specific mechanisms for model conditioning, particularly for WaveNet and LSTM architectures.

This guide is intended for users and developers looking to understand and customize the training of Neural Amp Models using the provided tools.

## Command-Line Arguments (`nam_full`)

The `nam_full` script (found in `nam/cli.py`) is the primary entry point for training models. It requires paths to configuration files and an output directory.

### Positional Arguments

The `nam_full` script requires the following positional arguments in order:

*   `data_config_path`: Path to the JSON configuration file for the dataset. This file specifies details like the dataset source, features to use, and any preprocessing steps.
*   `model_config_path`: Path to the JSON configuration file for the NAM model. This file defines the architecture of the model, including the number of layers, activation functions, and other hyperparameters.
*   `learning_config_path`: Path to the JSON configuration file for the training process. This file controls aspects like the learning rate, batch size, number of epochs, and optimization algorithm.
*   `outdir`: Path to the directory where the output files (e.g., trained model, evaluation results, plots) will be saved.

### Optional Arguments

The `nam_full` script also accepts the following optional arguments:

*   `--no-show` or `-ns`: If specified, matplotlib plots will not be displayed interactively. This is useful when running the script in an environment without a display server.
*   `--no-plots` or `-np`: If specified, the generation of plots will be skipped entirely. This can save time and resources if plots are not needed.

## JSON Configurations

The training process is heavily guided by three main JSON configuration files: Data, Learning, and Model.

### Data Configuration (JSON Format)

The data configuration JSON file defines the datasets used for training and validation, along with common processing parameters.

The general structure of the JSON includes up to three top-level keys: `train`, `validation`, and `common`.

```json
{
  "train": {
    // Training dataset configuration
  },
  "validation": {
    // Validation dataset configuration
  },
  "common": {
    // Common parameters applicable to both train and validation
  },
  "_notes": [
    "This is a conventional key for adding comments or notes.",
    "It is ignored by the parser."
  ]
}
```

#### `train` and `validation` Sections

These sections define the specific data sources for training and validation. Each can be either a single dataset object or a list of dataset objects (for `ConcatDataset`).

A single dataset object has the following parameters:

*   `x_path` (string): Path to the input audio file (e.g., unprocessed guitar signal).
*   `y_path` (string): Path to the target audio file (e.g., reamped guitar signal or output from the device being modeled).
*   `ny` (integer, optional): Specifies the number of output samples per training or validation item.
    *   **Effect on Efficiency and Diversity**: Using `ny` breaks down the audio files into smaller chunks. This can improve training efficiency by allowing smaller batches and increases diversity within each batch, as each item is a different segment of the audio.
    *   **Validation Behavior**: If `ny` is `null` or not provided in the `validation` section, the entire validation file is used as a single item. This is typical for evaluating the model on a continuous, longer piece of audio.
*   `start_seconds` (float, optional): The time in seconds from the beginning of the audio file at which to start processing.
*   `stop_seconds` (float, optional): The time in seconds from the beginning of the audio file at which to stop processing. Negative values are interpreted as counting from the end of the file (e.g., -10.0 means stop 10 seconds before the end).
*   `start_samples` (integer, optional): The sample index from the beginning of the audio file at which to start processing. This offers more precise control than `start_seconds`.
*   `stop_samples` (integer, optional): The sample index from the beginning of the audio file at which to stop processing. Negative values are interpreted as counting from the end of the file.

    **Note**: The `start_seconds` and `stop_seconds` parameters are considered deprecated. It is recommended to use `start_samples` and `stop_samples` for more precise, sample-accurate control over audio segments.

#### `common` Section

This section defines parameters that are applied globally to both the training and validation datasets unless overridden within a specific `train` or `validation` dataset configuration.

*   `delay` (integer, optional): Specifies a latency in samples to apply between the input (`x`) and target (`y`) signals.
    *   A **positive delay** shifts `y` later relative to `x`. This means `y` will effectively start `delay` samples after `x` starts.
    *   A **negative delay** shifts `y` earlier relative to `x`. This can be used to compensate for inherent latencies in the recording or processing chain of the target audio.
*   `nx` (integer): Defines the number of input samples, also known as the receptive field of the model.
    *   This value is critical as it determines the length of the input audio segment fed to the model at each step.
    *   Typically, `nx` is determined by the model's architecture (e.g., the sum of kernel sizes and dilations in a WaveNet-style model).
    *   The `nam/train/full.py` script usually calculates and adds/overrides this value in the configuration based on the model being trained. If `nx` is present in the JSON, it might be overridden by the training script to match the model's actual receptive field.
*   `allow_unequal_lengths` (boolean, optional, default: `false`): If set to `true`, the script will allow `x_path` and `y_path` to refer to audio files of different lengths. The processing will be truncated to the length of the shorter of the two files after considering `start_samples`, `stop_samples`, and `delay`.
*   `sample_rate` (integer, optional): The expected sample rate of the audio files in Hertz (e.g., 44100, 48000).
    *   If not provided, the sample rate is inferred from the header of the audio file specified in `x_path`.
    *   If provided, the script will verify that all loaded audio files match this sample rate. An error will be raised if there's a mismatch.
*   `y_preroll` (integer, optional): Number of initial samples to drop from the beginning of the `y_path` audio data. This can be useful to remove unwanted artifacts or silence at the start of the target audio before alignment with `x`.
*   `input_gain` (float, optional, default: `0.0`): Specifies a gain value in decibels (dB) to be applied to the input signal `x`. This allows for adjusting the level of the input audio before it's fed to the model.
*   `require_input_pre_silence` (float, optional, default: `0.4`): Specifies a duration in seconds of silence that is required at the beginning of the input signal `x` (after any `start_samples` offset and before the main content).
    *   **Importance**: This is crucial for models that have a significant receptive field (`nx`). If the model is fed non-silent audio that immediately precedes the intended training data, the initial part of the target audio `y` might be "leaked" into the model's prediction due to the model "seeing" future input samples that correspond to past target samples. Enforcing pre-silence ensures that the model's initial state is not influenced by audio that should effectively be in its "past" relative to the target `y` signal, preventing contamination of the training signal. The training script typically trims this pre-silence from `y` to ensure correct alignment.

#### `ConcatDataset` Configuration

If the value associated with the `train` or `validation` key is a **list of dataset configuration objects** (where each object follows the structure described for a single dataset), then a `ConcatDataset` will be created.

This means that data will be drawn sequentially from each dataset defined in the list. This is useful for combining multiple recording sessions, different instruments, or various conditions into a single logical dataset for training or validation.

Example of `ConcatDataset` for `train`:
```json
{
  "train": [
    {
      "x_path": "path/to/input1.wav",
      "y_path": "path/to/target1.wav",
      "ny": 4096
    },
    {
      "x_path": "path/to/input2.wav",
      "y_path": "path/to/target2.wav",
      "ny": 4096,
      "start_samples": 44100, // Start 1 second into the second file
      "input_gain": -3.0
    }
  ],
  "common": {
    "sample_rate": 44100,
    "delay": 0
  }
}
```

#### `_notes` Key

It's a common convention to include a `_notes` key at the top level of the JSON file. The value is typically an array of strings. This key is ignored by the configuration parser and serves purely as a place for users to leave comments, reminders, or descriptions about the dataset configuration.

### Learning Configuration (JSON Format)

The learning configuration JSON file controls aspects of the training process, including how data is fed to the model and how the PyTorch Lightning `Trainer` is configured.

```json
{
  "train_dataloader": {
    // Parameters for the training DataLoader
  },
  "val_dataloader": {
    // Parameters for the validation DataLoader
  },
  "trainer": {
    // Parameters for the PyTorch Lightning Trainer
  },
  "trainer_fit_kwargs": {
    // Optional additional arguments for trainer.fit()
  },
  "_notes": [
    "Configuration for training loops and trainer."
  ]
}
```

#### `train_dataloader` Parameters

This object configures the `DataLoader` for the training dataset.

*   `batch_size` (integer): The number of training items (chunks of audio, as defined by `ny` in data config) to include in each batch.
*   `shuffle` (boolean): If `true`, the training data will be shuffled at the beginning of each epoch. This is important for ensuring the model doesn't learn the order of the data.
*   `pin_memory` (boolean): If `true`, the DataLoader will copy tensors into pinned memory before returning them. This can speed up data transfer from CPU to GPU.
*   `drop_last` (boolean): If `true`, the last batch will be dropped if it is incomplete (i.e., has fewer items than `batch_size`). This can be useful for ensuring consistent batch sizes, especially for distributed training.
*   `num_workers` (integer): The number of subprocesses to use for data loading. A higher number can speed up data loading by parallelizing it, but also increases memory usage. `0` means data will be loaded in the main process.

#### `val_dataloader` Parameters

This object configures the `DataLoader` for the validation dataset.

*   It typically remains empty (`{}`) in the default configuration, meaning default `DataLoader` settings will be used.
*   However, it can accept the same parameters as `train_dataloader`.
*   `shuffle` is usually set to `false` (or omitted, defaulting to false) for validation, as shuffling is not necessary and can make it harder to compare metrics across epochs if specific validation samples are of interest.
*   If `ny` is not set for the validation data (meaning the whole file is used), `batch_size` should typically be 1.

#### `trainer` Parameters

This object contains parameters that are passed directly to the constructor of the PyTorch Lightning `Trainer` instance.

*   `accelerator` (string): Specifies the hardware accelerator to use for training. Common values are:
    *   `"cpu"`: Use CPU for training.
    *   `"gpu"`: Use GPU for training.
    *   `"mps"`: Use Apple Metal Performance Shaders (for M1/M2 Macs).
*   `devices` (integer or list): Specifies the devices to use.
    *   If integer: The number of devices (e.g., `1` for one GPU, `2` for two GPUs).
    *   If list: A list of specific device IDs (e.g., `[0, 1]` to use GPU 0 and GPU 1).
*   `max_epochs` (integer): The maximum number of full passes over the training dataset.
*   `val_check_interval` (float or integer, optional): Controls how often to run the validation loop within a training epoch.
    *   If float (e.g., `0.25`): Validation will run after this fraction of the training batches in an epoch.
    *   If integer (e.g., `100`): Validation will run every `N` training batches.
    *   This is useful for getting more frequent feedback on validation performance, especially for long epochs.
*   `check_val_every_n_epoch` (integer, optional, default: `1`): Validation will run every `N` epochs. This is used if `val_check_interval` is not specified or is set to a value that doesn't trigger validation within an epoch.
*   **Other `Trainer` Arguments**: Many other arguments accepted by the PyTorch Lightning `Trainer` can be included here (e.g., `logger`, `callbacks`, `precision`, `gradient_clip_val`). Refer to the PyTorch Lightning documentation for a full list.

#### `trainer_fit_kwargs` (dictionary, optional)

This object allows specifying additional keyword arguments that are passed directly to the `trainer.fit()` method. This is less common but can be used for advanced use cases or for arguments that are specific to the `.fit()` call rather than the `Trainer` initialization.

Example:
```json
{
  "trainer_fit_kwargs": {
    "ckpt_path": "path/to/some/checkpoint.ckpt" // Example: resume training from a specific checkpoint
  }
}
```

### Model Configuration (JSON Format)

The model configuration JSON file defines the neural network architecture, optimizer, learning rate scheduler, and loss function parameters.

```json
{
  "optimizer": {
    // Optimizer settings
  },
  "lr_scheduler": {
    // Learning rate scheduler settings
  },
  "loss": {
    // Loss function settings (optional)
  },
  "net": {
    "name": "WaveNet", // Or "LSTM", etc.
    "config": {
      // Network-specific architecture parameters
    }
  },
  "_notes": [
    "Configuration for the model architecture and optimization parameters."
  ]
}
```

#### Common Model Parameters

These parameters are typically defined at the root of the model configuration JSON.

*   **`optimizer`** (object): Configures the optimizer used for training.
    *   `lr` (float): The learning rate.
    *   Other parameters (e.g., `weight_decay`, `betas` for Adam) can be specified here if the `configure_optimizers()` method in `nam.train.lightning_module.LightningModule` is adapted to pass them to the chosen PyTorch optimizer (e.g., `torch.optim.Adam`).
*   **`lr_scheduler`** (object): Configures the learning rate scheduler.
    *   `class` (string): The name of a PyTorch learning rate scheduler class from `torch.optim.lr_scheduler` (e.g., `"ExponentialLR"`, `"ReduceLROnPlateau"`, `"StepLR"`).
    *   `kwargs` (object): A dictionary of keyword arguments to be passed to the scheduler's constructor.
        *   Example for `"ExponentialLR"`: `{"gamma": 0.99}`
        *   Example for `"ReduceLROnPlateau"`: `{"mode": "min", "factor": 0.1, "patience": 10}`
*   **`loss`** (object, optional): Parameters related to the loss function calculation.
    *   `val_loss` (string, optional, e.g., `"esr"`, `"mse"`): Specifies the primary metric to use for validation loss tracking and for schedulers like `ReduceLROnPlateau`. Defaults to a standard MSE if not specified. "esr" typically refers to Error-to-Signal Ratio.
    *   `mask_first` (integer, optional): Number of initial samples in the model's output to exclude from the loss calculation. This is particularly useful for recurrent models like LSTMs where the hidden state needs a "burn-in" period, or for causal convolutional models where the initial output samples might be affected by padding.
    *   `pre_emph_mrstft_weight` (float, optional): If using a loss function that includes a Multi-Resolution Short-Time Fourier Transform (MRSTFT) component, this sets the weight for the MRSTFT loss calculated on pre-emphasized audio. Pre-emphasis can help the loss function focus more on higher frequencies.
    *   `pre_emph_mrstft_coef` (float, optional): The coefficient used for the first-order filter in pre-emphasis (e.g., 0.85, 0.97). Typically `y[n] = x[n] - coef * x[n-1]`.
    *   Other loss-related parameters might be available depending on the specific loss functions implemented or chosen within the `LightningModule`.

#### Network-Specific Configuration (`net`)

The `net` object contains the configuration specific to the neural network architecture.

*   `name` (string): Specifies the type of model architecture to use. Currently supported values include:
    *   `"WaveNet"`: For WaveNet-style convolutional neural networks.
    *   `"LSTM"`: For Long Short-Term Memory recurrent neural networks.
*   `config` (object): Contains the detailed architectural parameters for the chosen network `name`.

##### For `name: "WaveNet"`

The `net.config` object for a WaveNet model typically includes:

*   `layers_configs` (list of objects): An array, where each object defines a "stack" or "block" of WaveNet dilated convolutional layers. Multiple stacks can be used to build deeper models or to process signals at different resolutions.
    *   `input_size` (integer): Number of input channels to this stack. For the first stack, this is usually 1 (for mono audio). For subsequent stacks, it depends on the output of the previous stack or how inputs are concatenated/summed.
    *   `condition_size` (integer): Number of channels in the conditioning signal, if used. Conditioning allows external signals (e.g., embeddings, other features) to influence the WaveNet layers. Often 0 if no conditioning is used for a particular stack.
    *   `head_size` (integer): Number of output channels from this stack that are routed to the overall skip connection sum. These are typically processed by a 1x1 convolution within the stack to produce this output.
    *   `channels` (integer): The number of internal channels used within the dilated convolutional layers of this stack. This is also often the number of channels output by the main convolutional path before splitting for gating or skip connections.
    *   `kernel_size` (integer): The size of the 1D convolution kernels in the dilated layers (e.g., 2 or 3).
    *   `dilations` (list of integers): A list specifying the dilation factor for each layer within this stack (e.g., `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]`). The length of this list determines the number of layers in this stack.
    *   `activation` (string): The name of the activation function to use within the convolutional blocks (e.g., `"Tanh"`, `"ReLU"`, `"PReLU"`).
    *   `gated` (boolean): If `true`, uses gated activations (e.g., `tanh(conv_filter) * sigmoid(conv_gate)`), which is common in WaveNet architectures.
    *   `head_bias` (boolean): If `true`, the final 1x1 convolution within each layer stack that produces the `head_size` output (skip connection contribution) will include a bias term.
*   `head_config` (object, optional): Defines a "head" network that processes the sum of skip connections from all `layers_configs`. This typically consists of one or more 1x1 convolutional layers to map the summed skip signals to the final output dimension.
    *   `in_channels` (integer): Number of input channels to the head. This should generally match the `head_size` of the `layers_configs` (if all stacks have the same `head_size` and are summed) or the sum of `head_size`s if they differ.
    *   `channels` (integer): Number of channels in the hidden layers of the head.
    *   `activation` (string): Activation function used in the head layers (e.g., `"ReLU"`).
    *   `num_layers` (integer): Number of layers in the head.
    *   `out_channels` (integer): Final number of output channels for the model (typically 1 for audio modeling).
*   `head_scale` (float): A scalar value used to scale the sum of all skip connections before it is processed by the `head_config` (or, if no `head_config`, before being output). This can help control the overall magnitude of the signal path.

##### For `name: "LSTM"`

The `net.config` object for an LSTM model includes:

*   `hidden_size` (integer): The number of features in the hidden state `h` of the LSTM. This also determines the number of output features if no projection layer is added.
*   `num_layers` (integer, optional): The number of recurrent layers. For example, setting `num_layers=2` would stack two LSTMs on top of each other, with the first LSTM's output providing the input to the second. Defaults to PyTorch's LSTM default (usually 1) if not specified.
*   `train_burn_in` (integer, optional, default: `0`): The number of initial time steps in each training sequence for which the hidden state will be calculated but the output will not be used for backpropagation. This allows the LSTM's hidden state to "warm up" or "burn in" on the sequence before trying to make predictions.
*   `train_truncate` (integer, optional, default: `null`): If specified, sequences longer than this value will be truncated during training to this length for Truncated Backpropagation Through Time (TBPTT). If `null` or `0`, the full length of the training sequences (as determined by `ny` in data config) is processed.
*   `input_size` (integer, optional, default: `1`): The number of expected features in the input `x` to the LSTM. For single-channel (mono) audio, this is 1.
*   Other standard PyTorch `torch.nn.LSTM` constructor arguments (e.g., `bias`, `batch_first`, `dropout`, `bidirectional`, `proj_size`) can potentially be passed if the `LSTM` model wrapper in `nam/models/recurrent.py` is extended to handle them in its `lstm_kwargs`. Currently, `batch_first` is hardcoded to `False` in the wrapper.

---
*The "Common Model Parameters" and "Network-Specific Configuration" subsections, including details for WaveNet and LSTM, are based on analysis of the provided configuration files and Python scripts. Specific behaviors and available parameters may evolve with the codebase.*

## Model Training Flow

This section describes the typical sequence of operations when training a model using the `nam_full` script.

1.  **Entry Point (`nam/cli.py`)**:
    *   The process begins when the user executes the `nam_full` script.
    *   `argparse` is used to parse command-line arguments. These include:
        *   Paths to the three main JSON configuration files: data (`data_config_path`), model (`model_config_path`), and learning (`learning_config_path`).
        *   The output directory (`outdir`) where results and trained models will be saved.
        *   Boolean flags like `--no-show` and `--no-plots` to control matplotlib plot generation and display.
    *   These arguments are passed to the `run` function in `nam/train/full.py`.

2.  **Configuration Loading (`nam/train/full.py`)**:
    *   The `run` function in `nam/train/full.py` takes the parsed file paths and loads the JSON configuration files into Python dictionaries: `data_config`, `model_config`, and `learning_config`.
    *   These dictionaries provide all the necessary parameters for setting up the model, data, and training loop.

3.  **Model Initialization (`nam/train/full.py`, `nam/train/lightning_module.py`, `nam/models/*`)**:
    *   The core PyTorch Lightning model, `LightningModule` (defined in `nam/train/lightning_module.py`), is initialized.
    *   It receives the `model_config` which contains parameters for the optimizer, learning rate scheduler, loss function, and the network architecture itself (`model_config["net"]`).
    *   Inside `LightningModule.__init__`, the specific neural network (e.g., `WaveNet` from `nam/models/wavenet.py` or `LSTM` from `nam/models/recurrent.py`) is instantiated based on `model_config["net"]["name"]` and its corresponding `model_config["net"]["config"]`.
    *   A crucial step here is that the model's receptive field (`model.net.receptive_field`) is calculated after the network is built. This value is then used to set `data_config["common"]["nx"]`. This ensures that the data provided to the model during training and inference is of the correct length for the model to make a prediction for each output sample without seeing future information.

4.  **Data Preparation (`nam/data.py`, `nam/train/full.py`)**:
    *   The `init_dataset` function from `nam/data.py` is called separately for the training and validation phases.
    *   It uses the (potentially updated) `data_config` to create dataset objects.
        *   If `data_config["train"]` (or `data_config["validation"]`) is a single object, a `Dataset` instance is created, which loads and processes a single pair of input (`x_path`) and target (`y_path`) audio files according to parameters like `ny`, `start_samples`, `stop_samples`, `delay`, etc.
        *   If `data_config["train"]` (or `data_config["validation"]`) is a list of such objects, a `ConcatDataset` is created, which concatenates multiple `Dataset` instances.
    *   Back in `nam/train/full.py`, PyTorch `DataLoader`s (`train_dataloader`, `val_dataloader`) are created using these datasets. The `learning_config["train_dataloader"]` and `learning_config["val_dataloader"]` provide parameters like `batch_size`, `shuffle`, `num_workers`, etc.

5.  **Training Process (`nam/train/full.py`, `nam/train/lightning_module.py`)**:
    *   A PyTorch Lightning `Trainer` instance is configured and initialized. Parameters for the `Trainer` (e.g., `max_epochs`, `accelerator`, `devices`, `val_check_interval`) are taken from `learning_config["trainer"]`.
    *   Standard callbacks are typically added, most importantly `ModelCheckpoint` (configured to save the best model based on validation loss, e.g., `val_loss_esr` or `val_loss_mse`) and potentially others like `EarlyStopping` or `LearningRateMonitor`.
    *   The main training loop is started by calling `trainer.fit()`. This method takes:
        *   The `LightningModule` instance (`model`).
        *   The `train_dataloader`.
        *   The `val_dataloader`.
        *   Additional arguments from `learning_config["trainer_fit_kwargs"]` can also be passed.
    *   Inside the `LightningModule`:
        *   `training_step(batch, batch_idx)`: Receives a batch of data, passes the input `x` through `model.net` to get predictions, calculates the loss (e.g., using MSE, potentially with MRSTFT components as configured in `model_config["loss"]`), and logs training metrics. The loss is returned to PyTorch Lightning for backpropagation and optimizer steps.
        *   `validation_step(batch, batch_idx)`: Similar to `training_step`, but for the validation data. It computes and logs validation metrics (e.g., ESR, MSE). These metrics are used by `ModelCheckpoint` and learning rate schedulers like `ReduceLROnPlateau`.
        *   `configure_optimizers()`: This method, called by the `Trainer`, sets up the optimizer (e.g., Adam) and learning rate scheduler (e.g., ExponentialLR) based on `model_config["optimizer"]` and `model_config["lr_scheduler"]`.

6.  **Post-Training (`nam/train/full.py`)**:
    *   After `trainer.fit()` completes, the script loads the best model checkpoint that was saved by `ModelCheckpoint` during training using `LightningModule.load_from_checkpoint()`.
    *   If plotting is enabled (i.e., `--no-plots` was not specified), the script generates and potentially shows various diagnostic plots (e.g., comparison of target vs. predicted signals on validation data, loss curves).
    *   Finally, the trained network (`model.net`) is exported for deployment (e.g., as a `.nam` file for the plugin or a JIT-compiled TorchScript module) using its `export(outdir)` method. The specific export format and content depend on the model type (WaveNet, LSTM, etc.).

---
*This documentation provides a high-level overview of the training flow. Specific details can be found within the respective Python scripts and are subject to change as the codebase evolves.*

## Conditioned Model Training Mechanism

This section details how conditioning is implemented or can be potentially implemented for WaveNet and LSTM models within this framework.

### WaveNet Conditioning

The WaveNet models implemented in `nam/models/wavenet.py` utilize a form of **self-conditioning**. This means the primary input audio itself is used as the conditioning signal that influences the behavior of the WaveNet layers.

*   **Source of Conditioning:** The input audio tensor, loaded from `x_path` as defined in the data configuration, serves a dual purpose. It is the main signal processed by the dilated convolutions, and it is also the conditioning signal. No separate conditioning audio file is loaded by default for this architecture.

*   **Role of `condition_size`:** In the `wavenet.json` model configuration, within `net.config.layers_configs`, the `condition_size` parameter for each layer stack dictates the number of channels expected for the conditioning input to that stack.
    *   If the main input audio `x` is mono and used directly as the conditioner, `condition_size` would typically be set to `1`.
    *   If `condition_size` is `0`, no conditioning is applied in that specific layer stack.

*   **Signal Path:**
    1.  The input audio tensor `x` (from `nam.data.Dataset`) is passed to the `WaveNet.forward()` method (which internally calls `_WaveNet._forward()`).
    2.  In `_WaveNet._forward(self, x, c=None, ...)`, if `c` is not provided (which is typical for self-conditioning as the conditioning signal `c` is derived from `x` internally), `x` itself is assigned to `c`.
    3.  This conditioning tensor `c` is then passed to the `_WaveNet._Layers.forward(self, x, c, ...)` method for each stack of layers.
    4.  Within each `_Layers` block, `c` is passed as the `h` (conditioning) argument to every `_Layer.forward(self, x, h)` call.
    5.  Inside `_Layer.forward`, this conditioning signal `h` is processed by `self._input_mixer` (a `Conv1d` layer acting as a 1x1 convolution). The output of this mixer is then added to the output of the main dilated convolution path (after its activation and before the residual connection).

    This mechanism allows each layer in the WaveNet to be influenced by the overall characteristics of the input audio segment, facilitating the learning of complex, input-dependent behaviors.

### LSTM Conditioning

The standard LSTM implementation in `nam/models/recurrent.py` (configured via `models/lstm.json`) does **not** inherently employ an explicit external conditioning signal in the same way the WaveNet's self-conditioning is structured, nor does it automatically use the input audio for a separate conditioning path by default.

*   **Standard Input:** Typically, the LSTM expects a single feature per time step if `input_size` in `net.config` is `1`. For audio, this is the mono audio sample at each time step. The input tensor is usually shaped `[batch_size, sequence_length, input_size]`.

*   **Potential for Conditioning ("catnet" approach):**
    The `__init__` method of `nam.models.recurrent.LSTM` includes a comment: *"TODO: catnet means we'll change input_size"*. This hints at a potential mechanism for incorporating conditioning with LSTMs, often referred to as a "CatNet" style where features are concatenated:
    1.  **Data Preparation:** The user would need to prepare their conditioning signal(s) separately. This might involve loading another audio file, extracting features, or generating a control signal.
    2.  **Feature Concatenation:** This conditioning signal would then be concatenated with the primary audio input along the feature dimension. For example, if the primary audio is 1 feature and the conditioning signal has 2 features, the combined input would have 3 features per time step. This concatenation would typically be handled in a custom `Dataset` or data preprocessing script before the data is fed to the `DataLoader`.
    3.  **Adjust `input_size`:** The `input_size` parameter in the LSTM's `net.config` (e.g., in `lstm.json`) must be updated to reflect the new total number of features. For the example above, `input_size` would be set to `3`.

*   **User Responsibility:** It's important to emphasize that this concatenation-based conditioning for LSTMs is **not an automatic feature** of the current default setup. The framework provides the flexibility to set `input_size`, but the actual data preparation, feature engineering for the conditioning signal, and concatenation logic must be implemented by the user. The current `nam.data.Dataset` does not automatically handle loading or concatenating auxiliary conditioning inputs for LSTMs.

---
*This documentation clarifies the conditioning mechanisms based on the current codebase. Future developments might introduce more explicit or automated conditioning features.*

## Advanced: Extensibility for Custom Models and Data

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

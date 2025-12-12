<div align="center">

# GenTract

<strong>GenTract: Generative Global Tractography</strong><br>

<img src="docs/assets/gentract.gif" width=400/>

</div>

----

This repository provides a framework for training GenTract. The models include conditional diffusion models, variational autoencoders, and flow-matching models. The framework is designed to be modular, allowing for easy customization and experimentation.


## Repository Structure

- **`scripts/`**: Contains training scripts for different models.
- **`configs/`**: YAML configuration files for defining model and training parameters.
- **`src/Tract/`**: Core library with data loaders, model definitions, trainers, and utilities.
- **`data/`**: Directory for storing datasets and metadata.

## Prerequisites

1. **Python Version**: Ensure you have Python 3.8 or higher installed.
2. **Dependencies**: Install the required Python libraries using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset**: Prepare your dataset and ensure it is referenced in the configuration file.
4. **MAISI VAE**: The pre-trained MAISI VAE weights must be accessible, to finetune for the SH Coefficient VAEs.

## Training the Conditional Diffusion Model

The `train_cond_diff.py` script is used to train the conditional diffusion model. Follow these steps:

### 1. **Prepare the Configuration File**
   - Update the configuration file (e.g., `configs/diffusion_config.yaml`) with the appropriate paths and parameters:
     - Dataset path (`data.csv_file`)
     - Model parameters (`model_dim`, `diff_transformer_layers`, etc.)
     - Training parameters (`batch_size`, `n_epochs`, etc.)
     - Output paths (`base_dir`)

### 2. **Run the Training Script**
   Use the following command to train the model:
   ```bash
   python scripts/train_cond_diff.py --config configs/diffusion_config.yaml --temp_dir /path/to/temp/dir
   ```
   - `--config`: Path to the YAML configuration file.
   - `--temp_dir`: Temporary directory for caching data during training.

### 3. **Monitor Training**
   - Training progress is logged to the console.
   - Weights & Biases (wandb) is used for experiment tracking. Ensure you are logged into your wandb account:
     ```bash
     wandb login
     ```

### 4. **Results**
   - The trained model and results are saved in the directory specified in the configuration file (`paths.base_dir`).
   - The configuration file is copied to the results directory for reproducibility.

## Configuration File Example

Below is an example of a YAML configuration file for the conditional diffusion model:

```yaml
model:
  M: 128
  diff_transformer_layers: 6
  model_dim: 256
  cond: true

data:
  csv_file: /path/to/dataset.csv
  cache: true
  num_samples: 1000

training:
  batch_size: 32
  n_epochs: 100
  lr: 0.001
  timesteps: 1000

diffusion:
  schedule: linear
  beta_start: 0.0001
  beta_end: 0.02

paths:
  base_dir: /path/to/output

wandb:
  project: tract_diffusion
```

## Acknowledgments

This framework leverages PyTorch, MONAI, and other open-source libraries for generative modeling and medical imaging.
## Overview

Here, we provide tools to generate interpretable visualizations for class neurons in deep neural networks. 

---

## Core Script

- `class_neurons_fvis.py`: Main entry point for visualizing class neurons using VITAL. Now supports automatic configuration via `get_config()` to simplify usage.

Please fill the [IMAGENET_DIR](utils/opt_utils.py) variable before using the code!

---

## Configuration System

The version supports a structured configuration through `get_config(args)` defined in `config.py`. Instead of passing all hyperparameters manually, only high-level inputs like architecture, class index, and number of real images are needed.

### ✅ Supported Architectures:
- `resnet50`
- `densenet121`
- `convnext_base`
- `vit_l_16`
- `vit_l_32`

### Config File Logic
For each known architecture, the following parameters are set automatically:
- `layer_weights`: Contribution weights for each layer used in the loss
- `tv_l2`, `l2`: Regularization coefficients specific to each model
- `folder_name`: Automatically constructed as `class_neurons/<arch_name>/VITAL/rand<num_real_img>/cls<target>/t<run_id>`
- Optimization defaults:
  - `lr`: Learning rate (default: 1.0)
  - `feat_dist`: Feature distribution alignment weight (default: 1.0)
  - `epochs`: Number of iterations (default: 2000)
  - `resolution`: Image resolution (default: 224)
  - `jitter`, `bs`, `do_flip`, `tv_l1`, etc.

> If the provided architecture is not listed, you can manually set these parameters by generating a new config dictionary.

---

### 🔧 Usage (Simplified)
```bash
python class_neurons_fvis.py \
    --arch_name resnet50 \
    --target 1 \
    --run_id 1 \
    --num_real_img 50 \
    --gpuid 0
```

> Remaining arguments like `layer_weights`, `l2`, `tv_l2`, and folder paths will be auto-filled using the configuration.

---

## 📂 Bash Script

Use `run_VITAL.sh` for launching reproducible experiments from SLURM. Update the places in the script according to your environment, this is just an example script.

```bash
sbatch run_VITAL.sh
```
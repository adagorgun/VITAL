# Reverse-Engineering Neurons via Patch Activation

This repository provides a pipeline to identify and visualize the most relevant image regions (patches) that strongly activate specific neurons in deep neural networks (e.g., ResNet). The goal is to extract meaningful insights about internal representations of neurons by localizing top-activating patches in a memory- and time-efficient manner.

---

## 📌 Overview

We provide two main approaches to extract the most relevant patches from neurons:

1. **Baseline** using `reverse_engineer_neurons.py`: Computes patches directly across all ImageNet data. Accurate but costly.
2. **Efficient 2-Step Process** using `imagenet2txt.py` and `txt2patch.py`: First identify top-N activating images, then find top-k patches from them (where N >> k).

---

## 📦 Modules

### 🔹 1. `imagenet2txt.py`

Selects the top-N activating **images** per channel using max-pooled activations.

#### ✅ Features
- Uses forward hooks to access intermediate feature maps.
- Computes max activation per channel across spatial dimensions.
- Records the file paths for top-N images.

#### 🛠️ Usage

```bash
python imagenet2txt.py \
    --target_layer layer4 \
    --arch resnet50 \
    --batch_size 512 \
    --topk_images 10000
```

#### ⚙️ Arguments

| Argument           | Description                                              | Default     |
|--------------------|----------------------------------------------------------|-------------|
| `--target_layer`   | Target layer from which to extract neuron activations    | `layer4`    |
| `--arch`           | Model architecture to use (from torchvision)             | `resnet50`  |
| `--batch_size`     | Batch size for data loading                              | `512`       |
| `--topk_images`    | Number of top images to retain per neuron/channel        | `10000`     |
| `--split_N`        | Optional subset size from ImageNet                      | `None`      |

#### 📄 Output
- Saves a `files_all.txt` file per channel with top-N image paths.
- Output is stored under: `resnet50/neuron_<layer>/<channel>/files_all.txt`

---

### 🔹 2. `txt2patch.py`

Given the image list from `files_all.txt`, this module identifies the **top-k patches** per neuron/channel.

#### 🛠️ Usage
```bash
python txt2patch.py \
    --target_layer layer4 \
    --arch resnet50 \
    --patch_size 32 \
    --topk_patches 50 \
    --channel 162 \
    --save_dir path/to/save/
```

#### ⚙️ Arguments
| Argument           | Description                                           | Default                     |
|--------------------|-------------------------------------------------------|-----------------------------|
| `--target_layer`   | The layer where activations are extracted from       | `layer4`                    |
| `--arch`           | Torchvision model to use                             | `resnet50`                  |
| `--patch_size`     | Size of square patches to extract                    | `32`                        |
| `--topk_patches`   | Number of top patches to save                        | `50`                        |
| `--save_dir`       | Directory where results are saved                    | `resnet50/neuron_layer4`    |
| `--channel`        | Channel index to extract patches from                | `162`                       |

#### 📄 Output
- `patches.png`: grid of top-k patches per channel.
- `topk_files.txt`: image filenames where top patches originated.
- Saved under: `resnet50/neuron_layer4/<channel>/`

---

## 📄 Visualization Directory Structure

```bash
resnet50/
├── neuron_layer4/
│   ├── 162/
│       ├── patches.png
│       ├── files_all.txt
│       └── topk_files.txt
```

---

### 🔹 3. `reverse_engineer_neurons.py`

Extracts top-k patches for multiple channels from **all** ImageNet data.

#### 🛠️ Usage
```bash
python reverse_engineer_neurons.py \
    --target_layer layer4_2 \
    --arch resnet50 \
    --patch_size 64 \
    --chs 54,1935
```

#### ⚙️ Arguments
| Argument           | Description                                      | Default     |
|--------------------|--------------------------------------------------|-------------|
| `--target_layer`   | Truncation point in the model                   | `layer4_2`  |
| `--arch`           | Pretrained model architecture                  | `resnet50`  |
| `--batch_size`     | Number of samples per batch                    | `256`       |
| `--patch_size`     | Patch size to extract from image               | `64`        |
| `--chs`            | Channel indices to analyze (comma-separated)   | `54,1935`   |
| `--split_N`        | Load subset of ImageNet (if desired)           | `None`      |

#### 📄 Output
- `patches.png` and `files.txt` saved under: `resnet50/neuron_<layer>/<channel>/top<k>/`

---

### 🔹 4. `inner_neurons_fvis.py`

Generates feature visualizations from top-k patches using the VITAL optimization framework.

#### 🛠️ Usage
```bash
python inner_neurons_fvis.py \
    --arch_name resnet50 \
    --layer layer4_2 \
    --channel 1935 \
    --topk_dir resnet50/neuron_layer4/ \
    --epochs 2000 \
    --method LRP \
    --exp_name neuron_1935_layer4_2 \
    --folder_name images \
    --layer_weights 0.1,1,1,1,0
```

#### ⚙️ Arguments
| Argument           | Description                                       | Default         |
|--------------------|---------------------------------------------------|-----------------|
| `--arch_name`      | Model to visualize from                          | `resnet50`      |
| `--layer`          | Target layer                                     | `layer4_2`      |
| `--channel`        | Neuron channel ID                                | `1935`          |
| `--topk_dir`       | Directory containing top-k patches               | `resnet50/neuron_layer4/` |
| `--epochs`         | Number of optimization steps                     | `2000`          |
| `--method`         | Attribution method: LRP / LRPRestricted / GuidedBackprop         | `LRP`           |
| `--num_real_img`   | Number of reference images used                  | `50`            |
| `--folder_name`    | Folder where results are saved                   | `images`        |
| `--exp_name`       | Experiment prefix for saved image name          | `None`          |
| `--layer_weights` | Weights for scaling each block for the distribution loss |  `0.1,1,1,1,0`

#### 📄 Output
- Saves optimized neuron visualizations in `<folder_name>/`

---

## 📅 Notes

- For best performance, ensure access to full ImageNet dataset.
- You may need to adjust `ImageNet` dataset paths inside `utils/imagenet_dataset.py`.
- Outputs are neatly stored in `resnet50/neuron_<layer>/<channel>/` depending on the method.



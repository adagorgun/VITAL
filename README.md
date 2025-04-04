# <span style="color:#006c66;"><b>VITAL</b></span>: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow
*[*Ada Görgün*](https://adagorgun.github.io/), [*Bernt Schiele*](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/), [*Jonas Fischer*](http://explainablemachines.com/members/jonas-fischer.html)*  
 

🚧 **Full code is coming soon!** Stay tuned.  
🔗 Project Page: [adagorgun.github.io/VITAL-Project](https://adagorgun.github.io/VITAL-Project/)  
📄 Paper: [arXiv 2503.22399](https://arxiv.org/abs/2503.22399)

---

## 📌 Overview

<div style="text-align: justify"

<span style="color:#006c66;"><b>VITAL</b></span> is a framework for improving feature visualization by aligning distributional properties of synthesized features and promoting relevant information flow through the network. The method is designed to yield more interpretable visualizations.

---

## 🧠 Abstract

<div style="text-align: justify"

Neural networks are widely adopted to solve complex and challenging tasks. Especially in high-stakes decision-making, understanding their reasoning process is crucial, yet proves challenging for modern deep networks. Feature visualization (FV) is a powerful tool to decode what information neurons are responding to and hence to better understand the reasoning behind such networks. In particular, in FV we generate human-understandable images that reflect the information detected by neurons of interest. However, current methods often yield unrecognizable visualizations, exhibiting repetitive patterns and visual artifacts that are hard to understand for a human. To address these problems, we propose to guide FV through statistics of real image features combined with measures of relevant network flow to generate prototypical images. Our approach yields human-understandable visualizations that both qualitatively and quantitatively improve over state-of-the-art FVs across various architectures. As such, it can be used to decode which information the network uses, complementing mechanistic circuits that identify where it is encoded.

---

## 🚀 Getting Started

### Requirements

Code was tested in virtual environment with Python 3.11. Install requirements:

- torch
- torchvision
- numpy
- Pillow
- matplotlib
- tqdm
---

## Class Neuron Visualization

Class neuron visualization aims to reveal what a neural network "sees" when it thinks about a specific class (e.g., "dog" or "airplane"). This is done by generating an image that maximally activates the output neuron corresponding to that class. The resulting visualization gives insight into the features the model associates with that category—such as shapes, textures, or patterns. 

**VITAL** enhances this by aligning these visualizations with real-world feature distributions, resulting in clearer and more realistic class representations. This is achieved by matching the generated image's feature distribution to that of real images from the same class through the sort matching algorithm. The result is a more interpretable and meaningful visualization that can help us understand how the model perceives different classes.

👉 You can explore the full implementation in the [./class_fvis/](./class_fvis/) directory.

## Intermediate Neuron Visualization

Intermediate neuron visualization focuses on understanding how information is represented deep inside the network, rather than just at the classification layer. These internal neurons often respond to abstract concepts like "fur texture" or "wheel shapes," even if they're not directly tied to a class. By visualizing what activates these hidden neurons, we can uncover emergent concepts and compositional features the model builds up to make decisions. 

**VITAL** improves this process by filtering neurons based on their relevance and by guiding the visualizations with real feature statistics—leading to more meaningful and interpretable representations. Instead of just maximizing neuron activation like in traditional methods, VITAL traces how much relevant information flows from the neuron toward the model’s final decision for that class and aligns the feature distribution of generated images with the feature distribution of real images that activates the target neuron the most.

**Code is coming soon!**


## Concept Visualization

**Coming soon!**

## Metrics

**Coming soon!**

## 📬 Contact

For questions, feel free to contact:

**Ada Görgün**  
📧 [agoerguen@mpi-inf.mpg.de](mailto:agoerguen@mpi-inf.mpg.de)  
🔗 [adagorgun.github.io](https://adagorgun.github.io/)

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{gorgun2025vitalunderstandablefeaturevisualization,
  title={VITAL: More Understandable Feature Visualization through Distribution Alignment and Relevant Information Flow}, 
  author={Ada Gorgun and Bernt Schiele and Jonas Fischer},
  year={2025},
  eprint={2503.22399},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.22399}, 
}
```
---

_This repository will be updated shortly. Thank you for your interest!_

# AI Art Generation Toolkit

A PyTorch implementation of creative AI models for artistic image generation and style translation.

## Overview

This repository contains implementations of three groundbreaking deep learning approaches for artistic image generation:

- **Neural Style Transfer**: Blend content and style from different images
- **Pix2Pix**: Paired image-to-image translation using conditional GANs
- **CycleGAN**: Unpaired image translation with cycle consistency

## What's Inside

### Neural Style Transfer

Combine the content of one image with the artistic style of another using a pre-trained CNN. The magic happens through three key components:

**Content Preservation**
- Matches high-level content features between source and output
- Uses intermediate CNN layer activations to capture object structure

**Style Transfer**
- Gram matrix computation captures texture and style patterns
- Multiple layers capture different brushstroke scales
- Shallower layers = finer details, deeper layers = broader patterns

**Visual Smoothing**
- Total variation loss reduces noise and artifacts
- Encourages spatial coherence in generated images

###  Pix2Pix - Paired Image Translation

For when you have matched input-output pairs, Pix2Pix uses conditional GANs to learn the mapping:

**Generator Architecture**
- U-Net with skip connections to preserve spatial information
- Encoder-decoder with bottleneck features
- Skip connections shuttle low-level details directly

**Discriminator**
- PatchGAN classifier operating on image patches
- Focuses on high-frequency structure
- More efficient and effective than full-image discrimination

**Training Strategy**
- Combines adversarial loss with L1 pixel-wise loss
- Instance normalization instead of batch normalization
- Adam optimizer with carefully tuned parameters

###  CycleGAN - Unpaired Translation

When you don't have paired training data, CycleGAN learns bidirectional translations:

**Key Innovation**
- Two generators: X→Y and Y→X
- Two discriminators for each domain
- Cycle consistency ensures meaningful translations

**Cycle Consistency**
- Forward cycle: x → G(x) → F(G(x)) ≈ x
- Backward cycle: y → F(y) → G(F(y)) ≈ y
- Prevents mode collapse and ensures semantic preservation

**Architecture Details**
- Generator: ResNet blocks with instance normalization
- Discriminator: 70×70 PatchGAN
- Identity loss for color preservation in some domains

## Quick Start

```python
# Style Transfer
python run_style_transfer.py --content image.jpg --style painting.jpg

# Pix2Pix Training
python train_pix2pix.py --dataset facades --direction BtoA

# CycleGAN Training  
python train_cyclegan.py --dataset horse2zebra
```

## Technical Highlights

**PyTorch Lightning Integration**
- Reproducible experiments
- Mixed precision training support
- Hardware scalability (CPU/GPU/TPU)
- Clean separation of research and engineering code

**Optimization Advances**
- Least squares GAN loss for stable training
- Historical generated images buffer for reduced oscillations
- Multiple normalization strategies (instance/batch/layer)
- Adaptive learning rate scheduling

## Results Gallery

The models excel at various creative tasks:

- **Style Transfer**: Photo → Van Gogh, Picasso, etc.
- **Pix2Pix**: Architectural labels → facades, maps → satellite
- **CycleGAN**: Horses ↔ zebras, photos → Monet paintings

## Requirements

```
torch>=1.7
pytorch-lightning>=1.0
torchvision
pillow
numpy
```

## License

MIT License - feel free to use these models for your creative projects!

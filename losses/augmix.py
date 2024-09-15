import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# ImageNet code should change this value
IMAGE_SIZE = 32


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
    translate_x, translate_y, color, contrast, brightness, sharpness
]

# Convert tensor to PIL Image
def tensor_to_pil(img_tensor):
    # Check if there is a batch dimension (shape: [B, C, H, W])
    if len(img_tensor.shape) == 4:
        # Process each image in the batch individually
        pil_images = []
        for img in img_tensor:
            pil_images.append(tensor_to_pil(img))  # Recursively handle each image in the batch
        return pil_images
    elif isinstance(img_tensor, torch.Tensor):
        # Handle single image tensor
        img_tensor = img_tensor.mul(255).byte()
        img_tensor = img_tensor.permute(1, 2, 0)
        return Image.fromarray(img_tensor.cpu().numpy())
    return img_tensor  # Return as-is if it's not a tensor

# Convert PIL Image to tensor
def pil_to_tensor(pil_img):
    return transforms.ToTensor()(pil_img)
  
def aug(image, preprocess, mixture_width=np.random.randint(1, 5), mixture_depth=np.random.randint(1, 4), aug_severity=np.random.randint(1, 5)):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  if len(image.shape) == 4:  # Check if the input has a batch dimension [B, C, H, W]
    mixed_batch = []
    for img in image:
        mixed_batch.append(single_image_aug(img, preprocess, mixture_width, mixture_depth, aug_severity))
    return torch.stack(mixed_batch)  # Return a batch of mixed images
  else:
    return single_image_aug(image, preprocess, mixture_width, mixture_depth, aug_severity)

def augmix_loss(model, x_natural, y):
    
    preprocess = transforms.Compose(
      [transforms.Normalize([0.5] * 3, [0.5] * 3)])

    im_tuple = (preprocess(x_natural), aug(x_natural, preprocess), aug(x_natural, preprocess))
    images_all = torch.cat(im_tuple, 0).cuda()
    targets = y.cuda()
    logits_all = model(images_all)
    logits_clean, logits_aug1, logits_aug2 = torch.split(
        logits_all, im_tuple[0].size(0))
    # Cross-entropy is only computed on clean images
    loss = F.cross_entropy(logits_clean, targets)
    p_clean, p_aug1, p_aug2 = F.softmax(
        logits_clean, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)
    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                  F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                  F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
    return loss

def single_image_aug(image, preprocess, mixture_width, mixture_depth, aug_severity):
    device = image.device
    aug_list = augmentations_all
    ws = torch.tensor(np.float32(np.random.dirichlet([1] * mixture_width)), device=device)
    m = torch.tensor(np.float32(np.random.beta(1, 1)), device=device)

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.clone() if isinstance(image, torch.Tensor) else image.copy()
        
        image_aug = tensor_to_pil(image_aug) if isinstance(image_aug, torch.Tensor) else image_aug

        depth = mixture_depth if mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)

        image_aug = pil_to_tensor(image_aug) if isinstance(image_aug, Image.Image) else image_aug
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image).to(device) + m * mix
    return mixed

Author: Patrik Dominik Pördi
Email: ppordi@umd.edu

Importing Libraries

```python
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import pytorch3d
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import mcubes
import pickle

Instructions
    cd Final
    python rendering.py
The results will be generate in the results folder
Documentation:
    Final/docs/report.html


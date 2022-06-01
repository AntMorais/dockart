from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms as T

"""
GENERAL CONFIGURATIONS
"""

ROOT_DIR = Path(__file__).parent.parent.parent.parent

"""
DATA CONFIGURATIONS
"""
DATABASE_URI = 'postgresql://postgres:art_class@localhost:5432'



"""
TORCH CONFIGURATIONS
"""

# TODO should put all this in a yaml file that changes with command line Torch configs
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO This variable should depend on the number of classes of the current model
NUM_CLASSES = 27
MODEL_PATH = '../models/art_style/artclass.pt'

transform = nn.Sequential(
            T.Resize([256, ]),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )


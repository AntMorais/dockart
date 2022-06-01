import torch
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms as T
from PIL import Image, ImageFile
from pydantic import BaseModel
from typing import List
from enum import Enum
import data_functions


transform = nn.Sequential(
            T.Resize([256, ]),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

# TODO change this when moving to AWSs
device = torch.device('cpu')
img_path = 'ml/test_data/pop_art.jpg'
image = Image.open(img_path)
image = T.functional.to_tensor(image)
image = transform(image).to(device)
image = image.unsqueeze(0)

style_model_path = 'ml/models/art_style/artclass.pt'
num_classes = 27
style_model = models.resnet18(pretrained=False, progress=True)
num_ftrs = style_model.fc.in_features
style_model.fc = nn.Linear(num_ftrs, num_classes)
style_model.load_state_dict(
                torch.load(
                    style_model_path,
                    map_location=device
                )
            )

def load_models():
	"""
	load the models from disk
	and put them in a dictionary
	Returns:
	dict: loaded models
	"""
	models = {
		"style": style_model,
	}
	print("models loaded from disk")
	return models

models = load_models()

def get_predictions(model, inputs: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        return preds

  
class ModelType(str, Enum):
	style = "style"
	genre = "genre"	
	artist = "artist"

class UserRequestIn(BaseModel):
	name: str
	path: str
	model_type: ModelType = "style"
  

class EntityOut(BaseModel):
	name: str
	description: str
	label: str

class EntitiesOut(BaseModel):
	entities: List[EntityOut]


print(get_predictions(models['style'], image))

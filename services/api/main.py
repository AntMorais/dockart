from copy import deepcopy
from fastapi import FastAPI, File, UploadFile, Form
import os
import io
import torch
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms as T
from PIL import Image, ImageFile
from api.utils import datafunctions
from api.utils.configs import *
from api.utils.classes import *



data_path = os.path.join(ROOT_DIR, "data")
metadata_path = os.path.join(data_path, 'metadata')
info_path = os.path.join(data_path, 'info')

artist_class = datafunctions.read_class_txt(os.path.join(metadata_path, 'artist_class.txt'))

genre_class = datafunctions.read_class_txt(os.path.join(metadata_path, 'genre_class.txt'))

style_class = datafunctions.read_class_txt(os.path.join(metadata_path, 'style_class.txt'))
# change name because of data error
style_class[3]='Art_Nouveau_Modern'

style_urls = datafunctions.read_class_txt(os.path.join(info_path, 'urls_style.txt'))

device = torch.device(DEVICE)

style_model = models.resnet18(
    pretrained=False,
    progress=True
    )

num_ftrs = style_model.fc.in_features

style_model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

style_model.load_state_dict(
                torch.load(
                    MODEL_PATH,
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

def get_info(pred, type):
    """Gets information about a class.

    Args:
        pred (int): Predicted label of image
        type (str): Type of class, 'style', 'genre' or 'artist'

    Returns:
        dict: Dictionary with info about a class
    """
    dict_keys = [
        'name',
        'description',
        'wiki_url'
    ]
    info = dict.fromkeys(dict_keys, None)
    # TODO change this so that we get the data from the PostgreSQL database that is in another Docker container
    # TODO Orchestrate everything with k8s
    info['name'] = style_class[pred]
    info['description'] = "Read more"
    info['wiki_url'] = style_urls[pred]
    return info


app = FastAPI()


@app.get("/")
def hello():
    return {"message": "you're not supposed to be here :)"}



#, status_code=201
@app.post("/uploadfile/", response_model=EntitiesOut)
async def create_upload_file(name: str = Form(...),model_type: str = Form(...),file: UploadFile = File(...)):
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    image = T.functional.to_tensor(image)
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    preds = get_predictions(models['style'], image)
    # TODO change type here
    info = get_info(preds, 'style')

    return {'entities': [EntityOut(
        name=info['name'],
        description=info['description'],
        wiki_url=info['wiki_url']
    )]}
import io
import os
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision import transforms, models
import json
import time


# load labels extracted from annotations to find at https://github.com/visipedia/inat_comp/tree/master/2021
with open('categories_inat2021.json') as f:
  categories = json.load(f)


model = None
use_gpu = False

# TODO: adapt path to images folder
image_path = '../images/iNat2021/general/'

def load_model():
    global model

    # TODO: Download pre-trained models from https://github.com/EibSReM/newt/tree/main/benchmark
    # TODO: adapt path to respective model
    model_weights_fp = 'cvpr21_newt_pretrained_models.tar\cvpr21_newt_pretrained_models\pt\inat2021_supervised_large_from_scratch.pth.tar'
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10000)
    checkpoint = torch.load(model_weights_fp, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    model.eval()

    # if use_gpu:
    #     model.cuda()


def prepare_image(image, target_size):
    print(type(image))
    
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image nad preprocess it.
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    # Convert to Torch.Tensor and normalize.
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)


def predict(image_path):

    data = {"success": False}

    image = open(image_path, 'rb').read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image, target_size=(224, 224))

    preds = F.softmax(model(image), dim=1)
    # adapt number of results k as needed
    results = torch.topk(preds.cpu().data, k=6, dim=1)
    results = (results[0].cpu().numpy(), results[1].cpu().numpy())
    data['predictions'] = list()

    for prob, label in zip(results[0][0], results[1][0]):
        label_name = categories['categories'][label]['name']
        r = {"label": label_name, "probability": float(prob)}
        data['predictions'].append(r)
        print(r)


    # Loop over the predictions and display them.
    #print(image_path,end ="\t")
    output_string=''
    output_string=output_string+image_path +'\t'
    
    for (i, result) in enumerate(data['predictions']):
        output_string=output_string+'{}'.format(result['label'])+'\t' +'{:.4f}'.format(result['probability'])+'\t'

    return output_string

    
        
if __name__ == '__main__':
    start = time.time()
    load_model()
    
    # TODO: adjust image path
    mypath=image_path
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    
    output_all=''
    for x in onlyfiles:
        print('this file is processed')
        print(x)
        string = predict(mypath+x)
        #string=''
        output_all=output_all+string+ '\n'
    
    print(output_all)

    text_file = open("Output.txt", "w")
    text_file.write(output_all)
    text_file.close()

    end = time.time()
    total_time = end - start
    print("total runtime: " + str(total_time))

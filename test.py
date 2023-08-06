"""
@Project: Farmer_identify
@File   : test.py
@Author : Ruiqing Tang
@Date   : 2023/8/6 9:37
"""
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import warnings
import cv2
warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
idx_to_labels = np.load('E:\\dataset\\farmerdata\\idx_to_labels.npy', allow_pickle=True).item()
model = torch.load('models/best-0.900.pth')
model = model.eval().to(device)
test_transform = transforms.Compose([transforms.Resize(224),
                                     # transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
images_folder_path = 'E:\\dataset\\farmerdata\\test'
images_path = os.listdir(images_folder_path)
n = 1
columns = ['name', 'label']
df_result = pd.DataFrame(columns=columns)
# test_result = {}
'''
for image_path in tqdm(images_path):
    # test_result['name'] = image_path.strip('.jpg')
    # img_name = image_path.strip('.jpg')
    img_name = image_path
    image_path = os.path.join(images_folder_path, image_path)
    img_pil = Image.open(image_path)
    input_img = test_transform(img_pil)
    input_img = input_img.unsqueeze(0).to(device)
    pred_logits = model(input_img)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = torch.topk(pred_softmax, n)
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()
    pred_ids = pred_ids.tolist()
    # test_result['label'] = pred_ids
    df_result = df_result.append({'name': img_name, 'label': pred_ids}, ignore_index=True)
'''
for image_path in tqdm(images_path):
    # test_result['name'] = image_path.strip('.jpg')
    # img_name = image_path.strip('.jpg')
    img_name = image_path
    image_path = os.path.join(images_folder_path, image_path)
    img_pil = cv2.imread(image_path)
    # img_pil = cv2.cvtColor(img_pil,code=cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img_pil,cv2.COLOR_BGR2RGB))
    # img_pil = cv2.resize(img_pil,(256,256))
    # img_pil = img_pil[16:240,16:240]
    input_img = test_transform(img_pil)
    input_img = input_img.unsqueeze(0).to(device)
    pred_logits = model(input_img)
    pred_softmax = F.softmax(pred_logits, dim=1)
    top_n = torch.topk(pred_softmax, n)
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()
    pred_ids = pred_ids.tolist()
    # test_result['label'] = pred_ids
    df_result = df_result.append({'name': img_name, 'label': pred_ids}, ignore_index=True)

df_result.to_csv('submit.csv', index=False)

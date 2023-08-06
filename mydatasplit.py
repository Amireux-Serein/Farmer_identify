"""
@Project: Farmer_identify
@File   : mydatasplit.py
@Author : Ruiqing Tang
@Date   : 2023/8/5 16:07
"""
import os
import shutil
import random
import pandas as pd

dataset_path = "E:\\dataset\\farmerdata\\mydataset"
dataset_name = "mydataset"
classes = os.listdir(dataset_path)
os.mkdir(os.path.join(dataset_path, "mytrain"))
os.mkdir(os.path.join(dataset_path, "myval"))
for farmer_class in classes:
    os.mkdir(os.path.join(dataset_path, "mytrain", farmer_class))
    os.mkdir(os.path.join(dataset_path, "myval", farmer_class))

val_train_ratio = 0.2
random.seed(100)
df = pd.DataFrame()
print('{:^18} {:^18} {:^18}'.format('calsses', 'trainset_num', 'valset_num'))
for farmer_class in classes:
    orginal_dir = os.path.join(dataset_path, farmer_class)
    images_filename = os.listdir(orginal_dir)
    random.shuffle(images_filename)
    valset_num = int(len(images_filename) * val_train_ratio)
    valset_images = images_filename[:valset_num]
    trainset_images = images_filename[valset_num:]

    for image in valset_images:
        orginal_img_path = os.path.join(dataset_path, farmer_class, image)
        target_val_path = os.path.join(dataset_path, "myval", farmer_class, image)
        shutil.move(orginal_img_path, target_val_path)

    for image in trainset_images:
        orginal_img_path = os.path.join(dataset_path, farmer_class, image)
        target_train_path = os.path.join(dataset_path, "mytrain", farmer_class, image)
        shutil.move(orginal_img_path, target_train_path)
    assert len(os.listdir(orginal_dir)) == 0
    shutil.rmtree(orginal_dir)
    print('{:^18} {:^18} {:^18}'.format(farmer_class, len(trainset_images), len(valset_images)))

    df = df.append({'class': farmer_class, "trainset": len(trainset_images), 'valset': len(valset_images)},
                   ignore_index=True)

shutil.move(dataset_path, dataset_name + '_split')
df['total'] = df['trainset'] + df['valset']
df.to_csv('E:\\dataset\\farmerdata\\data.csv', index=False)
"""
@Project: Farmer_identify
@File   : newdataset.py
@Author : Ruiqing Tang
@Date   : 2023/08/04 10:51
"""
import csv
import os
import cv2
import torch
from tqdm import tqdm
# file_path = "E:\\dataset\\农民身份识别挑战赛公开数据\\train.csv"
# mydict = {}
# with open(file_path, encoding="utf-8") as f:
#     csv_reader = csv.DictReader(f)
#     # work = 10
#     for line in csv_reader:
#         # if work > 0:
#         #     print(f"{line['name']} and {line['label']}")
#         #     work = work - 1
#         # else:
#         #     break
#         mydict.update({f"{line['name']}": f"{line['label']}"})
#         pass
# torch.save(mydict, "mydict.bin")

mydict = torch.load("mydict.bin")
orginal_path = "E:\\dataset\\farmerdata\\train"
target_path = "E:\\dataset\\farmerdata\\mytrain"
files = os.listdir(orginal_path)
for file in tqdm(files):
    img = cv2.imread(orginal_path + "\\" + file)
    for i in range(25):
        if str(i) == mydict[file]:
            cv2.imwrite(os.path.join(target_path + "\\" + str(i), file), img)
            break

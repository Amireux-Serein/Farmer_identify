"""
@Project: Farmer_identify
@File   : newfile.py
@Author : Ruiqing Tang
@Date   : 2023/08/04 10:46
"""

import os

path = "E:\\dataset\\farmerdata\\myval"
for i in range(25):
    x = "{}".format(i)
    file_name = path + "\\" + x
    os.makedirs(file_name)

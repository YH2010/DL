# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 18:21:29 2019

@author: HanY
"""

# coding:utf-8
import os
from PIL import Image

# bmp 转换为jpg，灰度图转RGB
def bmpToJpg_grayToRGB(file_path):
   for fileName in os.listdir(file_path):
       print(fileName)
#       newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
#       print(newFileName)
       im = Image.open(file_path+"\\"+fileName)
       rgb = im.convert('RGB')      #灰度转RGB
       rgb.save(file_path+"\\"+fileName)

# 删除原来的位图
#def deleteImages(file_path, imageFormat):
#   command = "del "+file_path+"\\*."+imageFormat
#   os.system(command)

def main():
   file_path = "J:\\师兄\\malignant\\JPG"
   bmpToJpg_grayToRGB(file_path)
#   deleteImages(file_path, "bmp")

if __name__ == '__main__':
   main()
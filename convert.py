#!/usr/bin/env python

import json
import os
from PIL import Image
import shutil
import re

#image_out_dir = "data/chest_xray_crop/train/NORMAL"

def convert_one_dir(from_dir, to_dir):
    image_dir = from_dir
    if not os.path.isdir(os.path.join(image_dir, "vott-json-export")):
        return
    image_files = os.listdir(os.path.join(image_dir, "vott-json-export"))
    jpeg_files = [f for f in image_files if re.match('.*json', f)]
    if len(jpeg_files) == 0:
        return
    json_path = os.path.join(image_dir, "vott-json-export", jpeg_files[0])
    if(not os.path.isfile(json_path)):
        print(json_path)
        return
    fp = open(json_path, "r")
    json_data = json.load(fp)
    image_out_dir = to_dir
    for asset_id in json_data['assets'].keys():
        asset = json_data['assets'][asset_id]
        image_filename = asset['asset']['name']
        region_list = asset['regions']
        image_file_path = os.path.join(image_dir, image_filename)
        if len(region_list) == 0:
            shutil.copyfile(image_file_path,os.path.join(image_out_dir, image_filename))
            continue
        for region in region_list:
            bbox = region['boundingBox']
            im = Image.open(image_file_path)
            im_crop = im.crop((bbox['left'], bbox['top'], bbox['left']+bbox['width'], bbox['top']+bbox['height']))
            im_crop.save(os.path.join(image_out_dir, image_filename))
            print(image_filename, region['boundingBox'])

def run(from_dir, to_dir, labels):
    for phase in ["train", "val", "test"]:
        for label in labels:
            from_dir_path = os.path.join(from_dir, phase, label)
            to_dir_path = os.path.join(to_dir, phase, label)
            if(not os.path.isdir(to_dir_path)):
                os.makedirs(to_dir_path)
            convert_one_dir(from_dir_path, to_dir_path)


#labels = ["NORMAL", "PNEUMONIA"]
#convert("data/chest_xray_exe", "data/chest_xray_crop", labels)

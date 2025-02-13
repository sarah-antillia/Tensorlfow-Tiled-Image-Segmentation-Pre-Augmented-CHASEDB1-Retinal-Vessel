# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Preprocessor.py
# 2025/01/30 sarah@antillia.com

import glob
import os
import sys
import shutil
from PIL import Image
import traceback
import io
import gzip


def resize_masks(input_dir,  output_dir, ratio=3):
    # Image_01L_2ndHO
    mask_files = glob.glob(input_dir + "/*_2ndHO.png")
    for mask_file in mask_files:
        image = Image.open(mask_file)
        w, h  = image.size
        image = image.resize((w*ratio, h*ratio))
        basename = os.path.basename(mask_file)
        name     = basename.split(".")[0]
        name     = name.replace("_2ndHO", "")
        output_file = os.path.join(output_dir, name + ".jpg")
        image.save(output_file, 'JPEG')
        print("Saved {}".format(output_file))

def resize_images(input_dir,  output_dir, ratio=3):
    image_files = glob.glob(input_dir + "/*.jpg")
    for image_file in image_files:
        basename = os.path.basename(image_file)
        name     = basename.split(".")[0]
        image = Image.open(image_file)
        w, h  = image.size
        # Expand image.
        image = image.resize((w*ratio, h*ratio))
        output_file = os.path.join(output_dir, name + ".jpg")
        image.save(output_file, 'JPEG')
        print("Saved {}".format(output_file))

if __name__ == "__main__":
  try:
    labels_dir       = "./CHASEDB1/"
    output_masks_dir = "./CHASEDB1-master/masks"
    if os.path.exists(output_masks_dir):
        shutil.rmtree(output_masks_dir)
    os.makedirs(output_masks_dir)

    resize_masks(labels_dir, output_masks_dir)

    images_dir  = "./CHASEDB1/"
    output_images_dir  = "./CHASEDB1-master/images"
    if os.path.exists(output_images_dir):
        shutil.rmtree(output_images_dir)
    os.makedirs(output_images_dir)
    resize_images(images_dir, output_images_dir)
  except:

    traceback.print_exc()

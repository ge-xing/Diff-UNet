# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.model_selection import KFold  ## K折交叉验证

import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 

def resample_img(
    image: sitk.Image,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    is_label: bool = False,
    pad_value = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image

class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d  = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):
        
        image_path = data_path[0]
        label_path = data_path[1]

        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))

        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)

        seg_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(label_path), out_spacing=[2.0, 1.5, 1.5], is_label=True))

        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)

        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data
        } 

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else :
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"数据读取出现问题，{self.datalist[i]}\n")
                if i != len(self.datalist)-1:
                    return self.__getitem__(i+1)
                else :
                    return self.__getitem__(i-1)
        if self.transform is not None :
            image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.datalist)

import glob 
def get_loader_btcv(data_dir, cache=True):
    
    all_images = sorted(glob.glob(f"{data_dir}imagesTr/*.nii.gz"))
    all_labels =  sorted(glob.glob(f"{data_dir}labelsTr/*.nii.gz"))

    all_paths = [[all_images[i], all_labels[i]] for i in range(len(all_images))]

    train_files = []
    val_files = []
    for p in all_paths:
        ## case0008
        # case0022
        # case0038
        # case0036
        # case0032
        # case0002
        # case0029
        # case0003
        # case0001
        # case0004
        # case0025
        # case0035
        ## 
        ##

        if "0008" in p[0] or "0022" in p[0] or \
             "0038" in p[0] or "0036" in p[0] or \
                 "0032" in p[0] or "0002" in p[0] or \
                    "0029" in p[0] or \
                        "0003" in p[0] or \
                            "0001" in p[0] or "0004" in p[0] or \
                                "0025" in p[0] or "0035" in p[0]:
            
            val_files.append(p)
        else :
            train_files.append(p)

    train_transform = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96,96,96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"],),
        ]
    )
    val_transform = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [   
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),            
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )

    train_ds = PretrainDataset(train_files, transform=train_transform, cache=cache)

    val_ds = PretrainDataset(val_files, transform=val_transform, cache=cache)

    test_ds = PretrainDataset(val_files, transform=test_transform)

    loader = [train_ds, val_ds, test_ds]

    return loader

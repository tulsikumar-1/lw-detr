# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:51:20 2024

@author: Administrator
"""

# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.utils.data
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, DistributedSampler
import misc

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # Prepare the data for transformations
        transformed_data = {}
        transformed_data['image'] = img
        transformed_data['bboxes'] = target['boxes'].numpy()  # Ensure bboxes are in numpy format
        transformed_data['labels'] = target['labels'].numpy()  # Ensure labels are in numpy format

        if self._transforms is not None:
            transformed_data = self._transforms(**transformed_data)  # Unpack the dict for transformations

        # Get transformed image and target
        img = transformed_data['image']
        target['boxes'] = transformed_data['bboxes']
        target['labels'] = transformed_data['labels']

        return img, target



class ConvertCoco(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    if image_set == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    if image_set == 'val':
        return A.Compose([
            A.Resize(height=640, width=640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    raise ValueError(f'unknown {image_set}')


def make_coco_transforms_square_div_64(image_set):
    if image_set == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    elif image_set == 'val':
        return A.Compose([
            A.Resize(height=640, width=640),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    raise ValueError(f'unknown {image_set}')


def Calculate_class_weights(dataset):
    labels = []
    for _, target in dataset:
        labels.extend(target['labels'].tolist())
    
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    
    # Create sample weights
    sample_weights = [class_weights[target['labels']].mean() for _, target in dataset]
    sample_weights = torch.tensor(sample_weights)
    
    return sample_weights

def build_dataset(image_folder, ann_file, image_set, batch_size, num_workers, square_div_64=False):
    if square_div_64:
        dataset = CocoDetection(image_folder, ann_file, transforms=make_coco_transforms_square_div_64(image_set))
    else:
        dataset = CocoDetection(image_folder, ann_file, transforms=make_coco_transforms(image_set))

    if image_set == 'train':
        drop_last = True

        # Initialize RandomSampler
        sampler = torch.utils.data.RandomSampler(dataset)
        
        # Use the sampler in a DataLoader
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                 collate_fn=misc.collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
        
    else:
        drop_last = False
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size, sampler=sampler_val, drop_last=drop_last,
                                 collate_fn=misc.collate_fn, num_workers=num_workers, pin_memory=True)

    return data_loader


"""
example use

img_folder='/content/traffic_monitoring/coco/train'
ann_file='/content/traffic_monitoring/coco/train/_annotations.coco.json'
image_set='val'
batch_size=8
num_workers=2
square_div_64=True
train_dataset=build_dataset(img_folder, ann_file, image_set,batch_size,num_workers,square_div_64)
"""

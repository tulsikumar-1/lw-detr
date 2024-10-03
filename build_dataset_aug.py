import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import misc
import cv2

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCoco()

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # Albumentations expects numpy arrays for image and bounding boxes
        img = np.array(img)
        h, w = img.shape[:2]

        # Normalize bounding boxes to be in range [0, 1] for Albumentations
        bboxes = target['boxes'] / torch.tensor([w, h, w, h], dtype=torch.float32)
        bboxes = bboxes.tolist()
        class_labels = target['labels'].tolist()

        if self._transforms is not None:
            transformed = self._transforms(
                image=img,
                bboxes=bboxes,
                class_labels=class_labels
            )
            img = transformed['image']

            # Denormalize bounding boxes back to pixel coordinates
            bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)

            if bboxes.size(0) > 0:  # Only process if there are bounding boxes
                bboxes *= torch.tensor([w, h, w, h], dtype=torch.float32)

                target['boxes'] = bboxes
                target['labels'] = torch.tensor(transformed['class_labels'], dtype=torch.int64)
            else:
                # Handle the case where there are no bounding boxes
                target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
                target['labels'] = torch.empty((0,), dtype=torch.int64)

        return img, target


class ConvertCoco(object):
    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        anno = [obj for obj in target["annotations"] if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Convert COCO bounding boxes (x, y, w, h) to (x_min, y_min, x_max, y_max)
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
            A.RandomResizedCrop(640, 640, scale=(0.8, 1.0), p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ColorJitter(p=0.2),
            A.Resize(640, 640),  # Ensure all images are resized to 640x640
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='xyxy', label_fields=['class_labels'], min_visibility=0.3))

    if image_set == 'val':
        return A.Compose([
            A.Resize(640, 640),  # Resize to 640x640 for validation
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='xyxy', label_fields=['class_labels'], min_visibility=0.3))

    raise ValueError(f'unknown {image_set}')


def build_dataset(image_folder, ann_file, image_set, batch_size, num_workers, square_div_64=False):
    if square_div_64:
        dataset = CocoDetection(image_folder, ann_file, transforms=make_coco_transforms(image_set))
    else:
        dataset = CocoDetection(image_folder, ann_file, transforms=make_coco_transforms(image_set))

    if image_set == 'train':
        drop_last = True
        sampler = RandomSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                 collate_fn=misc.collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=True)
    else:
        drop_last = False
        sampler_val = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler_val,
                                 collate_fn=misc.collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=True)

    return data_loader

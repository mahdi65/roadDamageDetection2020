""" COCO dataset (quick and dirty)

Hacked together by Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# from imgaug import augmenters as iaa
from bbaug.policies import policies
from random import random

class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, root, ann_file, transform=None,policy_container=policies.policies_v2()):
        super(CocoDetection, self).__init__()
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        self.policy_container = policies.PolicyContainer(policy_container)

        self.transform = transform
        self.yxyx = True   # expected for TF model, most PT are xyxy
        self.include_masks = False
        self.include_bboxes_ignore = False
        self.has_annotations = 'image_info' not in ann_file
        self.is_val = 'val' in ann_file or 'test' in ann_file
        self.coco = None
        self.cat_ids = []
        self.cat_to_label = dict()
        self.img_ids = []
        self.img_ids_invalid = []
        self.img_infos = []
        self._load_annotations(ann_file)

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.loadImgs([img_id])[0]
            if 'val' in ann_file :
                valid_annotation = True
            else :
                valid_annotation = not self.has_annotations or img_id in img_ids_with_ann
            if valid_annotation and min(info['width'], info['height']) >= 32:
                self.img_ids.append(img_id)
                self.img_infos.append(info)
            else:
                self.img_ids_invalid.append(img_id)

    def _parse_img_ann(self, img_id, img_info):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        cls = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            # To subtract 1 or not, TF doesn't appear to do this so will keep it out for now.
            if self.yxyx:
                #bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                #bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_to_label[ann['category_id']] if self.cat_to_label else ann['category_id'])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(img_id=img_id, bbox=bboxes, cls=cls, img_size=(img_info['width'], img_info['height']))

        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        # print("here")
        img_id = self.img_ids[index]
        img_info = self.img_infos[index]
        if self.has_annotations:
            ann = self._parse_img_ann(img_id, img_info)
        else:
            ann = dict(img_id=img_id, img_size=(img_info['width'], img_info['height']))

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        # if self.transform is not None:
        #     imgx, annx = self.transform(img, ann) 
        #     print("origannn",type(imgx),ann)
        ######adding policy aug
        # self.policy_container=False
        if not self.is_val and self.policy_container :
            random_policy = self.policy_container.select_random_policy()
            # print(random_policy)
            # print(ann)
            # print("beffff1111111111",type(img),ann)
                # Apply this augmentation to the image, returns the augmented image and bounding boxes
                # The boxes must be at a pixel level. e.g. x_min, y_min, x_max, y_max with pixel values
            # thisimagebbox=[ann['bbox'][1],ann['bbox'][0],ann['bbox'][3],ann['bbox'][2] ] 
            thisbbox = ann['bbox'].copy()
            thisbbox = thisbbox.tolist()
            # print("before", thisbbox)
            for item in thisbbox : 
                item[0],item[1],item[2],item[3] = item[1],item[0],item[3],item[2]
            # print("after", thisbbox)

            img_aug, bbs_aug = self.policy_container.apply_augmentation(
                random_policy,np.array(img),thisbbox,ann['cls'].tolist())

            # print("bbaug",bbs_aug)
            # print("bbaug arr",np.array(bbs_aug, dtype=np.float32))
            img_aug = Image.fromarray(np.uint8(img_aug)).convert('RGB')
            # print("tttttttt",type(img),type(np.array(img)),type(img_aug))
            #######end policy aug
            # Only return the augmented image and bounded boxes if there are
            # boxes present after the image augmentation
            # print(type(bbs_aug))
            if bbs_aug.size > 0:
                clss = []
                newbbox = []
                # print("BBSSS",bbs_aug)
                for item in bbs_aug : 
                    clss.append(int(item[0]))
                    item[0],item[1],item[2],item[3] = item[2],item[1],item[4],item[3]
                    item = np.delete(item,-1,0)
                    newbbox.append(item)
                    # print(item,"    ",item[:-1])
                # print("just finish",bbs_aug,"->",np.array(bbs_aug, dtype=np.float32))
                # print("BBSSS2222",newbbox,clss)

                ann['bbox'] = np.array(newbbox, dtype=np.float32)
                ann['cls'] = np.array(clss,dtype=np.int64)
                # img_aug.save("test.jpg") 

                # print("beffff",type(img),ann)
                if self.transform is not None:
                    img_aug, ann = self.transform(img_aug, ann)
                # xx= Image.fromarray(np.uint8(img_aug)).convert('RGB')
                # xx.save("test.jpg")
                # print("ennddddddd",type(img_aug),ann)     
                return img_aug, ann
        if self.transform is not None:
            img, ann = self.transform(img, ann)  
        return img, ann

    def __len__(self):
        return len(self.img_ids)

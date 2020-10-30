#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import json
import time
import logging
import torch
import torch.nn.parallel
try:
    from apex import amp
    has_amp = True
except ImportError:
    has_amp = False

from effdet import create_model
from data import create_loader, CocoDetection
from timm.utils import AverageMeter, setup_default_logging
from data.transforms import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import _init_paths
from utils import * 
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
# from google.colab.patches import cv2_imshow
import cv2
import torchvision.datasets as dt


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--anno', default='val2017',
                    help='mscoco annotation set (one of val2017, train2017, test-dev2017)')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-t', '--threshold', default=0.001, type=float,
                    metavar='N', help='threshold to remove boxes smaller than it(def : 0.001)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=[0.4535, 0.4744, 0.4724], metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=[0.2835, 0.2903, 0.3098], metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='mean', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
parser.add_argument('--tosave', default='./predictions', type=str, metavar='DIR',
                    help='folder to save predictions result')




getthresholds = {'d0' : [0.234,0.21,0.251,0.242],'d0aug' : [0.223,0.21,0.23,0.213],
                'd1' : [0.312,0.227,0.321,0.291],'d1aug' : [0.249,0.216,0.245,0.237],
                'd2' : [0.316,0.234,0.339,0.298],'d2aug' : [0.286,0.257,0.327,0.301],
                'd3' : [0.385,0.318,0.436,0.384],'d3aug' : [0.328,0.375,0.411,0.342],
                'd4' : [0.388,0.322,0.399,0.391],'d7' : [0.353,0.277,0.378,0.368]}
              




def drawonimage(image,boxes,th):
    # print(image)
    img_read = cv2.imread(image)
    # break
    for item in boxes : 
        if item['category_id']>0 :
            if (item['category_id'] ==1 and item['score'] >= th[0] ) or (item['category_id'] ==2 and item['score'] >= th[1] ) or \
                (item['category_id'] ==3 and item['score'] >= th[2] ) or (item['category_id'] ==4 and item['score'] >= th[3] ) or item['score'] == -1:
                color = (0,0,0)
                label = 0
                if item['category_id'] ==1:
                    label = "D00"
                    color = (0,255,255)
                elif item['category_id'] ==2:
                    label = "D10"
                    color = (0,0,255)
                elif item['category_id'] ==3:
                    label = "D20"
                    color = (255,120,255)
                else : 
                    label = "D40"
                    color = (255,0,255)
                if item['score'] >= 0.34:
                    img_read = cv2.rectangle(img_read, (int(item['bbox'][0]),int(item['bbox'][1])), (int((item['bbox'][0]+item['bbox'][2])), int(item['bbox'][1]+item['bbox'][3])), color, 2)
                    img_read = cv2.putText(img_read,  str(label),(int(item['bbox'][0]),int(item['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX , 0.7, color, 2, cv2.LINE_AA) 
                    # cv2.imwrite("../data/res2/"+"200122_"+ str(k) +"_Camera_Port4.jpg",img_read)
                elif item['score'] == -1:
                    color = (51,153,0)
                    img_read = cv2.rectangle(img_read, (int(item['bbox'][0]),int(item['bbox'][1])), (int((item['bbox'][2])), int(item['bbox'][3])), color, 2)
                    img_read = cv2.putText(img_read,  str(item['category_id']),(int(item['bbox'][0]),int(item['bbox'][1])), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.7, color, 2, cv2.LINE_AA) 

    return img_read
        #   img_read = cv2.rectangle(img_read, (int(item['bbox'][0]),int(item['bbox'][1])), (int((item['bbox'][2])), int(item['bbox'][3])), color, 2)


def getimageNamefromid(im_id):
    str_im_id = str(im_id)
    if str_im_id[0] == "1" :
        im_name = "Czech_"
    elif str_im_id[0] == "2" : 
        im_name = "India_"
    elif str_im_id[0] == "3":
        im_name = "Japan_"
    else :
        raise Exception("ERROR")   
    return im_name + str(str_im_id[1:])+".jpg"

def validate(args): 
    setup_default_logging() 


    def setthresh():
        if args.checkpoint.split("/")[-1].split("_")[0] in getthresholds.keys() :
            return getthresholds[args.checkpoint.split("/")[-1].split("_")[0]]
        else :
            a = []
            [ a.append(args.threshold) for x in range(4) ]
            return a

    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    # create model
    bench = create_model(
        args.model,
        bench_task='predict',
        pretrained=args.pretrained,
        redundant_bias=args.redundant_bias,
        checkpoint_path=args.checkpoint,
        checkpoint_ema=args.use_ema,
    )
    input_size = bench.config.image_size

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()
    if has_amp:
        print('Using AMP mixed precision.')
        bench = amp.initialize(bench, opt_level='O1')
    else:
        print('AMP not installed, running network in FP32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    if 'test' in args.anno:
        annotation_path = os.path.join(args.data, 'annotations', f'image_info_{args.anno}.json')
        image_dir = args.anno
    elif 'val' in args.anno:
        annotation_path = os.path.join(args.data, 'annotations', f'instances_{args.anno}.json')
        image_dir = args.anno
    # else:
    #     annotation_path = os.path.join(args.data, f'{args.anno}.json')
    #     image_dir = args.anno
    print(os.path.join(args.data, image_dir),annotation_path)
    dataset = CocoDetection(os.path.join(args.data, image_dir), annotation_path)

    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        fill_color=args.fill_color,
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        mean = args.mean,
        std=args.std)

    if 'test' in args.anno :
        threshold = float(args.threshold)
    # elif 'detector' in args.anno:

    #     threshold = min(getthresholds['d0'])
    else  :
        threshold= .001
    img_ids = []
    results = []
    writetofilearrtay = []
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            output = bench(input, target['img_scale'], target['img_size'])
            output = output.cpu()
            # print(target['img_id'])
            sample_ids = target['img_id'].cpu()
            
            for index, sample in enumerate(output):
                image_id = int(sample_ids[index])
                # if 'test' in args.anno :
                #     tempWritetoFile = []
                #     tempWritetoFile.append(getimageNamefromid(image_id))
                
                for det in sample:
                    score = float(det[4])
                    if score < threshold:  # stop when below this threshold, scores in descending order
                        coco_det = dict(image_id=image_id,category_id=-1)
                        img_ids.append(image_id)
                        results.append(coco_det)
                        break
                    coco_det = dict(
                        image_id=image_id,
                        bbox=det[0:4].tolist(),
                        score=score,
                        category_id=int(det[5]),
                        sizes=target['img_size'].tolist()[0]
                        )
                    img_ids.append(image_id)
                    results.append(coco_det)

            # exit()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                     )
                )
    
    # if 'test' in args.anno :
    if not os.path.exists(args.tosave):
        os.makedirs(args.tosave)
    from itertools import groupby
    results.sort(key=lambda x:x['image_id']) 

    count=0        
    for k,v in tqdm(groupby(results,key=lambda x:x['image_id'])):
        # print(args.data +"/" + str(getimageNamefromid(k)))
        img = drawonimage(os.path.join(args.data, image_dir,str(getimageNamefromid(k))),v,setthresh())              
        cv2.imwrite(args.tosave+"/"+ str(getimageNamefromid(k)), img)
        count +=1
            # print(i['category_id']," ",i['bbox'][0]," ",i['bbox'][1]," ",i['bbox'][2]," ",i['bbox'][3]," ")   
    print("generated predictions for ",count," images.")

    return results


def main():
    args = parser.parse_args()
    validate(args)

    # dataset = CocoDetection(args.data, "")
    # print(dataset[0])
    # exit

if __name__ == '__main__':
    main()

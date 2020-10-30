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
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
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
    else:
        annotation_path = os.path.join(args.data, 'annotations', f'instances_{args.anno}.json')
        image_dir = args.anno
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
    else :
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
 
    if 'test' in args.anno :
        from itertools import groupby
        results.sort(key=lambda x:x['image_id']) 

        f= open(str(args.model)+"-"+str(args.anno)+"-"+str(args.threshold)+".txt","w+")
        # for item in tqdm(writetofilearrtay):
        xxx=0        
        for k,v in tqdm(groupby(results,key=lambda x:x['image_id'])):
            xxx+=1
            f.write(getimageNamefromid(k)+",")#print(getimageNamefromid(k),", ")
            for i in v : 
              if i['category_id']>0 :
                if (i['category_id'] ==1 and i['score'] >= 0.328 ) or (i['category_id'] ==2 and i['score'] >= 0.275 ) or \
                  (i['category_id'] ==3 and i['score'] >= 0.411 ) or (i['category_id'] ==4 and i['score'] >= 0.342 ) : 
                    f.write(str(round(i['category_id']))+" "+str(round(i['bbox'][0]))+" "+str(round(i['bbox'][1]))+" "+
                                                         str(round(float(i['bbox'][0])+float(i['bbox'][2])))+" "+str(round(float(i['bbox'][1])+float(i['bbox'][3])))+" ")
            f.write('\n')    
                # print(i['category_id']," ",i['bbox'][0]," ",i['bbox'][1]," ",i['bbox'][2]," ",i['bbox'][3]," ")   
        print("counttt",xxx)
        f.close()


              

    #   f.close()
    if 'test' not in args.anno:
        array_of_dm = []
        array_of_gt =[]

    
        i=0
        # if 'test' in args.anno :



            
        for _,item in tqdm(dataset):
        # if item["img_id"] == "1000780" :
            # print(item)
            for i in range(len(item['cls'])):
                # print(str(item["img_id"]),)
                array_of_gt.append(BoundingBox(imageName=str(item["img_id"]),
                                                classId=item["cls"][i],

                                                x=item["bbox"][i][1]*item['img_scale'],
                                                y=item["bbox"][i][0]*item['img_scale'],
                                    w=item["bbox"][i][3]*item['img_scale'],
                                    h=item["bbox"][i][2]*item['img_scale'],
                                    typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2,
                                    imgSize=(item['img_size'][0],item['img_size'][1])))

            
        for item in tqdm(results) :
            if item["category_id"] >= 0: 
                array_of_dm.append(BoundingBox(imageName=str(item["image_id"]), classId=item["category_id"],
                                    classConfidence=item["score"],
                                    x=item['bbox'][0],
                                    y=item['bbox'][1],
                                    w=item['bbox'][2],
                                    h=item['bbox'][3], typeCoordinates=CoordinatesType.Absolute,
                                    bbType=BBType.Detected, format=BBFormat.XYWH, imgSize=(item['sizes'][0],item['sizes'][1])))
        myBoundingBoxes = BoundingBoxes()
    # # # # Add all bounding boxes to the BoundingBoxes object:
        for box in (array_of_gt) : myBoundingBoxes.addBoundingBox(box)
        for dm in array_of_dm : myBoundingBoxes.addBoundingBox(dm)
      
        evaluator = Evaluator()
        f1res =[]
        f1resd0 = []
        f1resd10 = []
        f1resd20 = []
        f1resd40 = []
        for conf in tqdm(range(210,600,1)) :
            metricsPerClass = evaluator.GetPascalVOCMetrics(myBoundingBoxes,
                                                            IOUThreshold=0.5,ConfThreshold=conf/1000.0)
     
            totalTP=0
            totalp =0
            totalFP=0
            tp = []
            fp = []
            ta = []
            # print('-------')
            for mc in metricsPerClass :
                tp.append(mc['total TP'])
                fp.append(mc['total FP'])
                ta.append(mc['total positives'])

                totalFP = totalFP +mc['total FP']
                totalTP = totalTP + mc['total TP']
                totalp = totalp + (mc['total positives'])

            # print(totalTP," ",totalFP," ",totalp)  
            if totalTP+totalFP == 0 :
                p =-1
            else : 
                p = totalTP/(totalTP+totalFP)
            if totalp ==0 : 
                r = -1
            else : 
                r = totalTP/(totalp)
            f1_dict = dict(
                tp=totalTP,
                fp=totalFP,
                totalp=totalp,
                conf = conf/1000.0,
                prec =p,
                rec = r,
                f1score = (2*p*r)/(p+r) 
                )
            f1res.append(f1_dict)
        #must clean these parts
            f1resd0.append(dict(tp=tp[0],fp=fp[0],
            totalp=ta[0],conf = conf/1000.0,prec =tp[0]/(tp[0]+fp[0]),rec = tp[0]/ta[0],
            f1score = (2*(tp[0]/(tp[0]+fp[0]))*(tp[0]/ta[0]))/((tp[0]/(tp[0]+fp[0]))+(tp[0]/ta[0]) )))
            
            f1resd10.append(dict(tp=tp[1],fp=fp[1],
            totalp=ta[1],conf = conf/1000.0,prec =tp[1]/(tp[1]+fp[1]),rec = tp[1]/ta[1],
            f1score = (2*(tp[1]/(tp[1]+fp[1]))*(tp[1]/ta[1]))/((tp[1]/(tp[1]+fp[1]))+(tp[1]/ta[1]) ) ))
            
            f1resd20.append(dict(tp=tp[2],fp=fp[2],
            totalp=ta[2],conf = conf/1000.0,prec =tp[2]/(tp[2]+fp[2]),rec = tp[2]/ta[2],
            f1score = (2*(tp[2]/(tp[2]+fp[2]))*(tp[2]/ta[2]))/((tp[2]/(tp[2]+fp[2]))+(tp[2]/ta[2]) ) ))
            
            f1resd40.append(dict(tp=tp[3],fp=fp[3],
            totalp=ta[3],conf = conf/1000.0,prec =tp[3]/(tp[3]+fp[3]),rec = tp[3]/ta[3],
            f1score = (2*(tp[3]/(tp[3]+fp[3]))*(tp[3]/ta[3]))/((tp[3]/(tp[3]+fp[3]))+(tp[3]/ta[3]) ) ))
         
        sortedf1 = sorted(f1res, key=lambda k: k['f1score'],reverse=True)

        f1resd0 = sorted(f1resd0, key=lambda k: k['f1score'],reverse=True)
        f1resd10 = sorted(f1resd10, key=lambda k: k['f1score'],reverse=True)
        f1resd20 = sorted(f1resd20, key=lambda k: k['f1score'],reverse=True)
        f1resd40 = sorted(f1resd40, key=lambda k: k['f1score'],reverse=True)
        


        print(sortedf1[0])
        print("\n\n") 
        print(f1resd0[0])
        print(f1resd10[0])
        print(f1resd20[0])
        print(f1resd40[0])
        # sortedf1 = sorted(f1res, key=lambda k: k['f1score'],reverse=True)
        # print(sortedf1[0:2]) 
        # json.dump(results, open(args.results, 'w'), indent=4)
        json.dump(results, open(args.results, 'w'), indent=4)
#         coco_results = dataset.coco.loadRes(args.results)
#         coco_eval = COCOeval(dataset.coco, coco_results, 'bbox')
#         coco_eval.params.imgIds = img_ids  # score only ids we've used
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()
#         print(coco_eval.eval['params'])

    json.dump(results, open(args.results, 'w'), indent=4)

    return results


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()

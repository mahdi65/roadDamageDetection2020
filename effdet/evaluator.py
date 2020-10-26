import torch
import torch.distributed as dist
import abc
import json
from .distributed import synchronize, is_main_process, all_gather_container
from pycocotools.cocoeval import COCOeval

####
# torch.backends.cudnn.benchmark = True
import _init_paths
from utils import * 
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator as f1eval
from tqdm import tqdm

class Evaluator:

    def __init__(self):
        pass

    @abc.abstractmethod
    def add_predictions(self, output, target):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass


class COCOEvaluator(Evaluator):

    def __init__(self, coco_api, distributed=False,gtboxes = None):
        super().__init__()
        self.coco_api = coco_api
        self.distributed = distributed
        self.distributed_device = None
        self.img_ids = []
        self.predictions = []
        self.gtboxes = gtboxes
        self.myBoundingBoxes = BoundingBoxes()
        for box in (self.gtboxes) : self.myBoundingBoxes.addBoundingBox(box)

    def reset(self):
        self.img_ids = []
        self.predictions = []
        self.array_of_dm =[]
        self.myBoundingBoxes = BoundingBoxes()
        for box in (self.gtboxes) : self.myBoundingBoxes.addBoundingBox(box)

    def add_predictions(self, detections, target):
        if self.distributed:
            if self.distributed_device is None:
                # cache for use later to broadcast end metric
                self.distributed_device = detections.device
            synchronize()
            detections = all_gather_container(detections)
            #target = all_gather_container(target)
            sample_ids = all_gather_container(target['img_id'])
            if not is_main_process():
                return
        else:
            sample_ids = target['img_id']

        detections = detections.cpu()
        sample_ids = sample_ids.cpu()
        for index, sample in enumerate(detections):
            image_id = int(sample_ids[index])
            for det in sample:
                score = float(det[4])
                if score < .001:  # stop when below this threshold, scores in descending order
                    break
                coco_det = dict(
                    image_id=image_id,
                    bbox=det[0:4].tolist(),
                    score=score,
                    category_id=int(det[5]))
                self.img_ids.append(image_id)
                self.predictions.append(coco_det)

    def evaluate(self):
        if not self.distributed or dist.get_rank() == 0:
            assert len(self.predictions)
            json.dump(self.predictions, open('./temp.json', 'w'), indent=4)
            results = self.coco_api.loadRes('./temp.json')
            self.array_of_dm = []
            for item in tqdm(self.predictions) :
                # print(item)
                self.array_of_dm.append(BoundingBox(imageName=str(item["image_id"]), classId=item["category_id"],
                                classConfidence=item["score"],
                                x=item['bbox'][0],
                                y=item['bbox'][1],
                                w=item['bbox'][2],
                                h=item['bbox'][3], typeCoordinates=CoordinatesType.Absolute,
                                bbType=BBType.Detected, format=BBFormat.XYWH)) #, imgSize=(item['sizes'][0],item['sizes'][1])
            for dm in self.array_of_dm : self.myBoundingBoxes.addBoundingBox(dm)
            f1evalobj = f1eval()
            f1res = []
            for conf in tqdm(range(20,50,1)) :
                metricsPerClass = f1evalobj.GetPascalVOCMetrics(self.myBoundingBoxes,
                                                            IOUThreshold=0.5,ConfThreshold=conf/100.0)
        
                totalTP=0
                totalp =0
                totalFP=0
                # print('-------')
                for mc in metricsPerClass : 
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
                    conf = conf/100.0,
                    prec =p,
                    rec = r,
                    f1score = (2*p*r)/(p+r) 
                    )
                f1res.append(f1_dict)
            # print("tp:{} fp:{}  allp:{} conf: {:.4f} precision: {:.4f} recall: {:.4f} F1-score: {:.5f}\n"
            #         .format(totalTP,totalFP,totalp,conf/100.0,p,r,(2*p*r)/(p+r)))

            sortedf1 = sorted(f1res, key=lambda k: k['f1score'],reverse=True)
            print(sortedf1[0]) 
            coco_eval = COCOeval(self.coco_api, results, 'bbox')
            coco_eval.params.imgIds = self.img_ids  # score only ids we've used
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            metric = sortedf1[0]['f1score']
            # metric = coco_eval.stats[1]  # changed to AP50 was mAP 0.5-0.95
            if self.distributed:
                dist.broadcast(torch.tensor(metric, device=self.distributed_device), 0)
        else:
            metric = torch.tensor(0, device=self.distributed_device)
            dist.broadcast(metric, 0)
            metric = metric.item()
        self.reset() 
        return metric


class FastMapEvalluator(Evaluator):

    def __init__(self, distributed=False):
        super().__init__()
        self.distributed = distributed
        self.predictions = []

    def add_predictions(self, output, target):
        passaa

    def evaluate(self):
        pass

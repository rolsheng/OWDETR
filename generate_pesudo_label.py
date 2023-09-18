import os
import os.path as osp
import torch
import json
import argparse
from tqdm import tqdm
import xml.etree.ElementTree as ET
import numpy as np
import cv2
min_area= 500
aspect_ration = 4
BackGround_ids = [0,2,3,8,9,10]
def box_area(boxs):
    return (boxs[:,3]-boxs[:,1])*(boxs[:,2]-boxs[:,0])
def box_iou(boxs,gts):
    area_boxs = box_area(boxs)
    area_gts = box_area(gts) 

    tl = np.maximum(boxs[:,None,:2],gts[:,:2])# N M 2
    br = np.minimum(boxs[:,None,2:],gts[:,2:])

    wh = np.clip(br-tl,a_min=0,a_max=None)
    inter = wh[:,:,0]*wh[:,:,1] # N M
    union = area_boxs[:,None]+area_gts-inter
    iou = inter/union
    return iou,inter
def box_nms(instances,scores,threhold=0.3):
    keep =[]
    order = np.argsort(scores)[::-1]
    while order.size>0:
        i = order[0]
        if(order.size==1):
            keep.append(i)
            break
        else:
            keep.append(i)

            tl = np.maximum(instances[[i],:2],instances[order[1:],:2])
            br = np.minimum(instances[[i],2:],instances[order[1:],2:])

            wh = np.clip(br-tl,a_min=0,a_max=None)
            inter = wh[:,0]*wh[:,1]
            area1 = box_area(instances[[i],:])
            area2 = box_area(instances[order[1:],:])
            union = area1+area2-inter
            iou = inter/union
            inds = np.nonzero(iou<threhold)[0]

            order = order[inds+1]
    return keep

def parse_args():
    parser = argparse.ArgumentParser("Collection pesudo bbox from glip and sam")
    parser.add_argument('--glip_predict',type=str,default="glip_predict",help="path to folder of glip prediction")
    parser.add_argument('--sam_predict',type=str,default="semantic_predict",help="path to folder of sam predict")
    parser.add_argument('--gt',type=str,default = 'data/OWDETR/VOC2007/Annotations',help="path to folder of annotations file")
    parser.add_argument('--save',type=str,default='data/OWDETR/VOC2007/Pesudo_label',help="save path")
    parser.add_argument('--image',type=str,default='data/OWDETR/VOC2007/JPEGImages')
    parser.add_argument('--segment',type=str,default='segment_predict',help="panotic segmentation map")
    parser.add_argument('--viz',action='store_true',default=True)
    parser.add_argument('--viz_path',default='visualize',type=str)
    return parser.parse_args()

def load_xml(path):
    root = ET.parse(path).getroot()
    gt_instances = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        obj_struct = [int(bbox.find('xmin').text),
                    int(bbox.find('ymin').text),
                    int(bbox.find('xmax').text),
                    int(bbox.find('ymax').text)]
        gt_instances.append(obj_struct)
    return np.array(gt_instances,dtype=np.int32)

def load_glip_predict(path,seg_path,overlap=0.3):
    obj_file = json.load(open(path,'r'))
    glip_instances = []
    glip_scores = []
    prior_map = np.load(seg_path)
    for obj in obj_file:
        w = int(obj['bbox'][2]-obj['bbox'][0])
        h = int(obj['bbox'][3]-obj['bbox'][1])
        x1= int(obj['bbox'][0])
        y1= int(obj['bbox'][1])
        x2= int(obj['bbox'][2])
        y2= int(obj['bbox'][3])
        if w*h > min_area and max(w/h,h/w)<=aspect_ration:
            region = prior_map[y1:y2+1,x1:x2+1].reshape(-1).tolist()
            # bg_pixels = sum([region.count(ids) for ids in BackGround_ids])
            unique_category = len(set(region))
            # if bg_pixels/(w*h) > overlap:
            #     continue
            if(unique_category<2):continue
            glip_instances.append(obj['bbox'])
            glip_scores.append(obj['score'])
    glip_instances = np.array(glip_instances,dtype=np.int32)
    glip_scores = np.array(glip_scores,dtype=np.float32)
    keep = box_nms(glip_instances,glip_scores)
    if len(keep):                
        return glip_instances[keep,:],glip_scores[keep]
    else:
        return None,None

def load_sam_predict(path,seg_path,overlap = 0.5):
    obj_file = json.load(open(path,'r'))
    sam_instances = []
    sam_scores = []
    prior_map = np.load(seg_path)
    imgh,imgw = prior_map.shape
    img_area = imgh*imgw
    for obj in obj_file:
        area = obj['bbox'][2]*obj['bbox'][3]
        w = obj['bbox'][2]
        h = obj['bbox'][3]
        x1 = int(obj['bbox'][0])
        y1 = int(obj['bbox'][1])
        x2 = int(obj['bbox'][2]+obj['bbox'][0])
        y2 = int(obj['bbox'][3]+obj['bbox'][1])
        if w*h >min_area and w*h<int(img_area/4) and max(w/h,h/w) < aspect_ration:
            score = obj['predicted_iou']
            region = prior_map[y1:y2+1,x1:x2+1].reshape(-1).tolist()

            bg_pixels = sum([region.count(ids) for ids in BackGround_ids])
            if bg_pixels/area > 0.8:
                continue
            # unique_category = len(set(region))
            # if(unique_category<5):
            #     continue
            sam_instances.append([x1,y1,x2,y2])
            sam_scores.append(score)
    if len(sam_instances):
        return np.array(sam_instances,dtype=np.int32),np.array(sam_scores,dtype=np.float32)
    else:
        return None,None
            

def merge(glip_instances,sam_instances,glip_scores,sam_scores):
    if glip_instances is None and sam_instances is None:
        return None,None
    elif glip_instances is None:
        return sam_instances,sam_scores
    elif sam_scores is None:
        return glip_instances,glip_scores
    else:
        return np.concatenate([glip_instances,sam_instances],axis=0),np.concatenate([glip_scores,sam_scores],axis=0)

def proprecess(instances,gt_instances,threshold = 0.3):
    ious,inters = box_iou(instances,gt_instances)
    keep = []
    area_instances = box_area(instances)
    for inds in range(len(ious)):
        iou = ious[inds]
        inter = inters[inds]
        if (iou<=threshold).all() and not (area_instances[inds]==inter).any():
            keep.append(inds)
    return keep

def save_instances(instances,scores):
    result_json = []
    for instance,score in zip(instances,scores):
        result_json.append({
            'bbox':instance.tolist(),
            'score':score.item()
        })
    return result_json
def viz(args,results,path):
    image = cv2.imread(path)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    for item in results:
        bbox = item['bbox']
        cv2.rectangle(image,
                      tuple(bbox[:2]),
                      tuple(bbox[2:]),
                      (0,255,0),
                      1)
    cv2.imwrite(osp.join(args.viz_path,osp.basename(path)),image)

def main(args):
    os.makedirs(args.save,exist_ok=True)
    os.makedirs(args.viz_path,exist_ok=True)
    total_images = len(os.listdir(args.image))
    for inds_img,image in enumerate(sorted(os.listdir(args.image))):
        file_prefix = image.split('.')[0]
        seg_path = osp.join(args.segment,file_prefix+'.npy')
        print(file_prefix)
        #load ground truth annotations bbox
        gt_path = osp.join(args.gt,file_prefix+'.xml')
        gt_instances = load_xml(gt_path) if osp.exists(gt_path) else None
      
        # load glip prediction
        # glip_path = osp.join(args.glip_predict,file_prefix+'.json')
        # if osp.exists(glip_path):
        #     glip_instances,glip_scores = load_glip_predict(glip_path,seg_path)
        # else:
        glip_instances,glip_scores = None,None
        #load sam prediction
        sam_path = osp.join(args.sam_predict,file_prefix+'.json')
        if osp.exists(sam_path):
            sam_instances,sam_scores= load_sam_predict(sam_path,seg_path) 
        else:
            sam_instances,sam_scores = None,None
        # merge glip and sam prediction together
        predict_instances,predict_scores = merge(glip_instances,sam_instances,glip_scores,sam_scores)
        # filter proposals with big overlaps
        if predict_instances is None:
            continue
        if gt_instances is None:
            top_instances,top_scores = predict_instances,predict_scores
        else:
            keep = proprecess(predict_instances,gt_instances)
            if len(keep)==0:
                continue
            top_instances ,top_scores= predict_instances[keep,:], predict_scores[keep]
        #save results to json
        result_json = save_instances(top_instances,top_scores)
        with open(osp.join(args.save,file_prefix+'.json'),'w') as fp:
            json.dump(result_json,fp)
        #visual
        if args.viz:
            viz(args,result_json,osp.join(args.image,image))


if __name__=="__main__":
    args = parse_args()
    main(args)
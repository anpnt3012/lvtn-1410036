import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import os
from utils.utils import *
from PIL import Image
import matplotlib.pyplot as plt
from models.deepsort.deep_sort import DeepSort
from tqdm import tqdm
import argparse
import json
import cv2
from configs import Config

class VideoTracker():
    def __init__(self, args, config):
        self.video_name = args.video_name #cam_01
        self.out_path = args.out_path
        self.cam_id = int(self.video_name[-2:])
        self.display = args.display
        
        cfg = config.cam[self.video_name]
        cam_cfg = cfg['tracking_config']
        
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end
        self.zone_path = cfg['zone']
        self.video_path = cfg['video']
        self.boxes_path = cfg['boxes']
        self.classes = config.classes
        self.idx_classes = {idx:i for idx,i in enumerate(self.classes)}
        self.num_classes = len(config.classes)
        self.width, self.height = config.size

        ## Those polygons and directions are included in the dataset
        self.polygons, self.directions = self.get_annotations()

        ## Build up a tracker for each class
        self.deepsort = [self.build_tracker(config.checkpoint, cam_cfg) for i in range(self.num_classes)]


        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        output_vid = os.path.join(self.out_path,self.video_name+'.mp4')
        self.writer = cv2.VideoWriter(output_vid,cv2.VideoWriter_fourcc(*'mp4v'), 10, (self.width,self.height))
        
    def build_tracker(self, checkpoint, cam_cfg):
        return DeepSort(
                checkpoint, 
                max_dist=cam_cfg['MAX_DIST'],
                min_confidence=cam_cfg['MIN_CONFIDENCE'], 
                nms_max_overlap=cam_cfg['NMS_MAX_OVERLAP'],
                max_iou_distance=cam_cfg['MAX_IOU_DISTANCE'], 
                max_age=cam_cfg['MAX_AGE'],
                n_init=cam_cfg['N_INIT'],
                nn_budget=cam_cfg['NN_BUDGET'],
                use_cuda=1)

    def get_annotations(self):
        with open(self.zone_path, 'r') as f:
            anno = json.load(f)
        
        directions =  {}
        zone = anno['shapes'][0]['points']
        for i in anno['shapes']:
            if i['label'].startswith('direction'):
                directions[i['label'][-2:]] = i['points']
        return zone, directions
    '''
    def submit(self, moi_detections):
        submission_path = os.path.join(self.out_path, 'submission')
        if not os.path.exists(submission_path):
            os.mkdir(submission_path)
        file_name = os.path.join(submission_path, self.video_name)
        result_filename = '{}.txt'.format(file_name)
        result_debug = '{}_debug.txt'.format(file_name)
        with open(result_filename, 'w+') as result_file, open(result_debug, 'w+') as debug_file:
            for obj_id , frame_id, movement_id, vehicle_class_id in moi_detections:
                result_file.write('{} {} {} {}\n'.format(self.video_name, frame_id, str(int(movement_id)), vehicle_class_id+1))
                if self.display:
                    debug_file.write('{} {} {} {} {}\n'.format(obj_id, self.video_name, frame_id, str(int(movement_id)), vehicle_class_id+1))
        print('Save to',result_filename,'and', result_debug)
    '''

    def run(self):
        # Dict to save object's tracks per class
        self.obj_track = [{} for i in range(self.num_classes)]
        vidcap = cv2.VideoCapture(self.video_path)
        idx_frame = 0
        frame_id =-1
        movement_id = -1
        results={}
        try:
            with tqdm(total=self.frame_end) as pbar:

                num_obj     =[]
                count_veh_00=[]
                count_veh_01=[]
                count_veh_02=[]
                count_veh_03=[]

                count_veh_lane1_00=[]
                count_veh_lane1_01=[]
                count_veh_lane1_02=[]
                count_veh_lane1_03=[]

                count_veh_lane2_00=[]
                count_veh_lane2_01=[]
                count_veh_lane2_02=[]
                count_veh_lane2_03=[]

                while vidcap.isOpened():
                    success, im = vidcap.read()
                    if idx_frame < self.frame_start:
                        idx_frame+=1
                        continue
                    anno = os.path.join(self.boxes_path, str(idx_frame).zfill(5) + '.json')
                    if not success:
                        break

                    moi_detections = counting_moi(self.directions ,self.obj_track, self.polygons, self.cam_id)
                    #print(str(moi_detections))

                    ## Draw polygons to frame
                    ori_img = im[..., ::-1]
                    overlay_moi = im.copy()
                    alpha = 0.2
                    cv2.fillConvexPoly(overlay_moi, np.array(self.polygons).astype(int), (255,255,0))
                    #cv2.arrowedLine(overlay_moi, np.array(self.directions).astype(int), (255,255,0))
                    im_moi = cv2.addWeighted(overlay_moi, alpha, im, 1 - alpha, 0)
                    cv2.putText(im_moi,"Frame_id: " + str(idx_frame), (10,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,255,0) , 2)
                    
                    stt=0
                    for obj_id , frame_id, movement_id, vehicle_class_id in moi_detections:
                      results[stt] = {'obj_id' : [obj_id],'frame_id' : [frame_id],'movement_id':[movement_id],'vehicle_class_id' : [vehicle_class_id]}
                      stt=stt+1
                    
                    
                    ## Read in detection results
                    try:
                        with open(anno, 'r') as f:
                            objs = json.load(f)
                    except FileNotFoundError:
                        print("Tracked {} frames".format(idx_frame+1))
                        break

                    bbox_xyxy = np.array(objs['bboxes'])
                    cls_conf = np.array(objs['scores'])
                    cls_ids = np.array(objs['classes'])

                    # Check only bbox in roi
                    ## Those rois are provided with the dataset
                    mask = np.array([1 if check_bbox_intersect_polygon(self.polygons,i.tolist()) else 0 for i in bbox_xyxy])
                    bbox_xyxy_ = np.array(bbox_xyxy[mask==1])
                    cls_conf_ = np.array(cls_conf[mask==1])
                    cls_ids_ = np.array(cls_ids[mask==1])

                    for i in range(self.num_classes):
                        mask = cls_ids_ == i
                        bbox_xyxy = bbox_xyxy_[mask]
                        cls_conf = cls_conf_[mask]
                        cls_ids = cls_ids_[mask]

                        if len(cls_ids) > 0:
                            outputs = self.deepsort[i].update(bbox_xyxy, cls_conf, ori_img)

                            ## Save object's tracks for later counting
                            ###     identity: object's id number
                            ###     label: object's class
                            ###     coords: center of object's bounding box
                            ###     frame_id: the current frame
                            ###     movement_id: the current lane
                            for obj in outputs:
                                identity = obj[-1]
                                center = [(obj[2]+obj[0]) / 2, (obj[3] + obj[1])/2]
                                label = i
                                 
                                #print(str(i)+"|"+str(identity)+"|"+str(center))
                                if identity not in self.obj_track[i]:
                                    self.obj_track[i][identity] = {
                                        'identity': identity,
                                        'labels': [label],
                                        'coords': [center],
                                        'frame_id': [idx_frame]
                                    }
                                else:
                                    self.obj_track[i][identity]['labels'].append(label)
                                    self.obj_track[i][identity]['coords'].append(center)
                                    self.obj_track[i][identity]['frame_id'].append(idx_frame)
                                    self.obj_track[i][identity]['identity'] = identity

                            # print("===")
                            #print(self.obj_track[i])

                            #for x in self.obj_track[i][identity]['movement_id']:
                              #print(x)
                              #print("===== ===== =====", identity, center, label,movement_id)
                                
                    
                            im_show = re_id(outputs, im_moi, labels=i)
                        else:
                            im_show = im_moi

                        lane = 0
                        idtxt = 0
                        for k,v in self.obj_track[i].items():
                          
                            idframe = v['frame_id'][-1]
                            idx = v['identity']
                            #print(str(idframe))
                            #print(str(v['identity']))

                            #tim lane cho cac loai xe------
                            for a,x in results.items():
                                  if x['obj_id'] == idx:
                                    idtxt = x['obj_id']
                                    lane = x['movement_id']
                                    if x['frame_id']==[idframe]:
                                      lane = x['movement_id']
                                      #print(str(identity)+"|"+str(frame_id)+"|"+str(movement_id))
                                      break

                            #tim lane cho cac loai xe------
                            #print(str(lane))
                            for k1,v1 in v.items():
                                ## xe may
                                if k1=="labels" and v1[0]==0:
                                  if k not in count_veh_00:
                                    count_veh_00.append(k)
                                    #print(str(idx)+" lane:"+str(lane)+": "+str(len(count_veh_00)))
                                    if lane != [1] and lane != [2]:
                                      print(str(k))
                                    if k not in count_veh_lane1_00 and lane == [1]:
                                      count_veh_lane1_00.append(k)
                                      #print("lane1 :"+str(len(count_veh_lane1_00)))
                                    if k not in count_veh_lane2_00 and lane == [2]:
                                      count_veh_lane2_00.append(k)
                                      #print("lane1 :"+str(len(count_veh_lane2_00)))

                                ## xe hoi
                                if k1=="labels" and v1[0]==1:
                                    if k not in count_veh_01:
                                      count_veh_01.append(k)
                                      if k not in count_veh_lane1_01 and lane == [1]:
                                        count_veh_lane1_01.append(k)
                                      if k not in count_veh_lane2_01 and lane == [2]:
                                        count_veh_lane2_01.append(k)

                                ## xe buyst
                                if k1=="labels" and v1[0]==2:
                                    if k not in count_veh_02:
                                      count_veh_02.append(k)
                                      if k not in count_veh_lane1_02 and lane == [1]:
                                        count_veh_lane1_02.append(k)
                                      if k not in count_veh_lane2_02 and lane == [2]:
                                        count_veh_lane2_02.append(k)
                                
                                ## xe tai
                                if k1=="labels" and v1[0]==3:
                                    if k not in count_veh_03:
                                      count_veh_03.append(k)
                                      if k not in count_veh_lane1_03 and lane == [1]:
                                        count_veh_lane1_03.append(k)
                                      if k not in count_veh_lane2_03 and lane == [2]:
                                        count_veh_lane2_03.append(k)

                        im_show = cv2.arrowedLine(im_show, (570, 180), (300, 700), (0, 255, 0), 2)
                        cv2.putText(im_show,"1", (270,700), cv2.FONT_HERSHEY_COMPLEX , 1 , (0,255,0) , 1)   
                        im_show = cv2.arrowedLine(im_show, (620, 700), (650, 180), (0, 0, 255), 2)
                        cv2.putText(im_show,"2", (650,180), cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,255) , 1)

                        cv2.rectangle(im_moi,(1280-500,0),(1280,140),(255, 255, 255),-1)
                        cv2.putText(im_show, "motorcycle " + "car " + "bus " + "truck ", (1280-400,30), cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,0) , 1)
                        cv2.putText(im_show, " lane1:    "+str(len(count_veh_lane1_00)) +"      "+str(len(count_veh_lane1_01)) +"   "+str(len(count_veh_lane1_02)) +"   "+str(len(count_veh_lane1_03)) , (1280-500,60), cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,255,0) , 1)
                        cv2.putText(im_show, " lane2:    "+str(len(count_veh_lane2_00)) +"      "+str(len(count_veh_lane2_01)) +"   "+str(len(count_veh_lane2_02)) +"   "+str(len(count_veh_lane2_03)) , (1280-500,90), cv2.FONT_HERSHEY_SIMPLEX , 1 , (0,0,255) , 1)
                        

                    if self.display:
                        self.writer.write(im_show)
                    
                    idx_frame += 1
                    pbar.update(1)
        except KeyboardInterrupt:
            pass
       
        ### You can use the obj_track to output some numeric results such as counting
        
        with open(os.path.join(self.out_path, f'{self.video_name}_tracking_result.json'), 'w') as fout:
            print("----")
            print("name: ", self.video_name)
            print("path: ", self.out_path)
            json.dump(str(self.obj_track), fout)


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference AIC Challenge Dataset')
    parser.add_argument('video_name', help='configuration cam file')
    parser.add_argument('--out_path', type=str, default='results', help='output path') 
    parser.add_argument('--frame_start', default = 0,  help='start at frame')
    parser.add_argument('--frame_end', default = 6000,  help='end at frame')
    parser.add_argument('--config', type=str, default='cam_configs.yaml', help='configuration cam file')
    parser.add_argument('--display', action='store_true', default = True, help='debug print object id to file')          
    args = parser.parse_args()
    configs = Config(os.path.join('configs',args.config))
    tracker = VideoTracker(args, configs)
    tracker.run()
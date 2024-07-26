from ultralytics import YOLO 
import supervision as sv 
from tensorflow import keras 
from utils import videoRead 
from utils import draw_elips
from utils import draw_rect_and_put_tract_num
from utils import draw_circle
from utils import draw_rect
from utils import drwa_points
import numpy as np 
import cv2 
import pandas as pd 
import os 
import pickle 

class detectionObject:

    def __init__(self,path_ball,path_without_ball,path_keypoint):
        #load the model 
        self.model_ball=YOLO(path_ball)
        self.model_without_ball=YOLO(path_without_ball)
        # load the keupoints detection model 
        self.key_point_model=keras.models.load_model(path_keypoint)
        self.track=sv.ByteTrack()

    def predictingKeypoints(self,frames,h,w):
        # reshape the video frame
        reshape_frames = []
        for frame in frames:
            frame = cv2.resize(frame, (224, 224))
            reshape_frames.append(frame)
        prediction=self.key_point_model.predict(np.array(reshape_frames))

        prediction[:,:,0]=(prediction[:,:,0]/224)*w
        prediction[:,:,1]=(prediction[:,:,1]/224)*h

        return prediction.tolist()
    
    def interpolate_ball_position(self,info_dic):
        ball_box=[val.get('bbox',[]) for val in info_dic]
        df=pd.DataFrame(ball_box, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate()
        df = df.bfill()
        ball_position = [{'bbox': val} for val in df.to_numpy().tolist()]
        return ball_position 

    def interpolatePlayerPosition(self,info):
        #for 1st player 
        track_id1=2
        d1=[i.get(track_id1,{}).get('bbox',[]) for i in info['player']]
        #for 2nd player
        track_id2=4
        d2=[i.get(track_id2,{}).get('bbox',[]) for i in info['player']]

        #convert it into data frame 
        df1=pd.DataFrame(d1,columns=['x1','y1','x2','y2'])
        df2=pd.DataFrame(d2,columns=['x1','y1','x2','y2'])

        #interpolate both the frame 
        df1 = df1.interpolate()
        df1 = df1.bfill()
        df2 = df2.interpolate()
        df2 = df2.bfill()
        
        # data frame to list 
        df2list1=df1.to_numpy().tolist()
        df2list2=df2.to_numpy().tolist()
        for i in range(len(info['player'])):
            
            info['player'][i][track_id1]={'bbox':df2list1[i]}
            info['player'][i][track_id2]={'bbox':df2list2[i]}
        
         




    def detect_bbox_ball(self,input_video_apth,save_info_path=None):
        #first read the video 
        frames,fps=videoRead(input_video_apth)

        if (save_info_path is not None) and (os.path.exists(save_info_path)) :
            
            with open(save_info_path,'rb') as file:
                info=pickle.load(file)
            file.close()
            return info,frames,fps


        total_num_frames=len(frames)
        num_of_frame_in_batch=20
        h,w,_=frames[0].shape
        bbox_prediction=[]
        ball_prediction=[]
        keypoint_prediction=[]
        # let we want to predict max 20 frame in once 
        for indx in range(0,total_num_frames,num_of_frame_in_batch):
            last_indx=indx+num_of_frame_in_batch
            temp_frames=frames[indx:last_indx]
            temp_box_pred=self.model_without_ball.predict(temp_frames,conf=0.3)
            temp_ball_pred=self.model_ball.predict(temp_frames,conf=0.3)
            temp_keypoint_pred=self.predictingKeypoints(temp_frames,h,w)

            bbox_prediction += temp_box_pred
            ball_prediction +=temp_ball_pred
            keypoint_prediction +=temp_keypoint_pred
    
        # find the avg og the detected key point 
        keypoint_prediction=np.array(keypoint_prediction)
        avg_point=np.mean(keypoint_prediction,axis=0)

        info = {'player': [],
                'ball': [],
                'net':[],
                'court':[],
                'avgkeypoint': []}

        for frame_num,detectn in enumerate(bbox_prediction):
            detectn=sv.Detections.from_ultralytics(detectn)
            detectn = self.track.update_with_detections(detectn)

            info['player'].append({})
        
    
            for tracker_detection in detectn:
                bbox = tracker_detection[0].tolist()
                cls_name = tracker_detection[5]['class_name']

                tracker_id = tracker_detection[4]

                if (cls_name == 'court') and (frame_num == 0) :
                    info['court'].append({'bbox':bbox})

                if (cls_name == 'net') and (frame_num == 0) :
                    info['net'].append({'bbox':bbox})

                if cls_name == 'player':
                    info['player'][frame_num][tracker_id]={'bbox':bbox}
            
            ball_detect=ball_prediction[frame_num].boxes.xyxy
            ball_detect=ball_detect.tolist()
            if len(ball_detect)>0:
                ball_detect=ball_detect[0]
            info['ball'].append({'bbox':ball_detect})
        info['avgkeypoint']=avg_point
        
        info['ball']=self.interpolate_ball_position(info['ball'])
        self.interpolatePlayerPosition(info) 

        with open('info1.pkl','wb') as file:
            pickle.dump(info,file)
        file.close()


        return info,frames,fps 

    def annotate_frames(self,info,frames):

        out_frames=[]
        for frame_num,value in enumerate(info['player']):

            frame=frames[frame_num].copy()
            pres_id=[]
            count=0
            for track_id,bboxinfo in value.items():
                """if track_id not in  pres_id:
                    pres_id.append(track_id)
                    count +=1
                    track_id = count

                else:
                    track_id = np.where(np.array(prers_id)==track_id)[0][0]
                """
                color =  (0, 0, 255)
                frame = draw_elips(
                    frame, bboxinfo['bbox'], color)
                color = (255, 255, 255)
                frame = draw_rect_and_put_tract_num(
                    frame, bboxinfo['bbox'], color, track_id)

            ball_box=info['ball'][frame_num]['bbox']
            color = (0, 255, 255)
            frame=draw_circle(frame,ball_box,color)  
            #draw net 
            color=(255,255,255)
            bbox=info['net'][0]['bbox']
            frame =  draw_rect(frame,bbox,color,'net')

            #draw court 
            color=(255,255,255)
            bbox=info['court'][0]['bbox']
            frame =  draw_rect(frame,bbox,color,'court')

            #draw points 
            color = (255, 0, 0)
            avg_point=info['avgkeypoint']
            frame=drwa_points(frame,avg_point,color)

            out_frames.append(frame)

        return out_frames


      




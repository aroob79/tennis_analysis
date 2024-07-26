import numpy as np 
import pandas as pd 
from utils import distance 

class SpeedDistance:
    def __init__(self):

        self.meter2pixesl=21.9  # 1 meter = 21.9 pixel 
        self.pixel2meter=1/21.9 # 

    def findBallHitIndex(self,info):
        #first find the frame where ball was hitted 
        d=[val['transformpoint'] for val in info['ball']]
        df=pd.DataFrame(d,columns=['x','y'])
        #find the rolling mean 
        df['mov_avg']=df['y'].rolling(window=10,min_periods=1,center=True).mean()
        #find the diff of the rolling mean 
        df['diff']=df['mov_avg'].diff(periods=3)/2

        #find those frame where the plote crosses the zero 
        start = df['diff'].iloc[1]
        index = []
        i = 1
        for i in range(len(df)-1):
            temp = df['diff'].iloc[i + 1]
            if (temp * start == 0) or (temp * start < 0):
                index.append(i)
            start = df['diff'].iloc[i + 1]

        # then filter those index 

        final_indx=[]
        for i in range(len(index)-1):
            start=index[i]
            nextindx=index[i+1]

            if (nextindx - start )>30:
                final_indx.append(start)
            if i == (len(index)-2) :
                final_indx.append(nextindx)

        return final_indx 


    def shortSpeed(self,info,fps):

        indx=self.findBallHitIndex(info)
        l=len(indx)
        
        for i in range(l-1):
            start_indx=indx[i]
            end_indx=indx[i+1]
            start_ball_position=info['ball'][start_indx]['transformpoint']
            end_ball_position=info['ball'][end_indx]['transformpoint']

            #find the distance in pixel 
            dist=distance(start_ball_position,end_ball_position)

            #convert pixel to meter 
            distMeter= dist * self.pixel2meter

            time=(end_indx - start_indx)/fps 

            #find the velocity 
            ball_velocity=(distMeter / time)*3.6

            info['ball'][start_indx]['ball_velocity']=ball_velocity 

    def playerVelocity(self,info,fps):

        #for getting the stable velocity i willl find the avg velocity of 5 frames 
        l=len(info['player'])
        next_frame_to_cal=3
        for i in range(0,l-1,next_frame_to_cal):
            start_index= i 
            end_index= i+next_frame_to_cal 
            if end_index > (l-1):
                end_index= l-1 

            start_info=info['player'][start_index]
            end_info=info['player'][end_index]
            for track_id,box_info in start_info.items():
                start_point=box_info['transformpoint']
                end_point=end_info[track_id]['transformpoint']
                dist=distance(start_point,end_point)
                #convert pixel to meter 
                distMeter= dist * self.pixel2meter

                time=(end_index - start_index)/fps 

                #find the velocity 
                velocity=(distMeter / time)*3.6
                info['player'][start_index][track_id]['velocity']=velocity
















   
    
        




        
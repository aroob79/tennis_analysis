import cv2 
import numpy as np 
import pickle 
from utils import footPoint
from utils import midPoint
class homographyTransformation:
    def __init__(self,sourcePoints):
        self.sourcePoints=sourcePoints
        self.destPoints=np.array([[70,70+70],
                     [310,70+70],
                     [70,590+70],
                     [310,590+70],
                     [90,70+70],
                     [90,590+70],
                     [270,70+70],
                     [270,590+70],
                     [90,190+70],
                     [270,190+70],
                     [90,470+70],
                     [270,470+70],
                     [180,190+70],
                     [180,470+70]],dtype=np.float32)

        self.meter2pixesl=21.9  # 1 meter = 21.9 pixel 
        self.pixel2meter=1/21.9 # 

        self.h,_= cv2.findHomography(self.sourcePoints,self.destPoints)
        self.point2connect=np.array([[0,1],
                                     [0,2],
                                     [4,5],
                                     [6,7],
                                     [1,3],
                                     [2,3],
                                     [8,9],
                                     [10,11],
                                     [12,13]])

    def drawTransparentcourt(self,frame):
        startPoint=(20,70)
        endPoint=(360,630+70+50)
        rectangle_color = (255, 255, 255)  
        alpha = 0.8
        overlay = frame.copy()

        # Draw the rectangle on the overlay image
        cv2.rectangle(overlay, startPoint, endPoint, rectangle_color, -1)

        # Blend the overlay with the original image using the transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) 

        return frame


    def drawMinicourt(self,frames):
        outFrames=[]
        for frame in frames:
            image = frame.astype(np.uint8).copy()
            image=self.drawTransparentcourt(image)
            for i,point in enumerate(self.destPoints):
                # Define the point coordinates
                point = tuple(point.astype(np.int32)) # Coordinates (x, y)
                # Define the color (BGR format) and radius of the point
                color = (255, 0, 0)  
                radius = 5  # Radius of the point (circle)

                # Define the thickness of the circle outline (-1 to fill the circle)
                thickness = -1

                # Draw the point on the image
                image=cv2.circle(image, point, radius, color, thickness)

            #draw line 
            for linePoint in self.point2connect:
                start_point = tuple(self.destPoints[linePoint[0]].astype(np.int32))  # Coordinates (x1, y1)
                end_point = tuple(self.destPoints[linePoint[1]].astype(np.int32))     # Coordinates (x2, y2)

                # Define the color (BGR format) and thickness of the line
                color = (0, 255, 0)  # Green color
                thickness = 2        # Thickness of the line

                # Draw the line on the image
                image=cv2.line(image, start_point, end_point, color, thickness)
            #find mid point of the court for draw net 
            mid_x=int(self.destPoints[0][0])
            mid_y=int((self.destPoints[0][1]+self.destPoints[2][1])/2)
            image=cv2.circle(image, (mid_x,mid_y), 5, (0, 0, 255), -1)
            image=cv2.circle(image, (int(self.destPoints[1][0]),mid_y), 5, (0, 0, 255), -1)
            start_point=(mid_x,mid_y)
            end_point=(int(self.destPoints[1][0]),mid_y)
            image=cv2.line(image, start_point, end_point, color, thickness)
            outFrames.append(image)
        return outFrames   

    def transformPixel2Meter(self,info,frames):
        outFrames=[]
        for frame_num,value in enumerate(info['player']):
            frame=frames[frame_num].copy()

            #covert the player bounding box
            for track_id,bbox in value.items():
                box=bbox['bbox']

                #convert the bounding box to foot point by taking (xmin,ymax)
                footPoint_=footPoint(box)
                footPoint_=np.array([[[footPoint_[0],footPoint_[1]]]])

                # transform the point to minicourt version 
                transformPoint=cv2.perspectiveTransform(footPoint_,self.h)
                transformPoint=transformPoint.ravel().astype(np.int32).tolist()
                #draw the foot point to the mini cpurt 
                color = (0, 0, 255)  
                radius = 5  # Radius of the point (circle)
                # Define the thickness of the circle outline (-1 to fill the circle)
                thickness = -1
                # Draw the point on the image
                
                frame=cv2.circle(frame, tuple(transformPoint), radius, color, thickness)
                #store the point in to the dict
                info['player'][frame_num][track_id]['transformpoint']=transformPoint
            

            #convert the ball posiotion 
            bbox_dict=info['ball'][frame_num]
            box=bbox_dict['bbox']
            
            #convert the bounding box to foot point by taking (xmin,ymax)
            midpoint=midPoint(box)
            midpoint=np.array([[[midpoint[0],midpoint[1]]]])
            
            # transform the point to minicourt version 
            transformPoint=cv2.perspectiveTransform(midpoint,self.h)
    
            transformPoint=transformPoint.ravel().astype(np.int32).tolist()
            #draw the foot point to the mini cpurt 
            color = (0, 0, 0)  
            radius = 5  # Radius of the point (circle)
            # Define the thickness of the circle outline (-1 to fill the circle)
            thickness = -1
            # Draw the point on the image    
            frame=cv2.circle(frame, tuple(transformPoint), radius, color, thickness)
            #store it into the info dict
            info['ball'][frame_num]["transformpoint"]=transformPoint

            outFrames.append(frame)
            
        #save the edited dict 
        with open('info_edit.pkl','wb') as file:
            pickle.dump(info,file)
        file.close()

        return outFrames





   



        
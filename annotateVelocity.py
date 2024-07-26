import numpy as np 
import cv2 



class drawVelocity:
    def __init__(self):
        #define cornere point
        self.points=np.array([[1250,920],
                              [1880,920],
                              [1250,1040],
                              [1880,1040]])
        self.bg_color=(0,0,0)
        self.font_color=(255,255,255)

    def drawCanvas(self,frames):
        outframes=[]
        startPoint=self.points[0]
        endPoint =self.points[3]
        for frame in frames:
            
            alpha = 0.8
            overlay = frame.copy()

            # Draw the rectangle on the overlay image
            cv2.rectangle(overlay, startPoint, endPoint, self.bg_color, -1)

            # Blend the overlay with the original image using the transparency factor
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame) 
            outframes.append(frame)
        return outframes

    def writeVelocity(self,frames,info):
        frames=self.drawCanvas(frames)
        track_id1,track_id2=list(info['player'][0].keys())
        player1_velocity =0
        player2_velocity=0
        ball_velocity1=0
        ball_velocity2=0
        ball_velocity=0
        temp=0
        outframes=[]
        color=self.font_color
        for frame_num,frame in enumerate(frames):

            player1_velocity=info['player'][frame_num][track_id1].get('velocity',player1_velocity)
            player2_velocity=info['player'][frame_num][track_id2].get('velocity',player2_velocity)
            ball_velocity=info['ball'][frame_num].get('ball_velocity',ball_velocity)
            if (ball_velocity !=0) and (temp == 0):
                ball_velocity1 = ball_velocity
                temp =1
            elif (ball_velocity !=0) and (temp == 1):
                ball_velocity2 = ball_velocity
                temp =0
            else:
                pass 

            #write the heading 
            position=(1260,940)
            text=f'              player1         player2'
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, text, position, font,
                        0.8, color, 2, cv2.LINE_AA)
            
            #write the ball velocity 
            position=(1260,980)
            text=f'ball speed    {ball_velocity1:0.2f} km/h   {ball_velocity2:.2f} km/h'
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, text, position, font,
                        0.8, color, 1, cv2.LINE_AA)
            
            #write the palyer velocity 
            position=(1260,1020)
            text=f'player speed {player1_velocity:.2f} km/h    {player2_velocity:.2f} km/h'
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, text, position, font,
                        0.8, color, 1, cv2.LINE_AA)         
            ball_velocity =0   
            outframes.append(frame)

        return outframes
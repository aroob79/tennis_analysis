from ultralytics import YOLO 
from tensorflow import keras 
from detection import detectionObject
from utils import write_video 
from homography_transformation import homographyTransformation 
from analysisSpeed import SpeedDistance 
from annotateVelocity import drawVelocity
import pickle 

path_ball_model=r'best_with_ball.pt'
path_without_ball_model=r'best_without_ball.pt'
video_path='input_video.mp4'
detect=detectionObject(path_ball=path_ball_model,path_without_ball=path_without_ball_model
                ,path_keypoint='best_model.keras')

info,frames,fps = detect.detect_bbox_ball(video_path,'info1.pkl')

frames=detect.annotate_frames(info,frames)
homoTrans=homographyTransformation(info['avgkeypoint'])
frames=homoTrans.drawMinicourt(frames)
frames=homoTrans.transformPixel2Meter(info,frames)

speed_dist=SpeedDistance()
speed_dist.shortSpeed(info,fps)
speed_dist.playerVelocity(info,fps)

drawv=drawVelocity()
frames=drawv.writeVelocity(frames,info)
with open('info_edit1.pkl','wb') as file:
    pickle.dump(info,file)
file.close()
write_video(frames,'output_video.avi')





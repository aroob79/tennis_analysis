import numpy as np
import cv2


def videoRead(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []

    while video.isOpened:
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            break
    video.release()
    return frames,fps

def draw_elips(frame, bbox, color):

    x_cen = int((bbox[0]+bbox[2])/2.)
    y_cen = int(bbox[3])-15
    width = 60
    frame = cv2.ellipse(frame, (x_cen, y_cen),
                        (width, int(0.5*width)), 0, -60, 240, color, 2)
    return frame


def draw_rect_and_put_tract_num(frame, bbox, color, tracker_id):
    x_cen = int((bbox[0]+bbox[2])//2.)
    upper_corner = (int(x_cen-20), int(bbox[3]+35))
    lower_corner = (int(x_cen+150), int(bbox[3]+65))
    frame = cv2.rectangle(frame, upper_corner, lower_corner, color, -1)
    text = f'Palyer Id:{tracker_id}'
    position = (int(x_cen+2), int(bbox[3]+60))
    color = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, text, position, font,
                        0.8, color, 1, cv2.LINE_AA)
    return frame

def draw_circle(frame,bbox,color):
    x_cen=(bbox[0]+bbox[2])/2
    y_cen=(bbox[1]+bbox[3])/2
    center = (int(x_cen),int(y_cen))
    radius = 10

    # Draw the filled circle
    frame=cv2.circle(frame, center, radius, color, 3)
    return frame 

def draw_rect(frame,bbox,color,text):
    upper_cor=(int(bbox[0]),int(bbox[1]))
    lowwer_cor=(int(bbox[2]),int(bbox[3]))
    frame = cv2.rectangle(frame, upper_cor, lowwer_cor, color, 1)
    color = (0, 0, 0)
    position = (upper_cor[0], upper_cor[1]-5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, text, position, font,
                        0.8, color, 1, cv2.LINE_AA)
    return frame 


def drwa_points(frame,avg_point,color):

    for point in avg_point:
        # Define the center and radius of the circle
        center = (int(point[0]),int(point[1]))
        radius = 10

        # Define the color (BGR) and thickness (-1 for filled)  
        thickness = -1  # Thickness of -1 px will fill the circle

        # Draw the filled circle
        frame=cv2.circle(frame, center, radius, color, thickness)
    return frame

#write video 
def write_video(frames, videoname):
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = 25

    resolution = (frames[0].shape[1], frames[0].shape[0])
    video = cv2.VideoWriter(videoname, codec, fps, resolution)
    for frame in frames:
        video.write(frame)
    video.release()
    return

def footPoint(box):
    x=(box[0]+box[2])/2
    y=box[3]

    return (x,y)

def midPoint(box):
    x=(box[0]+box[2])/2
    y=(box[1]+box[3])/2

    return (x,y)

def distance(start,end):
    return  ((start[0]-end[0])**2 + (start[1]-end[1])**2)**0.5


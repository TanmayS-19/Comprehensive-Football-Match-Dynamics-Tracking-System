from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


#create a class Tracker to track the bounding boxes
class Tracker:
    def __init__(self, model_path):
        #init gets called when we initialize attributes to the class
        #load in the tracker and model inside init
        self.model = YOLO(model_path) #by using the model we first detect the frames and then proceed with tracking
        self.tracker = sv.ByteTrack()

    #to track the bbox acc to the camera movement
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    #to interpolate the ball position
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        #back fill in case of first one being the missing detection
        df_ball_positions = df_ball_positions.bfill()
        #return back the ball position in original format
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def detect_frames(self, frames):
        batch_size=20
        #batch size limited to 20x20 frames to avoid memory issues
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections
    
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):        
        #avoid the whole process of creating track dict if pickle file already exists (saves time)
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        #we create a tracks dict of lists such that we can reference players,ref,ball individually
        #track id and corresponding bb for each player, referee in the frame and one single ball
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        #we overwrite gk as a normal player bc there is no exclusive changes for gk
        
        #enumerate generates pairs of the index (starting from 0 by default) and the value from the iterable(detections)
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}


            #convert the detections to supervision format in order to proceed with the tracking
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #convert gk to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]


            #now we can write our tracker
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            #dict has a key for track_id and value as the boundingbox
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            #we loop over each tracks to extract corresponding bounding boxes        
            
            #no need to track the ball as there is only one bounding box across frames
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3]) #bc we want the ellipse to be at the bottom
        x_center, _ = get_center_of_bbox(bbox) #we want it to be middle of the x
        width = get_bbox_width(bbox) #radius of ellipse

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            #major and minor axis of an ellipse
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            #not tracing complete circles to give a better look
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        #define rectangle for player index overlapping the circle
        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        #if track id exists then draw a rectangle
        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            #define the number
            x1_text = x1_rect+12 #random padding
            if track_id > 99:
                x1_text -=10 #start on the left if bigger number
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)), #position of box
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,#font size
                (0,0,0),#black color
                2 #thickness
            )
        return frame

    #to draw a triangle above the ball
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2) #borders

        return frame


#we replace the bounding boxes with minimal circles to eliminate overlapping


    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 ) #filled (-1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        total_possession_frames = team_1_num_frames + team_2_num_frames
        if total_possession_frames > 0:
            team_1 = team_1_num_frames / total_possession_frames
            team_2 = team_2_num_frames / total_possession_frames
        else:
            team_1 = team_2 = 0  # No possession recorded

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame    


    def draw_annotations(self,video_frames, tracks,team_ball_control):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
    

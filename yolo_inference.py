#import yolo model from ultrlytics
from ultralytics import YOLO
model = YOLO('models/best.pt')

#run the out of the box model on our vid and also save it
#model goes through each frame (750 total) and predicts video
results = model.predict('input_videos/08fd33_4.mp4',save=True)

#print results for the first frame
print(results[0])
print("*********************\n")

#prints boxes of the 0th frame
#each box has a class id, conf, diff bb coordinates
for box in results[0].boxes:
    print(box)




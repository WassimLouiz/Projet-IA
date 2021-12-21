import cv2
import torch
import numpy as np



#let's load the age detection model and proto
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/3éme/Projet IA/final code/yolov5/runs/train/exp10/weights/best.pt', force_reload=True)


#we call our models 
ageNet = cv2.dnn.readNet(ageModel,ageProto)

#create our list of values:
ageList =['(0-4)','(4-6)', '(8-12)', '(15-20)', '(20-25)','(25,30)', '(30-45)', '(45-55)', '(60-100)']

MODEL_MEAN_VALUES= (78.4263377603, 87.7689143744, 114.895847746)

#real time video capture, camera feed
video = cv2.VideoCapture(0)

padding=20
    
#since we have the video
while True :

    #we will capture all the video frames (frame-by-frame)
    ret , frame = video.read()

    print(frame.shape)
    #Call the facebox function
    results = model(frame)
    print(results)
    blob = cv2.dnn.blobFromImage(np.squeeze(results.render()), 1.0, (227,227),MODEL_MEAN_VALUES, swapRB=False)

    print("this is the result")
    print(np.squeeze(results.render()))
    
    ageNet.setInput(blob)
    agePred= ageNet.forward()
    age=ageList[agePred[0].argmax()]
    print("this is the age")
    print(age)
    
    print(frame)
    cv2.putText(frame,age,(100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    #we need to visualize the frames
    cv2.imshow("Age-Gender",frame)
    #waitKey(1) will display a frame for 1 ms, after which display will be automatically closed
    k=cv2.waitKey(1)

    #when we press q el we leave the loop and close the windows
    if k == ord('q'):
        break
   
    

    
    

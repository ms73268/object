import cv2

tres = 0.5
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
font = cv2.FONT_HERSHEY_COMPLEX

classNames = []
with open('C:/Users/ms73268/Downloads/ICMS/object_detection/coco.names') as f:
    classNames = f.read().rstrip('\n').split('\n')

# print(classNames)

configpath = 'C:/Users/ms73268/Downloads/ICMS/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightpath = 'C:/Users/ms73268/Downloads/ICMS/object_detection/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightpath, configpath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

# while True:
#     ret, frame = cap.read()
#     classIds , conf, bbox = net.detect(frame, confThreshold = tres)
#     for Id, confidence, box in zip(classIds.flatten(),conf.flatten(),bbox):
#         cv2.rectangle(frame,box,(0,255,0),2)
#         cv2.putText(frame,
#                     classNames[Id-1].upper(),
#                     (box[0]+10,box[1]+30),
#                     font,1,
#                     (0,0,255),
#                     2,
#                     cv2.LINE_4)
        
#         # cv2.putText(frame,
#         #             str(round(confidence*100,2)),
#         #             (box[0]+300,box[1]+30),
#         #             font,1,
#         #             (0,255,0),
#         #             2,
#         #             cv2.LINE_4)
        
#     cv2.imshow('Screen',frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
while True:
    ret, frame = cap.read()
    classIds , conf, bbox = net.detect(frame, confThreshold = tres)
    if len(classIds) != 0:
        for Id, confidence, box in zip(classIds.flatten(),conf.flatten(),bbox):
            print(f"Detected object: {classNames[Id-1]}")
            cv2.rectangle(frame, box, (0, 255, 0), 2)
            cv2.putText(frame, classNames[Id-1].upper(), (box[0]+10, box[1]+30),
                        font, 1, (0, 0, 255), 2, cv2.LINE_4)
            
    cv2.imshow('Screen', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

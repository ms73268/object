import cv2


def cornerRect(img, bbox, l=30, t=5, rt=1,
               colorR=(255, 0, 255), colorC=(0, 255, 0)):
    """
    :param img: Image to draw on.
    :param bbox: Bounding box [x, y, w, h]
    :param l: length of the corner line
    :param t: thickness of the corner line
    :param rt: thickness of the rectangle
    :param colorR: Color of the Rectangle
    :param colorC: Color of the Corners
    :return:
    """
    x, y, w, h = bbox
    x1, y1 = x + w, y + h
    if rt != 0:
        cv2.rectangle(img, bbox, colorR, rt)
    # Top Left  x,y
    cv2.line(img, (x, y), (x + l, y), colorC, t)
    cv2.line(img, (x, y), (x, y + l), colorC, t)
    # Top Right  x1,y
    cv2.line(img, (x1, y), (x1 - l, y), colorC, t)
    cv2.line(img, (x1, y), (x1, y + l), colorC, t)
    # Bottom Left  x,y1
    cv2.line(img, (x, y1), (x + l, y1), colorC, t)
    cv2.line(img, (x, y1), (x, y1 - l), colorC, t)
    # Bottom Right  x1,y1
    cv2.line(img, (x1, y1), (x1 - l, y1), colorC, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), colorC, t)

    return img


def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    """
    Creates Text with Rectangle Background
    :param img: Image to put text rect on
    :param text: Text inside the rect
    :param pos: Starting position of the rect x1,y1
    :param scale: Scale of the text
    :param thickness: Thickness of the text
    :param colorT: Color of the Text
    :param colorR: Color of the Rectangle
    :param font: Font used. Must be cv2.FONT....
    :param offset: Clearance around the text
    :param border: Outline around the rect
    :param colorB: Color of the outline
    :return: image, rect (x1,y1,x2,y2)
    """
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]




tres = 0.7
cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)
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

while True:
    ret, frame = cap.read()
    try:
        classIds , conf, bbox = net.detect(frame, confThreshold = tres)
        for Id, confidence, box in zip(classIds.flatten(),conf.flatten(),bbox):
            cornerRect(frame,  # The image to draw on
            box,  # The position and dimensions of the rectangle (x, y, width, height)
            l=30,  # Length of the corner edges
            t=5,  # Thickness of the corner edges
            rt=1,  # Thickness of the rectangle
            colorR=(255, 0, 255),  # Color of the rectangle
            colorC=(0, 255, 0)  # Color of the corner edges
            )
            frame, bbox = putTextRect(
            frame, classNames[Id-1].upper(), (box[0]+10,box[1]+30),  # Image and starting position of the rectangle
            scale=2, thickness=2,  # Font scale and thickness
            colorT=(255, 255, 255), colorR=(255, 0, 255),  # Text color and Rectangle color
            font=cv2.FONT_HERSHEY_PLAIN,  # Font type
            offset=10,  # Offset of text inside the rectangle
            border=3, colorB=(0, 255, 0)  # Border thickness and color
        )
    except Exception as e:
        print(e)  
    cv2.imshow('Screen',frame)

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
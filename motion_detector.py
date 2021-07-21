import cv2, time, pandas as pd
from datetime import datetime

first_frame = None #holds the initial image captured by the camera
status_list = [None, None]
times = [None, None] #holds the start/ end datetime movement was detected (created with null values to avoid index out of range error)
df = pd.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)

while True:
    #capture first frame, convert to greyscale and blur it
    check, frame = video.read()
    status = 0 #holds value to determine whether movement is detected: 0=no, 1=yes
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (21, 21), 0)

    if first_frame is None:
        first_frame = grey
        continue
    
    #calculate difference between first fram and current frame
    delta_frame = cv2.absdiff(first_frame, grey)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #detect movement by detecting any contours larger than 10000 pixels    
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    status_list.append(status)
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Grey Frame", grey)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Colour Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        #get time program ended if movement is currently detected
        if status == 1:
            times.append(datetime.now())
        break

    print(status)

for i in range(0, len(times), 2):
    df = df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()
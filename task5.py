import cv2
import numpy as np 
import cv2.aruco as aruco
import math
from decimal import Decimal

#CAMERA PARAMETERS
intrinsic = np.array([[975.28024031, 0,           542.32692791],
                      [ 0,           978.7789663, 333.58254409],
                      [ 0,           0,           1]], dtype = np.double)  #
                    


distortion = np.array([-5.88821114e-02, 1.51122291e+00, -9.58190989e-03, 2.62779503e-03, -6.17695602e+00], dtype = np.double) 


cap = cv2.VideoCapture(1)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) #we will detect aruco tags in this particular dictionary only
parameters = aruco.DetectorParameters_create()
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 3
side = 0.175

destination = cv2.imread("destination.jpg") #top-view of aruco tag taken as reference
ptsDest = np.array([[447,354], [780,352], [793,695], [441,693]]) #corner points of the tag in the reference image.

while(True):
    ret, frame = cap.read()
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)
    bnw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(bnw, aruco_dict, parameters = parameters) #corners of aruco tag detected along with their ids

    for corner in corners:
        H, _ = cv2.findHomography(corner, ptsDest) #generating the homography matrix which 
        #                                           maps the corner points from webcam image 
        #                                           to corner points in reference image
        topView = cv2.warpPerspective(frame, H, (destination.shape[1], destination.shape[0])) #generating the top-view of the webcam frame
        frame = cv2.line(frame, (corner[0][0][0], corner[0][0][1]), (corner[0][1][0], corner[0][1][1]), (0,255,0), 6) # Draw the
        frame = cv2.line(frame, (corner[0][1][0], corner[0][1][1]), (corner[0][2][0], corner[0][2][1]), (0,255,0), 6) # aruco tag
        frame = cv2.line(frame, (corner[0][2][0], corner[0][2][1]), (corner[0][3][0], corner[0][3][1]), (0,255,0), 6) # by joining
        frame = cv2.line(frame, (corner[0][3][0], corner[0][3][1]), (corner[0][0][0], corner[0][0][1]), (0,255,0), 6) # the corner points
        centerx = int(0.25*(corner[0][0][0]+corner[0][1][0]+corner[0][2][0]+corner[0][3][0]))  # find the centroid
        centery = int(0.25*(corner[0][0][1]+corner[0][1][1]+corner[0][2][1]+corner[0][3][1]))  # of the detected aruco tag
        rvec,tvec, garbage = aruco.estimatePoseSingleMarkers(corner, side, intrinsic, distortion) #estimate the pose
        frame = aruco.drawAxis(frame, intrinsic, distortion,rvec, tvec, 0.08)
        x = round(Decimal(tvec[0][0][0]*100),2)  # converting
        y = round(Decimal(tvec[0][0][1]*100),2)  # position of the tag
        z = round(Decimal(tvec[0][0][2]*100),2)  # from metre to cm rounded off to two decimal places
        D = round(Decimal(math.sqrt((x**2) +(y**2) +(z**2))),2) #calculate distance of tag-centre from webcam
        R = cv2.Rodrigues(rvec)[0] #convert the rodrigues vector rvec into rotation matrix
        
        alpha = math.atan2(R[1,0],R[0,0])*180/math.pi
        beta = math.atan2(-R[2,0],(math.sqrt(R[2,1]**2 + R[2,2]**2)))*180/math.pi
        gamma = math.atan2(R[2,1], R[2,2])*180/math.pi
        if gamma>0 :
            gamma = round(Decimal(180 - gamma),2)
        elif gamma<0:
            gamma = round(Decimal(180+gamma),2)
        alpha = round(Decimal(alpha),2)
        beta = round(Decimal(beta),2)

        roll = alpha
        yaw = beta
        pitch = gamma
        frame = cv2.putText(frame, "Roll: "+str(roll)+"deg", (10,40), font, fontScale, (255,0,0), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "Pitch: "+str(pitch)+"deg", (10,90), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "Yaw: "+str(yaw)+"deg", (10,140), font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, str(D)+"cm", (centerx, centery), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "X: "+str(x)+"cm", (10,440), font, fontScale, (0,0,255), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "Y: "+str(y)+"cm", (10,490), font, fontScale, (0,255,0), thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, "Z: "+str(z)+"cm", (10,540), font, fontScale, (255,0,0), thickness, cv2.LINE_AA)
        cv2.imshow("topView", topView)
    cv2.imshow("webcam", frame)
    cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
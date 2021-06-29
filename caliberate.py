import cv2
import numpy as np 
import glob

grid = (9,7)

crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 25, 0.01) #criteria for accuracy of the found corners

objpts = []
imgpts = []

objp = np.zeros((1, grid[0] * grid[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)  #coordinates of corners, taking top left corner as origin



imgs = glob.glob('./images/*.jpg')
SHAPE = (0,0)
for file in imgs:
    image = cv2.imread(file)
    bnw = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    SHAPE = bnw.shape[::-1]
    ret, corners = cv2.findChessboardCorners(bnw, grid, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    #finds the corner coordiantes in the image plane

    if ret == True:
        objpts.append(objp)
        _corners = cv2.cornerSubPix(bnw, corners, (10,10), (-1,-1), crit) #increases the accuracy of the detected corners
        imgpts.append(_corners)
        image = cv2.drawChessboardCorners(image, grid, _corners, ret) #draw the chessboard corners
        
    cv2.imshow('image', image)
    cv2.waitKey(5)

cv2.destroyAllWindows()

ret, intrinsic, dist, rotation, translation = cv2.calibrateCamera(objpts, imgpts, SHAPE, None, None) #camera caliberation


print("Distortion coeffs:\n")
print(dist)
print("Intrinsic matrix:\n")
print(intrinsic)
    


        
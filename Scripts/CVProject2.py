import numpy as np
import cv2
from matplotlib import pyplot as plt



def EdgeCornerDetection(minTH, maxTH, blur=False, block=2, k=0.04):
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if blur:
            frame = cv2.GaussianBlur(frame,(5,5),3)
        #Edge Detection
        edges = cv2.Canny(frame,minTH,maxTH)
        
        #Corner Detection
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,block,3,k)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        #Convert edge to RGB so we can use red dots for corner.
        edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
        # Threshold for an optimal value, it may vary depending on the image.
        edges[dst>0.01*dst.max()]=[0,0,255]
        
        cv2.imshow('Original', frame)
        cv2.imshow('Edges and Corners',edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def SIFT(blur=False):
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if blur:
            frame = cv2.GaussianBlur(frame,(5,5),3)
        # use grey to perform SIFT
        gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
        kp = sift.detect(gray,None)
        
        img=cv2.drawKeypoints(frame,kp,frame,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
        cv2.imshow('SIFT KEYPOINTS',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
   
def harrisMatching():
    img1 = cv2.imread('chair1.png') # queryImage
    img2 = cv2.imread('chair2.png') # trainImage
    
    #Detect Harris corners for img1
    gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    a = dst>0.01*dst.max()
    x,y = np.where(a==True)
    
    #Convert to keypoint object
    eye_corner_cordinates = []
    for i in np.arange(len(x)):
        eye_corner_cordinates.append([x[i],y[i]])
        
    kp1= [cv2.KeyPoint(crd[0], crd[1], 3) for crd in eye_corner_cordinates]

    # compute SIFT descriptors from corner keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    #des1 = [sift.compute(img1,[kp])[1] for kp in kp1]
    (kp1, des1) = sift.compute(img1, kp1)
    
    #Detect Harris Corner for img2
    gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    a = dst>0.01*dst.max()
    x,y = np.where(a==True)
    #Convert to keypoint object
    eye_corner_cordinates = []
    for i in np.arange(len(x)):
        eye_corner_cordinates.append([x[i],y[i]])
        
    kp2= [cv2.KeyPoint(crd[0], crd[1], 3) for crd in eye_corner_cordinates]

    # compute SIFT descriptors from corner keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    #des2 = [sift.compute(img2,[kp])[1] for kp in kp2]
    (kp2, des2) = sift.compute(img2, kp2)
    
    ds1=[]
    ds2=[]
    for i in np.arange(len(des1)):
        ds1.append(des1[i][0])
    for i in np.arange(len(des2)):
        ds2.append(des2[i][0])
    ds1 = np.array(ds1)
    ds2 = np.array(ds2)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(ds1,ds2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        
        good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2, outImg= img1)
    cv2.imwrite('harrisMatching1.png',img3)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3),plt.show()


    
def imgMatching():
    
    img1 = cv2.imread('door1.png') # queryImage
    img2 = cv2.imread('door2.png') # trainImage
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=1000)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2, outImg= img1)
    cv2.imwrite('siftMatching3.png',img3)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    plt.imshow(img3),plt.show()

   
    


if __name__ == "__main__":
    #EdgeCornerDetection(50,100,False, block=6)
    #SIFT()
    harrisMatching()
    #imgMatching()
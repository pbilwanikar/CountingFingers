import cv2 as cv
import HandTrackingModule as htm
import time
import os

cap = cv.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
print(myList)

overlayList = []
for imgPath in myList:
    image = cv.imread(f'{folderPath}/{imgPath}')
    overlayList.append(image)

print(len(overlayList))

pTime =0
cTime = 0
detector = htm.handDetector(detectionCon=0.9)

tipIds = [4, 8, 12, 16, 20]

while True:
    sucess, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList)!=0:
        fingers =[]

        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)

        else:
            fingers.append(0)


        #For 4 fingers

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] -2][2]:
                fingers.append(1)

            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        print(totalFingers)

        h, w, c = overlayList[totalFingers-1].shape

        img[0:h, 0:w] = overlayList[totalFingers-1]
        count = totalFingers


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, f"FPS: {str(int(fps))}", (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv.putText(img, f"Count: {str(int(count))}", (10, 600), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)



    cv.imshow("Finger Counter", img)
    cv.waitKey(1)





















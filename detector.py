from KalmanFilter import KalmanFilter
import numpy as np
import cv2


kf = KalmanFilter(0.1, 1, 1, 100, 0.1,0.1) #process noise 100

centers = []
for i in range(0, 600, 50):
    centers.append([[i], [int(i**2/1000)]])

print(centers)

img = cv2.imread("green_bgr.jpeg")
for pt in centers:
    pt_tuple = (pt[0][0], pt[1][0])
    print(pt_tuple)
    cv2.circle(img, pt_tuple, 15, (0, 20, 220), -1)
    (x,y) = kf.predict()
    print(x,y)
    predicted = kf.update(pt)
    print("Predicted")
    
    predicted_tuple = (int(predicted[0][0]), int(predicted[1][0])) 
    print(predicted_tuple)
    cv2.circle(img, predicted_tuple, 15, (220, 20, 0), 4)

for i in range(10):
    final_pred = kf.predict()
    predicted_tuple = (int(final_pred[0][0]), int(final_pred[1][0]))
    cv2.circle(img, predicted_tuple, 15, (220, 20, 0), 4)

cv2.imshow("Image", img)
cv2.waitKey(0)
    





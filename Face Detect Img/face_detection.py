import cv2
trained_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('multi.jpg')

gray_col = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)

faces =trained_data.detectMultiScale(gray_col)
# print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 254), 2)
cv2.imshow('Console',img)
# cv2.imshow('Gray',gray_col)
cv2.waitKey()

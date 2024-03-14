import cv2

##############################################################################################
FaceCascades = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
frameWidth = 640
frameHeight = 480
minArea = 500
color = (255,0,255)
##############################################################################################
cap = cv2.VideoCapture(0)


cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 130)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    Persons = FaceCascades.detectMultiScale(imgGray,1.1,4)
    personCounter = 1
    
    for (x, y, w, h) in Persons:
        Area = w * h
        if Area > minArea:
            # Dibuja el rectángulo alrededor del rostro y agrega el texto
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2) 
            cv2.putText(img, f"Person {personCounter}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            personCounter += 1  # Incrementa el contador para la siguiente persona
            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow("Img Roi", imgRoi)
    
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

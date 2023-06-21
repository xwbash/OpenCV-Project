import cv2
import os
 

def ReadAndSaveFaceDataFromVideo(videoPath, personName, maxDataCount = 30):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(videoPath)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=5, minSize=(30, 30))
            i = 0
            for (x, y, w, h) in faces:
                i += 1
                face = frame[y:y+h, x:x+w]

                cv2.putText(frame, "face", (x, y + y+h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                currentDirectory = os.getcwd()
                targetDirectory = currentDirectory + "\\Data\\FaceDatas\\Persons\\" + personName

                if os.path.exists(targetDirectory) == False:
                    os.makedirs(targetDirectory, True)
                
                cv2.imwrite(f"{targetDirectory}\\face_{frame_count}.jpg", face)

                if i > 1:
                    break

                frame_count += 1

            cv2.imshow("Video", frame)
        else:
            break

        if cv2.waitKey(1) == ord("q") or frame_count > maxDataCount:
            break

    cap.release()
    cv2.destroyAllWindows()


def ReadAndSaveFaceDataFromCamera(personName, maxDataCount = 30):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                currentDirectory = os.getcwd()
                targetDirectory = currentDirectory + "\\Data\\FaceDatas\\Persons\\" + personName

                if os.path.exists(targetDirectory) == False:
                    os.mkdir(targetDirectory)
                
                cv2.imwrite(f"{targetDirectory}\\face_{frame_count}.jpg", face)
                frame_count += 1

            cv2.imshow("Video", frame)
        else:
            print("Camera Error! ")
            break

        if cv2.waitKey(1) == ord("q") or frame_count > maxDataCount:
            break

    cap.release()
    cv2.destroyAllWindows()



def MainFunction():

    isCameraOrFromVideo = int(input("""
    #1 - Camera Input
    #2 - Video Input
    
    Is camera or video input? : """))

    dataName = input("Data Name : ")
    maxDataCount = int(input("Max Data Count (default 20) : "))

    if isCameraOrFromVideo == 1:
        ReadAndSaveFaceDataFromCamera(dataName, maxDataCount)
    else:
        ReadAndSaveFaceDataFromVideo("Data/ExampleVideo/traning.mp4", dataName, maxDataCount)

    

MainFunction()

#opencv folder
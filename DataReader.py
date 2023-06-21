import cv2
import os
import numpy as np


def ReadDataFromCamera():

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            i = 0
            for (x, y, w, h) in faces:
                i += 1
                cv2.rectangle(frame, (x, y), (w+x, y+h), (0, 255, 0), 2)
                if(i > 1):
                    break

            cv2.imshow("Video", frame)
        else:
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def ReadDataFromVideo(videoPath):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(videoPath)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (w+x, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "face", (x, y + y+h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


            cv2.imshow("Video", frame)
        else:
            break

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def DetectPart(image, template, person):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    threshold = 0.9

    loc = np.where(result >= threshold)

    template_height, template_width = template.shape[:2]

    i = 0
    for pt in zip(*loc[::-1]):
        i += 1
        if i > 1:
            break
        top_left = pt
        bottom_right = (pt[0] + template_width, pt[1] + template_height)

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        text_position = (top_left[0], bottom_right[1] + 20)
        cv2.putText(image, person, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


    return image

def CaptureTheCamera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            currentDirectory = os.getcwd()
            positive_folders = currentDirectory + "\\Data\\FaceDatas\\Persons"
            
            for folders in os.listdir(positive_folders):
                positiveImages = os.listdir(folders)
            
                for i in positiveImages:
                    template = cv2.imread(positiveImages+"\\"+folders+"\\"+i)
            
            result = DetectPart(frame, template)

        cv2.imshow('Result', result)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

def CaptureTheVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    currentDirectory = os.getcwd()
    positive_folders = currentDirectory + "\\Data\\FaceDatas\\Persons"
    
    for folders in os.listdir(positive_folders):
        positiveImages = os.listdir(positive_folders + "\\" + folders)
    
        for i in positiveImages:
            template = cv2.imread(positive_folders + "\\" + folders + "\\" + i)
    
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = DetectPart(frame, template, folders)
                cv2.imshow('Result', result)

                if cv2.waitKey(1) == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


def main():
    isCameraOrFromVideo = int(input("""
    #1 - Video Input
    #2 - Camera Input
    
    Is camera or video input? : """))

    isFacesOrSaved = int(input("""
    #1 - Saved Data
    #2 - Faces 
    
    Is saved data or just faces? : """))

    if isFacesOrSaved == 1:
        if isCameraOrFromVideo == 1:
            videoName = input("Enter the video name : ")
            CaptureTheVideo(f"Data/ExampleVideo/{videoName}.mp4")
        else:
            CaptureTheCamera()
    else:
        if isCameraOrFromVideo == 1:
            videoName = input("Enter the video name : ")
            ReadDataFromVideo(f"Data/ExampleVideo/{videoName}.mp4")
        else:
            ReadDataFromCamera()

    

if __name__ == '__main__':
    main()






#MainFunction()
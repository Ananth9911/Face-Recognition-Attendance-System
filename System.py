#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tkinter as tk

from tkinter import Message ,Text

import cv2,os

import shutil

import csv

import numpy as np

from PIL import Image, ImageTk

import pandas as pd

import datetime

import time

import tkinter.ttk as ttk

import tkinter.font as font



 











# In[30]:


window = tk.Toplevel()

#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')

window.title("Face_Recogniser")


# In[31]:


dialog_title = 'QUIT'

dialog_text = 'Are you sure?'

#answer = messagebox.askquestion(dialog_title, dialog_text)


# In[32]:


image = Image.open('e.jpeg')
photo_image = ImageTk.PhotoImage(image)
label = tk.Label(window, image = photo_image)
label.pack()


# In[33]:


message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System" , fg="White",bg="Black" ,width=50  ,height=2,font=('times', 30, 'italic bold underline')) 



message.place(x=100, y=20)



lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="red"  ,bg="yellow" ,font=('times', 15, ' bold ') ) 

lbl.place(x=200, y=150)



txt = tk.Entry(window,width=20  ,bg="yellow" ,fg="red",font=('times', 15, ' bold '))

txt.place(x=500, y=170)



lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="red"  ,bg="yellow"    ,height=2 ,font=('times', 15, ' bold ')) 

lbl2.place(x=200, y=250)



txt2 = tk.Entry(window,width=20  ,bg="yellow"  ,fg="red",font=('times', 15, ' bold ')  )

txt2.place(x=500, y=270)



lbl5 = tk.Label(window, text="Enter Subject",width=20  ,fg="red"  ,bg="yellow"    ,height=2 ,font=('times', 15, ' bold ')) 

lbl5.place(x=200, y=350)


txt5 = tk.Entry(window,width=20  ,bg="yellow"  ,fg="red",font=('times', 15, ' bold ')  )

txt5.place(x=500, y=370)




lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold underline ')) 

lbl3.place(x=200, y=450)



message = tk.Label(window, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 

message.place(x=500, y=450)



lbl3 = tk.Label(window, text="Attendance : ",width=20  ,fg="red"  ,bg="yellow"  ,height=2 ,font=('times', 15, ' bold  underline')) 

lbl3.place(x=300, y=650)





message2 = tk.Label(window, text="" ,fg="red"   ,bg="yellow",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 

message2.place(x=600, y=650)


# In[34]:


def clear():

    txt.delete(0, 'end')    

    res = ""

    message.configure(text= res)


# In[35]:


def clear2():

    txt2.delete(0, 'end')    

    res = ""

    message.configure(text= res)    


# In[36]:


def clear3():

    txt2.delete(0, 'end')    

    res = ""

    message.configure(text= res)  


# In[37]:


def is_number(s):

    try:

        float(s)

        return True

    except ValueError:

        pass

 

    try:

        import unicodedata

        unicodedata.numeric(s)

        return True

    except (TypeError, ValueError):

        pass

 

    return False


# In[38]:


def TakeImages():        

    Id=(txt.get())

    name=(txt2.get())
    
    subject=(txt5.get())

    if(is_number(Id) and name.isalpha() and subject.isalpha()):

        cam = cv2.VideoCapture(0)

        harcascadePath = "haarcascade_frontalface_default.xml"

        detector=cv2.CascadeClassifier(harcascadePath)

        sampleNum=0

        while(True):

            ret, img = cam.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:

                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        

                #incrementing sample number 

                sampleNum=sampleNum+1

                #saving the captured face in the dataset folder TrainingImage

                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

                #display the frame

                cv2.imshow('frame',img)

            #wait for 100 miliseconds 

            if cv2.waitKey(100) & 0xFF == ord('q'):

                break

            # break if the sample number is morethan 100

            elif sampleNum>60:

                break

        cam.release()

        cv2.destroyAllWindows() 

        res = "Images Saved for ID : " + Id +" Name : "+ name+" Subject : "+subject

        row = [Id , name, subject]

        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:

            writer = csv.writer(csvFile)

            writer.writerow(row)

        csvFile.close()

        message.configure(text= res)

    else:

        if(is_number(Id)):

            res = "Enter Alphabetical Name"

            message.configure(text= res)

        if(name.isalpha()):

            res = "Enter Numeric Id"

            message.configure(text= res)
        
        if(subject.isalpha()):

            res = "Enter Numeric Id"

            message.configure(text= res)


# In[39]:


def TrainImages():

    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()

    harcascadePath = "haarcascade_frontalface_default.xml"

    detector =cv2.CascadeClassifier(harcascadePath)

    faces,Id = getImagesAndLabels("TrainingImage")

    recognizer.train(faces, np.array(Id))

    recognizer.save("TrainingImageLabel\Trainner.yml")

    res = "Image Trained"#+",".join(str(f) for f in Id)

    message.configure(text= res)


# In[40]:


def getImagesAndLabels(path):

    #get the path of all the files in the folder

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 

    #print(imagePaths)

    

    #create empth face list

    faces=[]

    #create empty ID list

    Ids=[]

    #now looping through all the image paths and loading the Ids and the images

    for imagePath in imagePaths:

        #loading the image and converting it to gray scale

        pilImage=Image.open(imagePath).convert('L')

        #Now we are converting the PIL image into numpy array

        imageNp=np.array(pilImage,'uint8')

        #getting the Id from the image

        Id=int(os.path.split(imagePath)[-1].split(".")[1])

        # extract the face from the training image sample

        faces.append(imageNp)

        Ids.append(Id)        

    return faces,Ids


# In[41]:


def TrackImages():

    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()

    recognizer.read("TrainingImageLabel\Trainner.yml")

    harcascadePath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(harcascadePath);    

    df=pd.read_csv("StudentDetails\StudentDetails.csv")

    cam = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX        

    col_names =  ['Id','Name','Subject',' Date','Time']

    attendance = pd.DataFrame(columns = col_names)    

    while True:

        ret, im =cam.read()

        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        faces=faceCascade.detectMultiScale(gray, 1.2,5)    

        for(x,y,w,h) in faces:

            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)

            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   

            if(conf < 50):

                ts = time.time()      

                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                aa=df.loc[df['Id'] == Id]['Name'].values
                bb=df.loc[df['Id'] == Id]['Subject'].values

                tt=str(Id)+"-"+aa+"-"+bb

                attendance.loc[len(attendance)] = [Id,aa,bb,date,timeStamp]

                

            else:

                Id='Unknown'                

                tt=str(Id)  

            if(conf > 75):

                noOfFile=len(os.listdir("ImagesUnknown"))+1

                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            

            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        

        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    

        cv2.imshow('im',im) 

        if (cv2.waitKey(1)==ord('q')):

            break

    ts = time.time()      

    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

    Hour,Minute,Second=timeStamp.split(":")

    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+"_"+".csv"

    attendance.to_csv(fileName,index=False)

    cam.release()

    cv2.destroyAllWindows()

    #print(attendance)

    res=attendance

    message2.configure(text= res)


# In[42]:


clearButton = tk.Button(window, text="Clear", command=clear  ,fg="red"  ,bg="yellow"  ,width=20  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))

clearButton.place(x=800, y=140)

clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="red"  ,bg="yellow"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))

clearButton2.place(x=800, y=260)  

clearButton3 = tk.Button(window, text="Clear", command=clear3  ,fg="red"  ,bg="yellow"  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))

clearButton3.place(x=800, y=350)


takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))

takeImg.place(x=80, y=550)

trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))

trainImg.place(x=400, y=550)

trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))

trackImg.place(x=700, y=550)

quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))

quitWindow.place(x=1000, y=550)

window.mainloop()


# In[48]:


import numpy as np
a=[[ 5, 1 ,3], [ 1, 1 ,1],[ 1, 2 ,1]]
b=[1, 2, 3]
c=np.dot(a,b,axis=1)
print(c)


# In[ ]:





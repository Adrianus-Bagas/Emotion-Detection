import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import librosa.display
import os
from playsound import playsound

window = Tk()
window.title("Emotion Detection")
window.geometry("300x300")

img_good = ImageTk.PhotoImage(Image.open("happiness.png"))
img_bad = ImageTk.PhotoImage(Image.open("angry.png"))
img_fear = ImageTk.PhotoImage(Image.open("emoji.png"))
img_sad = ImageTk.PhotoImage(Image.open("sad.png"))

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    result = np.array([])
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
    
    result = np.array(result)
    
    return result

def file():
    currdir = os.getcwd()
    tempdir = filedialog.askopenfilename(parent=window, initialdir=currdir, title='Pilih audio')
    inputtxt.insert(END,tempdir)

classes = { 
    0:'Marah',
    1:'Takut',
    2:'Senang',
    3:'Sedih' 
}

def deteksi():
    path = inputtxt.get(1.0, "end-1c")
    model = load_model('model emotion.h5')
    X__ = []
    feature = get_features(path)
    for ele in feature:
        X__.append(ele)
    X__Test = np.array(X__).reshape(1,-1)
    test_input = np.expand_dims(X__Test, axis=2)
    pred_test__ = model.predict(test_input)
    predicted_label=np.argmax(pred_test__,axis=1)
    outputtxt.insert(END,classes[predicted_label[0]]) 

def clear():
    inputtxt.delete("1.0", "end")
    outputtxt.delete("1.0", "end")

def play():
    path = inputtxt.get(1.0, "end-1c")
    playsound(path)

label_good = Label(window, image = img_good, height=50, width=50)
label_good.place(x=10,y=50)

label_bad = Label(window, image = img_bad, height=50, width=50)
label_bad.place(x=240,y=50)

label_fear = Label(window, image = img_fear, height=50, width=50)
label_fear.place(x=10,y=200)

label_sad = Label(window, image = img_sad, height=50, width=50)
label_sad.place(x=240,y=200)

label1 = tk.Label(window, text = "Masukkan audio",fg="black")
label1.pack()

b2=Button(text="Load", width=6,command=file)
b2.pack()

inputtxt = tk.Text(window,
                   height = 2,
                   width = 10)
  
inputtxt.pack()

b3=Button(text="Play", width=6,command=play)
b3.pack()

b0=Button(text="Cek", width=6,command=deteksi)
b0.pack()

label2 = tk.Label(window, text = "Apa jenis emosi ini ?",fg="black")
label2.pack()

outputtxt = tk.Text(window,
                   height = 2,
                   width = 10)
  
outputtxt.pack()

b1=Button(text="Reset", width=6,command=clear)
b1.pack()

window.mainloop()
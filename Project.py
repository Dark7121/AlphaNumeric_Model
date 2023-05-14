import tkinter as tk
from tkinter import *
import PIL.Image
import PIL.ImageTk
import numpy as np
import cv2
import tensorflow as tf

# load the saved model
model = tf.keras.models.load_model('D:/my_model.keras')

# create a window
window = tk.Tk()
window.geometry("500x500")

# add a canvas
canvas_width = 300
canvas_height = 300
canvas = Canvas(window, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

# add a button
def predict():
    # take a screenshot of the canvas
    x = window.winfo_rootx() + canvas.winfo_x()
    y = window.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas_width
    y1 = y + canvas_height
    img = PIL.ImageGrab.grab((x, y, x1, y1))
    img = img.resize((28, 28), PIL.Image.ANTIALIAS)
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img/255.0

    # predict the digit or alphabet
    prediction = model.predict(img)
    output_label.config(text="Prediction: {}".format(chr(np.argmax(prediction)+55) if np.argmax(prediction)>=10 else np.argmax(prediction)))

predict_button = tk.Button(window, text="Predict", command=predict)
predict_button.pack()

# add an output label
output_label = tk.Label(window, text="")
output_label.pack()

# function to draw on the canvas
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)

canvas.bind("<B1-Motion>", paint)

window.mainloop()

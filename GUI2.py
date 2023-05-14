import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# load the saved model
model = tf.keras.models.load_model('D:/my_model.keras')

# create a dictionary that maps the prediction index to the corresponding label
labels_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
               10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
               20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
               30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
               40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n',
               50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x',
               60: 'y', 61: 'z'}

# create a canvas that allows the user to draw a digit or a letter
class DrawingCanvas(tk.Canvas):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.old_coords = None
        self.draw_width = 20
        self.draw_color = 'black'
        self.bind_events()
    
    # bind mouse events to canvas methods
    def bind_events(self):
        self.bind('<B1-Motion>', self.draw)
        self.bind('<ButtonRelease-1>', self.reset_coords)
        
    # get the current mouse coordinates and draw a line to the previous coordinates
    def draw(self, event):
        if self.old_coords:
            x1, y1 = self.old_coords
            x2, y2 = event.x, event.y
            self.create_line(x1, y1, x2, y2, width=self.draw_width, fill=self.draw_color,
                             capstyle=tk.ROUND, smooth=True)
        self.old_coords = event.x, event.y
    
    # reset the previous coordinates to None
    def reset_coords(self, event):
        self.old_coords = None
    
    # clear the canvas
    def clear(self):
        self.delete('all')
    
    # convert the canvas image to a numpy array and resize it to 28x28 pixels
   
    def get_image_array(self):
        # get the current size of the canvas
        x, y, x2, y2 = self.bbox('all')
        w = x2 - x
        h = y2 - y
        
        # create a new image with a white background
        image = Image.new('RGB', (w, h), (255, 255, 255))
        
        # draw the canvas onto the image
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, w, h), fill=(255, 255, 255))
        draw.line((0, 0, w, h), fill=self.draw_color, width=self.draw_width)
        self.update()
        image = image.resize((28, 28))
        
        # convert the image to a numpy array and normalize its values
        image_array = np.array(image)
        image_array = image_array[:, :, 0] / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
    
    # predict the digit or letter drawn on the canvas
    def predict(self):
        image_array = self.get_image_array()
        prediction = model.predict(image_array)
        prediction_index = np.argmax(prediction)
        predicted_label = labels_dict[prediction_index]
        self.master.update_prediction_label(predicted_label)
    
# create a GUI for the drawing canvas
class DrawingApp(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.create_widgets()
    
    # create the drawing canvas and the prediction label
    def create_widgets(self):
        self.canvas = DrawingCanvas(self, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.prediction_label = tk.Label(self, text='Draw a digit or a letter', font=('Arial', 20))
        self.prediction_label.grid(row=0, column=1, padx=10, pady=10)
        self.clear_button = tk.Button(self, text='Clear', command=self.canvas.clear)
        self.clear_button.grid(row=1, column=0, padx=10, pady=10)
        self.predict_button = tk.Button(self, text='Predict', command=self.canvas.predict)
        self.predict_button.grid(row=1, column=1, padx=10, pady=10)
    
    # update the prediction label with the predicted digit or letter
    def update_prediction_label(self, predicted_label):
        self.prediction_label.config(text=f'Predicted: {predicted_label}')
    
# create the main window and start the event loop
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Handwritten Digit and Letter Recognition')
    app = DrawingApp(root)
    app.pack()
    root.mainloop()

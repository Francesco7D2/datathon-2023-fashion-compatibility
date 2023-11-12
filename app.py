import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pickle
import pandas as pd

from model import Predictor


class Frame1(tk.Frame):
    def __init__(self, parent, frame2):
        super().__init__(parent)
        self.parent = parent
        self.frame_next = frame2
        
        self.image = ImageTk.PhotoImage(Image.open("./Frame_1.png"))
        self.image_lbl = tk.Label(self, image=self.image)
        self.image_lbl.pack(fill="both", expand=True)
        
        parent.bind("<space>", self.go_out)
        
    def go_out(self, *args):
        self.parent.unbind("<space>")
        self.pack_forget()
        self.frame_next.pack(fill="both", expand=True)
        
            
class Frame2(tk.Frame):
    def __init__(self, parent, frame3=None):
        super().__init__(parent)
        self.frame_next = frame3
        
        self.image = ImageTk.PhotoImage(Image.open("./Frame_2.png"))
        self.image_lbl = tk.Label(self, image=self.image)
        self.image_lbl.pack(fill="both", expand=True)
        
        self.image_lbl.bind("<Button-1>", self.select_cloth)
        
        
    def select_cloth(self, event):
        x, y = event.x, event.y
        
        if(562 < x < 797 and 402 < y < 731):
            self.go_out()
        
        # print(event.x, event.y)
        
        
    def go_out(self, *args):
        self.pack_forget()
        self.frame_next.pack(fill="both", expand=True)


class Frame3(tk.Frame):
    def __init__(self, parent, predictor, dataset):
        super().__init__(parent)
        self.dataset = dataset
        
        self.image = ImageTk.PhotoImage(Image.open("./Frame_3.png").resize((parent.winfo_screenwidth(), parent.winfo_screenheight())))
        self.image_lbl = tk.Label(self, image=self.image)
        self.image_lbl.pack()
        
        self.big_image_id = ""
        self.big_image_path = ""
        self.big_image = ""
        self.big_image_lbl = tk.Label(self)
        self.big_image_lbl.place(x=91, y=26)
        self.put_big_image()
        
        self.predictor = predictor
        self.imgs_paths = [[]]
        self.imgs = [[]]
        self.lbl_imgs = []
        self.create_imgs()
        
        self.index = 0
        self.pos = [(876, 24), (1218,23), (876,302), (1218,302), (876,579), (1218,579)]
        self.put_imatges()
        
        self.image_lbl.bind("<Button-1>", self.next_outfit)
        self.big_image_lbl.bind("<Button-1>", self.next_outfit)
        
        
    def next_outfit(self, event):
        print(event.x, event.y)
        x, y = event.x, event.y
        if 341 < x < 513 and 740 < y < 760:
            self.delete_images()
            self.put_imatges()
        elif 100 < x < 600 and 100 < y < 600:
            print(222)
            self.destroy_big_image()
            self.put_big_image()
            self.delete_images()
            self.create_imgs()
            self.put_imatges()
        pass
    
    def create_imgs(self):
        self.imgs_paths = self.predictor.predict(self.big_image_id, 5, 10)
        self.imgs = [[ImageTk.PhotoImage(Image.open("./../" + i).resize((200, 200))) for i in list_[1:]] for list_ in self.imgs_paths]
        
    def put_imatges(self):
        for i in range(len(self.imgs[self.index])):
            l = tk.Label(self, image=self.imgs[self.index][i])
            l.place(x=self.pos[i][0], y=self.pos[i][1])
            self.lbl_imgs.append(l)
        if self.index < 10:
            self.index += 1
        
    def delete_images(self):
        for l in self.lbl_imgs:
            l.destroy()
        self.lbl_imgs.clear()
        
    def put_big_image(self):
        row = self.dataset.sample(n=1)
        self.big_image_id = row["cod_modelo_color"].iloc[0]
        self.big_image_path = row["des_filename"].iloc[0]
        self.big_image = ImageTk.PhotoImage(Image.open("./../" + self.big_image_path).resize((650, 650)))
        self.big_image_lbl["image"] = self.big_image
        
    def destroy_big_image(self):
        self.index = 0
        
    def go_out(self, *args):
        self.pack_forget()
        self.frame_next.pack(fill="both", expand=True)



if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    
    classifier = ""
    with open("modelRFR.pkl", "rb") as f:
        classifier = pickle.load(f)
    predictor = Predictor("./../datathon/dataset/product_data.csv", classifier)
    dataset = pd.read_csv("./../datathon/dataset/product_data.csv")
    
    f3 = Frame3(root, predictor, dataset)
    f2 = Frame2(root, f3)
    f1 = Frame1(root, f2)
    
    f1.pack(fill="both", expand=True)
    
    root.mainloop()
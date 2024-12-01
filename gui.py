import tkinter as tk
from tkinter import filedialog
import os

from train import train

class GANTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GAN Trainer")

        self.file_path = tk.StringVar()
        self.learning_rate = tk.DoubleVar(value=1e-3)
        self.batch_size = tk.IntVar(value=16)
        self.noise_dim = tk.IntVar(value=100)
        self.num_epochs = tk.IntVar(value=100)
        self.features_disc = tk.IntVar(value=64)
        self.features_gen = tk.IntVar(value=64)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Matlab File:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.file_path, width=50).grid(row=0, column=1)
        tk.Button(self.root, text="Browse", command=self.browse_file).grid(row=0, column=2)

        tk.Label(self.root, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.learning_rate).grid(row=1, column=1)

        tk.Label(self.root, text="Batch Size:").grid(row=2, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.batch_size).grid(row=2, column=1)

        tk.Label(self.root, text="Noise Dimension:").grid(row=3, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.noise_dim).grid(row=3, column=1)

        tk.Label(self.root, text="Number of Epochs:").grid(row=4, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.num_epochs).grid(row=4, column=1)

        tk.Label(self.root, text="Features Disc:").grid(row=5, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.features_disc).grid(row=5, column=1)

        tk.Label(self.root, text="Features Gen:").grid(row=6, column=0, sticky=tk.W)
        tk.Entry(self.root, textvariable=self.features_gen).grid(row=6, column=1)

        tk.Button(self.root, text="Start Training", command=self.start_training).grid(row=7, column=0, columnspan=3)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("MATLAB files", "*.mat")])
        self.file_path.set(file_path)

    def start_training(self):
        file_path = self.file_path.get()
        learning_rate = self.learning_rate.get()
        batch_size = self.batch_size.get()
        noise_dim = self.noise_dim.get()
        num_epochs = self.num_epochs.get()
        features_disc = self.features_disc.get()
        features_gen = self.features_gen.get()

        if not os.path.exists(file_path):
            print("File not found!")
            return

        train(file_path, learning_rate, batch_size, noise_dim, num_epochs, features_disc, features_gen)

if __name__ == "__main__":
    root = tk.Tk()
    app = GANTrainerApp(root)
    root.mainloop()

import os
import sys
import ctypes
import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
loma_path = os.path.join(parent, "loma_public")
sys.path.append(loma_path)
import compiler

class Polygon:
    def __init__(self) -> None:
        self.num_vertices = 0
        self.vertices = []
        self.color = np.array([0, 0, 0])

    def render(self, image) -> None:
        draw = ImageDraw.Draw(image)
        vertices = [(x, y) for x, y in self.vertices]
        draw.polygon(vertices, fill=tuple(self.color.astype(int)))

    def generate(self, num_vertices=3, height=170, width=169) -> None:
        # to generate one triangle randomly
        self.num_vertices = num_vertices
        self.vertices = []
        for i in range(self.num_vertices):
            x = random.uniform(0, height)
            y = random.uniform(0, width)
            self.vertices.append(np.array([x, y]))
        self.color[0] = random.uniform(0, 255)
        self.color[1] = random.uniform(0, 255)
        self.color[2] = random.uniform(0, 255)

class Picture:
    def __init__(self, height=170, width=169) -> None:
        self.height = height
        self.width = width
        self.polygons = []
        self.image = Image.new("RGB", (width, height), "black")

    def init_with_file(self, filename) -> None:
        self.image = Image.open(filename)

    def generate(self, count=2) -> None:
        self.polygons = []
        for i in range(count):
            tri = Polygon()
            tri.generate()
            self.polygons.append(tri)
            tri.render(self.image)

    def show(self) -> None:
        self.image.show()
    
    def get_ctype_array(self):
        np_array = np.array(self.image, dtype=np.float32)
        type3 = ctypes.POINTER(ctypes.c_float)
        type2 = ctypes.POINTER(type3)
        arr = (type2 * self.height)()
        for i in range(self.height):
            arr[i] = (type3 * self.width)()
            for j in range(self.width):
                arr[i][j] = (ctypes.c_float * 3)()
                for k in range(3):
                    arr[i][j][k] = np_array[i][j][k]
        return arr
    
    def get_loss_trad(self, pic2):
        # get l2 loss of self and pic2
        assert self.height == pic2.height and self.width == pic2.width
        np1 = np.array(self.image, dtype=np.float32)
        np2 = np.array(pic2.image, dtype=np.float32)
        return np.sqrt(np.sum((np1 - np2) ** 2))
        
    def get_loss_loma(self, pic2):
        # get l2 loss of self and pic2
        assert self.height == pic2.height and self.width == pic2.width
        arr1 = self.get_ctype_array()
        arr2 = pic2.get_ctype_array()
        return lib.l2_loss(arr1, arr2, self.height, self.width, 3)

image = Picture()
image.generate()
target = Picture()
target.init_with_file("./target.png")

# image.show()
# target.show()

with open('./l2_loss.py') as f:
    structs, lib = compiler.compile(f.read(),
                                target = 'c',
                                output_filename = '_code/l2_loss')

print(image.get_loss_trad(target))
print(image.get_loss_loma(target))

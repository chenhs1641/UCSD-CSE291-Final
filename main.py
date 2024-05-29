
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
    def __init__(self):
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

image = Image.new("RGB", (169, 170), "black")
for i in range(2):
    tri = Polygon()
    tri.generate()
    tri.render(image)

# image.show()

filename = "./target.png"
target = Image.open(filename)
# target.show()

np1 = np.array(image, dtype=np.float32)
np2 = np.array(target, dtype=np.float32)

l2_loss_ = np.sqrt(np.sum((np1 - np2) ** 2))
print(l2_loss_)

with open('./l2_loss.py') as f:
    structs, lib = compiler.compile(f.read(),
                                target = 'c',
                                output_filename = '_code/l2_loss')

type3 = ctypes.POINTER(ctypes.c_float)
type2 = ctypes.POINTER(type3)
type1 = ctypes.POINTER(type2)

arr1 = (type2 * 170)()
arr2 = (type2 * 170)()
for i in range(170):
    arr1[i] = (type3 * 169)()
    arr2[i] = (type3 * 169)()
    for j in range(169):
        arr1[i][j] = (ctypes.c_float * 3)()
        arr2[i][j] = (ctypes.c_float * 3)()
        for k in range(3):
            arr1[i][j][k] = np1[i][j][k]
            arr2[i][j][k] = np2[i][j][k]

print(lib.l2_loss(arr1, arr2, 170, 169, 3))

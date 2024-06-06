import numpy as np
import sys
import os
import random
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)
from main import Picture

test_target = Picture()
test_target.generate(10, 3)
test_target.save("./_image/test_target.png")

test_img = Picture()
test_img.polygons = test_target.polygons

for polygon in test_img.polygons:
    for i in range(3):
        polygon.vertices[i][0] += random.uniform(-8, 8)
        polygon.vertices[i][1] += random.uniform(-8, 8)
    for i in range(3):
        polygon.color[i] += random.uniform(-8, 8)

test_img.optimization(test_target, num_iter=20, lr=1, save_output=True)
test_img.save("./_image/test_img.png")

from subprocess import call
call(["ffmpeg", "-framerate", "10", "-i",
    "./_image/iter_%d.png", "-vb", "20M",
    "./_image/out.mp4"])
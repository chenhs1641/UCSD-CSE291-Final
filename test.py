import numpy as np
import sys
import os
import random
current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)
from main import Picture

test_target = Picture()
test_target.generate(1, 3)
test_target.save("./_image/test_target.png")

test_img = Picture()
test_img.polygons.append(test_target.polygons[0])

for i in range(3):
    test_img.polygons[0].vertices[i][0] += random.uniform(-8, 8)
    test_img.polygons[0].vertices[i][1] += random.uniform(-8, 8)

for i in range(3):
    test_img.polygons[0].color[i] += random.uniform(-2, 2)

test_img.polygons[0].render(test_img.image)
test_img.optimization(test_target, num_iter=20, lr=1, save_video=True)
test_img.save("./_image/test_img.png")

from subprocess import call
call(["ffmpeg", "-framerate", "10", "-i",
    "./_image/iter_%d.png", "-vb", "20M",
    "./_image/out.mp4"])
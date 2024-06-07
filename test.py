import numpy as np
import sys
import os
import random
from datetime import datetime

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)
from main import Picture

def get_timestamp_string():
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp_str

random_time_stamp = get_timestamp_string()
os.mkdir(f"./_image_{random_time_stamp}")

test_target = Picture()
test_target.generate(10, 3)
# test_target.generate(1, 4)
# test_target.generate(1, 5)
test_target.save(f"./_image_{random_time_stamp}/test_target.png")

test_img = Picture()
test_img.polygons = test_target.polygons

for polygon in test_img.polygons:
    for i in range(3):
        polygon.vertices[i][0] += random.uniform(-8, 8)
        polygon.vertices[i][1] += random.uniform(-8, 8)
    for i in range(3):
        polygon.color[i] += random.uniform(-8, 8)

test_img.polygons[0].render(test_img.image)
test_img.optimization(test_target, num_iter=100, lr=1, save_output=True, random_time_stamp=random_time_stamp)
test_img.save(f"./_image_{random_time_stamp}/test_img.png")

from subprocess import call
call(["ffmpeg", "-framerate", "10", "-i",
    "./_image/iter_%d.png", "-vb", "20M",
    "./_image/out.mp4"])
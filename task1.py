import numpy as np
import sys
import os
import random
import copy
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
os.mkdir(f"./_image_{random_time_stamp}/l2")
os.mkdir(f"./_image_{random_time_stamp}/pyr")

test_target = Picture()
test_target.generate(5, 3)
test_target.save(f"./_image_{random_time_stamp}/l2/test_target.png")
test_target.save(f"./_image_{random_time_stamp}/pyr/test_target.png")

test_img = Picture()
test_img.polygons = test_target.polygons

for polygon in test_img.polygons:
    for i in range(3):
        polygon.vertices[i][0] += random.uniform(-12, 12)
        polygon.vertices[i][1] += random.uniform(-12, 12)
    for i in range(3):
        polygon.color[i] += random.uniform(-8, 8)

test_img_2 = copy.deepcopy(test_img)
test_img.polygons[0].render(test_img.image)
test_img.optimization(test_target, num_iter=500, lr=0.05, save_output=True, random_time_stamp=random_time_stamp + "/l2", use_loss="l2")
test_img.optimization(test_target, num_iter=500, lr=0.1, save_output=True, random_time_stamp=random_time_stamp + "/pyr")
test_img.save(f"./_image_{random_time_stamp}/l2/test_img.png")
test_img.save(f"./_image_{random_time_stamp}/pyr/test_img.png")
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

test_target = Picture()
test_target.generate(1, 10)
# test_target.generate(1, 4)
# test_target.generate(1, 5)
test_target.save(f"./_image_{random_time_stamp}/test_target.png")

test_img = Picture()
test_img.generate(1, 4)

        
test_img.polygons[0].vertices = np.array([[1.0, 1.0], [1.0, 63.0], [63.0, 63.0],[63.0, 1.0]])
test_img.polygons[0].color = test_target.polygons[0].color

test_img.polygons[0].render(test_img.image)

for grad in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    os.mkdir(f"./_image_{random_time_stamp}/{grad}")
    test_img_2 = copy.deepcopy(test_img)
    test_img_2.optimization(test_target, num_iter=500, lr=0.2, save_output=True, random_time_stamp=random_time_stamp+f"/{grad}", shear_strenth=0.03, merge_grad_thres=grad)
    test_img_2.save(f"./_image_{random_time_stamp}/{grad}/test_img.png")
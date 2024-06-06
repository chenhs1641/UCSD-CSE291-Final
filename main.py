import os
import sys
import ctypes
import random

from PIL import Image, ImageDraw
import numpy as np
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
loma_path = os.path.join(parent, "loma_public")
sys.path.append(loma_path)
from matplotlib.path import Path
import compiler
from datetime import datetime
import matplotlib.pyplot as plt
from subprocess import call

def get_timestamp_string():
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp_str

# with open('./l2_loss.py') as f:
#     structs, lib = compiler.compile(f.read(),
#                                 target = 'c',
#                                 output_filename = '_code/l2_loss')
    
    
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        updated_params = []
        for i, param in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            param -= update
            updated_params.append(param)
        
        return updated_params

class Polygon:
    def __init__(self) -> None:
        self.num_vertices = 0
        self.vertices = []
        self.color = np.array([0.0, 0.0, 0.0])
        self.perimeter = 0.0
        self.path = None
        self.dcolor = None
        self.dvertices = None
        # The smaller, the deeper
        self.order = 0.0
        self.dorder = 0.0
        
        self.adam_optimizer = None
        
    def zero_grad(self):
        self.dvertices = np.zeros_like(self.vertices)
        self.dcolor = np.zeros_like(self.color)
        self.dorder = 0.0

    def render(self, image) -> None:
        draw = ImageDraw.Draw(image)
        vertices = [(x, y) for x, y in self.vertices]
        fill_color = tuple(self.color.astype(np.uint8))
        draw.polygon(vertices, fill=fill_color)

    def generate(self, num_vertices=3, height=170, width=169) -> None:
        # Generate one triangle randomly
        self.num_vertices = num_vertices
        x_coords = np.random.uniform(0, height, self.num_vertices)
        y_coords = np.random.uniform(0, width, self.num_vertices)
        self.vertices = np.column_stack((x_coords, y_coords))
        # Matplotlib path uses different coordinate system!
        self.path = Path(self.vertices)
        self.color = np.random.uniform(0, 255, 3)
        self.order = 0.0
        self.perimeter = np.sum(np.linalg.norm(self.vertices - np.roll(self.vertices, 1, axis=0), axis=1))
        
        self.adam_optimizer = AdamOptimizer([self.vertices, self.color])
        self.order_optimizer = AdamOptimizer([self.order])
        
    def check_hit(self, x, y):
        # No need to transform x and y
        return self.path.contains_point((x, y))
    
    def update(self):
        if self.adam_optimizer is None:
            raise ValueError("Optimizer not initialized. Call generate() first.")
        
        grads = [self.dvertices, self.dcolor]
        updated_params = self.adam_optimizer.step(grads)
        self.vertices, self.color = updated_params
        
        grads = [self.dorder]
        self.order = self.order_optimizer.step(grads)[0]
        
        self.path = Path(self.vertices)
        self.perimeter = np.sum(np.linalg.norm(self.vertices - np.roll(self.vertices, 1, axis=0), axis=1))

class Picture:
    def __init__(self, height=170, width=169, bg_color=np.array([0.0, 0.0, 0.0])) -> None:
        self.height = height
        self.width = width
        self.bg_color = bg_color
        # store from the bottom
        self.polygons = []
        self.image = Image.new("RGB", (width, height), tuple(bg_color.astype(np.uint8)))

    def init_with_file(self, filename) -> None:
        self.image = Image.open(filename)
        self.width, self.height = self.image.size

    def generate(self, count=2, num_vertices=3) -> None:
        self.polygons = []
        for _ in range(count):
            shape = Polygon()
            shape.generate(num_vertices=num_vertices, height=self.height, width=self.width)
            self.polygons.append(shape)
            shape.render(self.image)

    def show(self) -> None:
        self.image.show()

    def save(self, filename):
        self.image.save(filename)
        
    def render(self):
        self.image = Image.new('RGB', (self.width, self.height), tuple(self.bg_color.astype(np.uint8)))
        # Sort polygons by order from the bottom to the top
        self.polygons.sort(key=lambda x: x.order)
        
        # The order is from the bottom to the top
        for p in self.polygons:
            p.render(self.image)
    
    # def get_ctype_array(self):
    #     np_array = np.array(self.image, dtype=np.float32)
    #     type3 = ctypes.POINTER(ctypes.c_float)
    #     type2 = ctypes.POINTER(type3)
    #     arr = (type2 * self.height)()
    #     for i in range(self.height):
    #         arr[i] = (type3 * self.width)()
    #         for j in range(self.width):
    #             arr[i][j] = (ctypes.c_float * 3)()
    #             for k in range(3):
    #                 arr[i][j][k] = np_array[i][j][k]
    #     return arr
    
    def get_ctype_array(self):
        np_array = np.array(self.image, dtype=np.float32)
        
        # 定义 ctypes 类型
        Float3 = ctypes.c_float * 3
        Float3Ptr = ctypes.POINTER(Float3)
        
        # 创建 ctypes 数组
        arr = (Float3Ptr * self.height)()
        
        # 填充 ctypes 数组
        for i in range(self.height):
            row = (Float3 * self.width)()
            for j in range(self.width):
                row[j][:] = np_array[i, j, :]
            arr[i] = row
        
        return arr
    
    # def get_ctype_d_array(self):
    #     type3 = ctypes.POINTER(ctypes.c_float)
    #     type2 = ctypes.POINTER(type3)
    #     arr = (type2 * self.height)()
    #     for i in range(self.height):
    #         arr[i] = (type3 * self.width)()
    #         for j in range(self.width):
    #             arr[i][j] = (ctypes.c_float * 3)()
    #             for k in range(3):
    #                 arr[i][j][k] = ctypes.c_float()
    #     return arr
    
    def get_ctype_d_array(self):
        # 定义 ctypes 类型
        Float3 = ctypes.c_float * 3
        Float3Ptr = ctypes.POINTER(Float3)
        
        # 创建 ctypes 数组
        arr = (Float3Ptr * self.height)()
        
        # 填充 ctypes 数组
        for i in range(self.height):
            row = (Float3 * self.width)()
            for j in range(self.width):
                row[j] = Float3()  # 初始化为零的 float3 数组
            arr[i] = row
        
        return arr
    
    def get_loss_trad(self, pic2):
        # Get l2 loss of self and pic2
        np1 = np.array(self.image, dtype=np.float32)
        np2 = np.array(pic2.image, dtype=np.float32)
        loss = np.sum((np1 - np2) ** 2) / (self.height * self.width * 3)
        return loss
    
    def get_dimage_trad(self, pic2):
        np1 = np.array(self.image, dtype=np.float32)
        np2 = np.array(pic2.image, dtype=np.float32)
        self.dimage = 2 * (np1 - np2) / (self.height * self.width * 3)
        return self.dimage
        
    def get_loss_loma(self, pic2):
        # Get l2 loss of self and pic2
        arr1 = self.get_ctype_array()
        arr2 = pic2.get_ctype_array()
        loss = lib.l2_loss(arr1, arr2, self.height, self.width, 3)
        return loss
    
    # def get_dimage_loma(self, pic2):
    #     arr1 = self.get_ctype_array()
    #     arr2 = pic2.get_ctype_array()
    #     d_arr1 = self.get_ctype_d_array()
    #     d_arr2 = self.get_ctype_d_array()
    #     d_height = ctypes.POINTER(ctypes.c_int)()
    #     d_width = ctypes.POINTER(ctypes.c_int)()
    #     d_color = ctypes.POINTER(ctypes.c_int)()
    #     d_loss = np.zeros((self.height, self.width, 3))
    #     lib.d_l2_loss(arr1, d_arr1, arr2, d_arr2, self.height, d_height, self.width, d_width, 3, d_color, 1.0)
    #     for i in range(self.height):
    #         for j in range(self.width):
    #             for k in range(3):
    #                 d_loss[i][j][k] = d_arr1[i][j][k]
    #     return d_loss
    
    def get_dimage_loma(self, pic2):
        arr1 = self.get_ctype_array()
        arr2 = pic2.get_ctype_array()
        d_arr1 = self.get_ctype_d_array()
        d_arr2 = self.get_ctype_d_array()
        d_height = ctypes.pointer(ctypes.c_int())
        d_width = ctypes.pointer(ctypes.c_int())
        d_color = ctypes.pointer(ctypes.c_int())
        d_loss = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # 调用外部函数
        lib.d_l2_loss(arr1, d_arr1, arr2, d_arr2, self.height, d_height, self.width, d_width, 3, d_color, 1.0)

        # 将 d_arr1 转换为 numpy 数组
        d_arr1_np = np.ctypeslib.as_array(d_arr1, shape=(self.height, self.width, 3))

        # 直接复制数据
        np.copyto(d_loss, d_arr1_np)

        return d_loss
    
    def raytrace(self, x, y, ret_all=False):
        hit_list = []
        # Check from the top
        for polygon in reversed(self.polygons):
            if polygon.check_hit(x, y):
                hit_list.append(polygon)
                if not ret_all:
                    break
        return hit_list

    def zero_grad(self):
        for p in self.polygons:
            p.zero_grad()
            
    def update(self, lr=0.1):
        for p in self.polygons:
            p.update(lr)
    
    def optimization(self, pic2, num_iter = 100, interier_samples_per_pixel = 4, edge_samples_per_pixel = 1, order_samples_per_pixel = 1, lr = 0.1, order_lr = 0.01, edge_sampling_error = 0.05, save_output = False, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        loss_record = []
        random_time_stamp = get_timestamp_string()
        
        # set Adam optimizer for each polygon
        for p in self.polygons:
            p.adam_optimizer = AdamOptimizer([p.vertices, p.color], lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
            p.order_optimizer = AdamOptimizer([p.order], lr=order_lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        
        for iter in range(num_iter):
            print("Iteration: ", iter)
            self.render()
            # already sorted by order
            if save_output:
                self.image.save(f"./image_{random_time_stamp}/iter_{iter}.png")
            
            # Only for my verify, it should be computed by loma
            loss = self.get_loss_loma(pic2)
            print("Loss: ", loss)
            loss_record.append(loss)
            # Only for my verify, it should be computed by loma
            dimage = self.get_dimage_loma(pic2)
            self.zero_grad()
            
            # Interier sampling
            for j in range(self.height):
                for i in range(self.width):
                    for _ in range(interier_samples_per_pixel):
                        x = i + random.random()
                        y = j + random.random()
                        hits = self.raytrace(x, y)
                        if len(hits) == 0:
                            continue
                        hit = hits[0]
                        hit.dcolor += dimage[j, i] / interier_samples_per_pixel
                        
            total_perimeter = np.sum([p.perimeter for p in self.polygons])
            weight = 1.0 / total_perimeter / edge_samples_per_pixel
                        
            # Edge sampling
            # Do not sample randomly, but sample uniformly
            for prim in self.polygons:
                vertices = np.vstack([prim.vertices, prim.vertices[0]])
                dvertices = np.zeros_like(vertices)
                in_color = prim.color
                for i in range(prim.num_vertices):
                    length = np.linalg.norm(vertices[i + 1] - vertices[i])
                    dir = (vertices[i + 1] - vertices[i]) / length
                    vertical_dir = np.array([-dir[1], dir[0]])
                    
                    # vertical_dir should point outside
                    test_point = (vertices[i + 1] + vertices[i]) / 2 + edge_sampling_error * vertical_dir
                    hits = self.raytrace(test_point[0], test_point[1])
                    if len(hits) > 0 and hits[0] == prim:
                        vertical_dir = -vertical_dir
                        
                    num_samples = int(length * edge_samples_per_pixel + 0.5)
                    # n+1 intervals, n samples, the two endpoints aren't included
                    num_intervals = num_samples + 1
                    
                    for j in range(num_samples):
                        portion = (j + 1) / num_intervals
                        pos = vertices[i] + portion * (vertices[i + 1] - vertices[i])
                        
                        int_x, int_y = int(pos[0]), int(pos[1])
                        if int_x < 0 or int_x >= self.width or int_y < 0 or int_y >= self.height:
                            continue
                        out_pos = pos + edge_sampling_error * vertical_dir
                        out_hits = self.raytrace(out_pos[0], out_pos[1])
                        
                        if len(out_hits) == 0:
                            out_color = self.bg_color
                        else:
                            out_color = out_hits[0].color
                        
                        dI_d0x = (in_color - out_color) * (1 - portion) * vertical_dir[0] * weight
                        dI_d0y = (in_color - out_color) * (1 - portion) * vertical_dir[1] * weight
                        dI_d1x = (in_color - out_color) * portion * vertical_dir[0] * weight
                        dI_d1y = (in_color - out_color) * portion * vertical_dir[1] * weight
                    
                        dvertices[i][0] += np.dot(dimage[int_y][int_x], dI_d0x)
                        dvertices[i][1] += np.dot(dimage[int_y][int_x], dI_d0y)
                        dvertices[i + 1][0] += np.dot(dimage[int_y][int_x], dI_d1x)
                        dvertices[i + 1][1] += np.dot(dimage[int_y][int_x], dI_d1y)
                
                dvertices[0] += dvertices[-1]
                prim.dvertices = dvertices[:-1]
                
            # Order sampling
            for j in range(self.height):
                for i in range(self.width):
                    
                    for _ in range(order_samples_per_pixel):
                        dx = random.random()
                        dy = random.random()
                        hit_list = self.raytrace(i + dx, j + dy, ret_all=True)

                        if len(hit_list) <= 1:
                            continue
                        dp_p = np.random.normal(loc=0, scale=1, size=len(hit_list))
                        samples = dp_p + np.array([p.order for p in hit_list])
                        top = hit_list[np.argmax(samples)]
                        pixel_color = top.color
                        
                        pixel_loss = np.sum((pixel_color - dimage[j, i]) ** 2)
                        
                        top.dorder += -pixel_loss * dp_p[np.argmax(samples)] / order_samples_per_pixel
                
            self.update()
            
            
            
            
        if save_output:
            # Draw the loss curve
            plt.clf()
            plt.plot(range(num_iter), loss_record, linestyle='-')
            plt.title('Loss over iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            #plt.grid(True)
            plt.savefig(f"./image_{random_time_stamp}/loss_curve.png")
            
            # Save the final output image
            self.render()
            self.image.save(f"./image_{random_time_stamp}/output.png")
            
            # Generate video
            call(["ffmpeg", "-framerate", "10", "-i",
                "./image_"+random_time_stamp+"/iter_%d.png", "-vb", "20M",
                "./image_"+random_time_stamp+"/output.mp4"])
        
if __name__ == "__main__":
    image = Picture()
    image.generate(30)
    target = Picture()
    target.init_with_file("./target.png")

    # image.optimization(target, num_iter=10, lr=10)

    image.show()
    target.show()

    print(image.get_loss_trad(target) - image.get_loss_loma(target))
    print(image.get_dimage_trad(target) - image.get_dimage_loma(target))

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

class Polygon:
    def __init__(self) -> None:
        self.num_vertices = 0
        self.vertices = []
        self.color = np.array([0.0, 0.0, 0.0])
        self.perimeter = 0.0
        self.path = None
        self.dcolor = None
        self.dvertices = None
        
    def zero_grad(self):
        self.dvertices = np.zeros_like(self.vertices)
        self.dcolor = np.zeros_like(self.color)

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
        self.perimeter = np.sum(np.linalg.norm(self.vertices - np.roll(self.vertices, 1, axis=0), axis=1))
        
    def check_hit(self, x, y):
        # No need to transform x and y
        return self.path.contains_point((x, y))
    
    def update(self, lr=0.1):
        self.vertices -= lr * self.dvertices
        self.color -= lr * self.dcolor
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
            tri = Polygon()
            tri.generate(num_vertices=num_vertices, height=self.height, width=self.width)
            self.polygons.append(tri)
            tri.render(self.image)

    def show(self) -> None:
        self.image.show()

    def save(self, filename):
        self.image.save(filename)
        
    def render(self):
        self.image = Image.new('RGB', (self.width, self.height), tuple(self.bg_color.astype(np.uint8)))
        for p in self.polygons:
            p.render(self.image)
    
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
        # Get l2 loss of self and pic2
        np1 = np.array(self.image, dtype=np.float32)
        np2 = np.array(pic2.image, dtype=np.float32)
        loss = np.sqrt(np.sum((np1 - np2) ** 2))
        return loss
    
    def get_dimage_trad(self, pic2):
        np1 = np.array(self.image, dtype=np.float32)
        np2 = np.array(pic2.image, dtype=np.float32)
        self.dimage = (np1 - np2) / np.sqrt(np.sum((np1 - np2) ** 2))
        return self.dimage
        
    def get_loss_loma(self, pic2):
        # Get l2 loss of self and pic2
        arr1 = self.get_ctype_array()
        arr2 = pic2.get_ctype_array()
        loss = lib.l2_loss(arr1, arr2, self.height, self.width, 3)
        return loss
    
    # TBD: 
    def get_dimage_loma(self, pic2):
        pass
    
    def raytrace(self, x, y):
        hit_list = []
        # Check from the top
        for polygon in reversed(self.polygons):
            if polygon.check_hit(x, y):
                hit_list.append(polygon)
        return hit_list

    def zero_grad(self):
        for p in self.polygons:
            p.zero_grad()
            
    def update(self, lr=0.1):
        for p in self.polygons:
            p.update(lr)
    
    def optimization(self, pic2, num_iter = 100, interier_samples_per_pixel = 4, edge_samples_per_pixel = 1, lr = 0.1, edge_sampling_error = 0.05, save_video = False):
        
        loss_record = []
        
        for iter in range(num_iter):
            print("Iteration: ", iter)
            self.render()
            # TBD: Save the latest image
            
            # Only for my verify, it should be computed by loma
            loss = self.get_loss_trad(pic2)
            print("Loss: ", loss)
            loss_record.append(loss)
            # Only for my verify, it should be computed by loma
            dimage = self.get_dimage_trad(pic2)
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
                # It should be deepcopy here
                prim.dvertices = dvertices[:-1]
                
            self.update(lr=lr)

            if save_video:
                self.image.save(f"./_image/iter_{iter}.png")
            
        # TBD: Draw the loss curve and save it
        
        # TBD: Save the final output image
        
        # TBD: Use the images to generate a video
        
if __name__ == "__main__":
    image = Picture()
    image.generate(30)
    target = Picture()
    target.init_with_file("./target.png")

    image.optimization(target, num_iter=10, lr=10)

    image.show()
    target.show()

    # with open('./l2_loss.py') as f:
    #     structs, lib = compiler.compile(f.read(),
    #                                 target = 'c',
    #                                 output_filename = '_code/l2_loss')

    print(image.get_loss_trad(target))

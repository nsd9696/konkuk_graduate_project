import numpy as np
from PIL import Image, ImageDraw
import os
import sys
import argparse

def set_image(image_path, size_mult=1):
    # load image
    image = Image.open(image_path)
    image = image.resize((image.size[0] * size_mult, image.size[1] * size_mult))
    return image



def average_colour(image):
    # convert image to np array
    image_arr = np.asarray(image)

    # get average of whole image
    avg_color_per_row = np.average(image_arr, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))


def weighted_average(hist):
    total = sum(hist)
    error = value = 0

    if total > 0:
        value = sum(i * x for i, x in enumerate(hist)) / total
        error = sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total
        error = error ** 0.5

    return error


def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1140

    return detail_intensity


class Quadrant():
    def __init__(self, image, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        # crop image to quadrant size
        image = image.crop(bbox)
        hist = image.histogram()

        self.detail = get_detail(hist)
        self.colour = average_colour(image)

    def split_quadrant(self, image):
        left, top, width, height = self.bbox

        # get the middle coords of bbox
        middle_x = left + (width - left) / 2
        middle_y = top + (height - top) / 2

        # split root quadrant into 4 new quadrants
        upper_left = Quadrant(image, (left, top, middle_x, middle_y), self.depth + 1)
        upper_right = Quadrant(image, (middle_x, top, width, middle_y), self.depth + 1)
        bottom_left = Quadrant(image, (left, middle_y, middle_x, height), self.depth + 1)
        bottom_right = Quadrant(image, (middle_x, middle_y, width, height), self.depth + 1)

        # add new quadrants to root children
        self.children = [upper_left, upper_right, bottom_left, bottom_right]


class QuadTree():
    def __init__(self, image, detail_threshold=13, max_depth = 8):
        self.width, self.height = image.size
        self.detail_threshold = detail_threshold
        self.max_depth = max_depth

        # keep track of max depth achieved by recursion
        self.track_depth = 0

        # start compression
        self.start(image)

    def start(self, image):
        # create initial root
        self.root = Quadrant(image, image.getbbox(), 0)

        # build quadtree
        self.build(self.root, image)

    def build(self, root, image):
        if root.depth >= self.max_depth or root.detail <= self.detail_threshold:
            if root.depth > self.track_depth:
                self.track_depth = root.depth

            # assign quadrant to leaf and stop recursing
            root.leaf = True
            return

            # split quadrant if there is too much detail
        root.split_quadrant(image)

        for children in root.children:
            self.build(children, image)

    def create_image(self, custom_depth, show_lines=False):
        # create blank image canvas
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.width, self.height), (0, 0, 0))

        leaf_quadrants = self.get_leaf_quadrants(custom_depth)

        # draw rectangle size of quadrant for each leaf quadrant
        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.colour, outline=(0, 0, 0))
            else:
                draw.rectangle(quadrant.bbox, quadrant.colour)

        return image, leaf_quadrants

    def get_leaf_quadrants(self, depth):
        if depth > self.track_depth:
            raise ValueError('A depth larger than the trees depth was given')

        quandrants = []

        # search recursively down the quadtree
        self.recursive_search(self, self.root, depth, quandrants.append)

        return quandrants

    def recursive_search(self, tree, quadrant, max_depth, append_leaf):
        # append if quadrant is a leaf
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)

        # otherwise keep recursing
        elif quadrant.children != None:
            for child in quadrant.children:
                self.recursive_search(tree, child, max_depth, append_leaf)

    def create_gif(self, file_name, duration=1000, loop=0, show_lines=False):
        gif = []
        end_product_image, temp = self.create_image(self.track_depth, show_lines=show_lines)

        for i in range(self.track_depth):
            image, temp = self.create_image(i, show_lines=show_lines)
            gif.append(image)

        # add extra images at end
        for _ in range(4):
            gif.append(end_product_image)

        gif[0].save(
            file_name,
            save_all=True,
            append_images=gif[1:],
            duration=duration, loop=loop)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path',type=str,required=True,help='input image path')
    parser.add_argument('--size_mult',type=int,required=False,default=1,help='size_mult')
    parser.add_argument('--max_depth',type=int,required=False,default=8,help='max_depth')
    parser.add_argument('--target_depth',type=int,required=False,default=6,help='target_depth')
    parser.add_argument('--detail_threshold',type=int,required=False,default=13,help='detail_threshold')
    args = parser.parse_args()
    image_path = args.img_path
    size_mult = args.size_mult
    max_depth = args.max_depth
    target_depth = args.target_depth
    detail_threshold = args.detail_threshold

    image_name = image_path.split('/')[-1].split('.')[0]
    save_path = './thanosed_image'
    image = set_image(image_path, int(size_mult))
    quadtree = QuadTree(image,int(detail_threshold),int(max_depth))
    image,temp = quadtree.create_image(int(target_depth),show_lines=False)
    isdir = os.path.isdir(save_path)
    if isdir == True:
        image.save(f"{save_path}/{image_name}.jpg")
    else:
        os.makedirs(save_path)
        image.save(f"{save_path}/{image_name}.jpg")

if __name__ == '__main__':
    main()
    print("image_quadtree")
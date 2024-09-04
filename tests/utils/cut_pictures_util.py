import os

import numpy as np
from PIL import Image


def cut_pictures_in_folder(base_path: str, crop_range: tuple[int,int,int,int], save_target_folder: str):
    if not os.path.exists(save_target_folder):
        os.makedirs(save_target_folder)
    for file_name in os.listdir(base_path):
        file_path = os.path.join(base_path, file_name)
        image = Image.open(file_path)
        croped = image.crop(crop_range)
        target_file_path = os.path.join(save_target_folder, os.path.basename(file_path))
        croped.save(str(target_file_path))


def del_white_of_pictures_in_folder(base_path: str, save_target_folder: str):
    """
    删除图片中的白边
    """
    if not os.path.exists(save_target_folder):
        os.makedirs(save_target_folder)
    for file_name in os.listdir(base_path):
        if not file_name.endswith('.png'):
            continue
        file_path = os.path.join(base_path, file_name)
        image = Image.open(file_path)
        region = get_not_white_region(image)
        print(f'{file_name} region', region, image.height, image.width)
        croped = image.crop(region)
        target_file_path = os.path.join(save_target_folder, os.path.basename(file_path))
        croped.save(str(target_file_path))


def get_not_white_region(image):
    # 从外向内，找出第一个非白行，作为要裁剪的区域
    np_image = np.array(image)
    height = np_image.shape[0]
    width = np_image.shape[1]

    left = 0
    for i in range(0, width):
        left = i
        all_white = True
        for j in range(0, height):
            if np_image[j, left, 0] != 255 or np_image[j, left, 1] != 255 or np_image[j, left, 2] != 255:
                all_white = False
                break
        if not all_white:
            break

    right = width - 1
    for i in range(0, width):
        right = width - 1 - i
        all_white = True
        for j in range(0, height):
            if np_image[j, right, 0] != 255 or np_image[j, right, 1] != 255 or np_image[j, right, 2] != 255:
                all_white = False
                break
        if not all_white:
            break

    top = 0
    for i in range(0, height):
        top = i
        all_white = True
        for j in range(0, width):
            if np_image[top, j, 0] != 255 or np_image[top, j, 1] != 255 or np_image[top, j, 2] != 255:
                all_white = False
                break
        if not all_white:
            break

    bottom = height - 1
    for i in range(0, height):
        bottom = height - 1 - i
        all_white = True
        for j in range(0, width):
            if np_image[bottom, j, 0] != 255 or np_image[bottom, j, 1] != 255 or np_image[bottom, j, 2] != 255:
                all_white = False
                break
        if not all_white:
            break
    return left, top, right, bottom

# 把旅法师营地的图片，裁剪掉白边，只要有效图片
base_absolute_path = 'E:/WorkSpace/MyGithub/PictureMatchTool/data/temp'
save_folder = 'D:/Workspace/Git/MyGithub/PictureMatchTool/data/database'
# 509 * 709
crop_range = (79, 50, 448, 590)
# 1024 * 1024
# crop_range = (79, 50, 448, 590)
# cut_pictures_in_folder(base_absolute_path + '/小饰品', crop_range, save_folder + '/小饰品')
# cut_pictures_in_folder(base_absolute_path + '/大饰品', crop_range, save_folder + '/大饰品')
# TODO-high：待测试 自动裁剪白边 能否使用，能的话就不用手动计算了 。2024/09/04 18:50:26
del_white_of_pictures_in_folder(base_absolute_path + '/大饰品', save_folder + '/大饰品')


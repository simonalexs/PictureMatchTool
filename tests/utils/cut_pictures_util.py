import os

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


# 把旅法师营地的图片，裁剪掉白边，只要有效图片
base_absolute_path = 'D:/Workspace/Git/MyGithub/PictureMatchTool/data/temp/原版图片-未处理/旅法师营地-1024-1024'
save_folder = 'D:/Workspace/Git/MyGithub/PictureMatchTool/data/database'
cut_pictures_in_folder(base_absolute_path + '/小饰品', (237, 71, 775, 857), save_folder + '/小饰品')
cut_pictures_in_folder(base_absolute_path + '/大饰品', (237, 71, 775, 857), save_folder + '/大饰品')


import os

import requests


def get_relative_path(absolute_path: str, app_folder):
    return absolute_path.replace(app_folder, '')[1:]


def get_absolute_path_from_relative(relative_path: str, app_folder) -> str:
    if relative_path.startswith('/') or relative_path.startswith('\\'):
        relative_path = relative_path[1:]
    return os.path.join(app_folder, relative_path)


def is_custom_shortcut(filepath):
    """检查文件是否为自定义的快捷方式"""
    basename = os.path.basename(filepath)
    return os.path.isfile(filepath) and basename.startswith('shortcut') and basename.endswith('.txt')


def get_absolute_path_from_unknown(unknown_path, app_folder):
    if unknown_path.find(":/") == -1 and unknown_path.find(":\\") == -1:
        # 相对路径
        return get_absolute_path_from_relative(unknown_path, app_folder)
    return unknown_path


def get_custom_shortcut_target(filepath, app_folder):
    """获取自定义快捷方式的目标路径"""
    with open(filepath, 'r') as file:
        target_path = file.readline().strip()
        return get_absolute_path_from_unknown(target_path, app_folder)


def get_all_file_in_dir(folder_path, app_folder):
    """
    里面没有快捷方式时，就不用传 app_folder 这个参数，因为不会涉及到“相对路径”
    """
    file_paths = []
    if os.path.isfile(folder_path):
        file_paths.append(folder_path)
        return file_paths
    for sub_file_name in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, sub_file_name)
        if is_custom_shortcut(sub_path):
            target_path = get_custom_shortcut_target(sub_path, app_folder)
            # 递归
            sub_files = get_all_file_in_dir(target_path, app_folder)
            file_paths.extend(sub_files)
        elif os.path.isdir(sub_path):
            # 递归
            sub_files = get_all_file_in_dir(sub_path, app_folder)
            file_paths.extend(sub_files)
        else:
            file_paths.append(sub_path)
    return file_paths


def get_image_sub_region(height, width, region):
    region_left = int(width * region[0])
    region_top = int(height * region[1])
    region_right = int(width * region[2])
    region_bottom = int(height * region[3])
    return region_left, region_top, region_right, region_bottom


def compare_version(v1: str, v2: str) -> int:
    s1 = v1.split('.')
    s2 = v2.split('.')
    v1_value = int(s1[0]) * 10000 + int(s1[1]) * 100 + int(s1[2])
    v2_value = int(s2[0]) * 10000 + int(s2[1]) * 100 + int(s2[2])
    return v1_value - v2_value


def check_update_in_github(releases_url, this_version):
    content_before_version = releases_url.replace('https://github.com', '') + '/tag/'
    try:
        response = requests.get(releases_url, verify=False, timeout=3)
        # 检查请求是否成功
        if response.status_code == 200:
            # 获取网页的源代码
            html_content = response.text
            index = html_content.find(content_before_version)
            if index == -1:
                return -1, '检查失败，网页中未找到版本信息关键字'
            else:
                version_index = index + len(content_before_version)
                version_info = html_content[version_index:(version_index + 15)]
                latest_version = version_info[:version_info.find('"')]
                if compare_version(latest_version, this_version) > 0:
                    return 1, latest_version
                else:
                    return 0, '已是最新版本'
        else:
            return -1, '请求失败，请检查是否能访问 github，可尝试使用微软商店中的“watt toolkit”加速github'
    except Exception as e:
        return -1, '请求失败，请检查是否能访问 github，可尝试使用微软商店中的“watt toolkit”加速github'


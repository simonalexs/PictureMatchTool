import os

import mss
import requests
from PIL import Image
from toga import ScrollContainer


def screenshot_region(region):
    with mss.mss() as sct:
        width = region[2] - region[0]
        height = region[3] - region[1]
        screenshot = sct.grab({"left": region[0], "top": region[1], "width": width, "height": height})
        pim = Image.new("RGB", screenshot.size)
        pim.frombytes(screenshot.rgb)
        return pim


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


def find_widget_by_id(container, widget_id):
    """
    在指定的容器中递归查找具有给定 ID 的 Widget。

    :param container: 要搜索的容器（Box 或其他包含子元素的 Widget）
    :param widget_id: 要查找的 Widget 的 ID
    :return: 找到的 Widget，如果没有找到则返回 None
    """
    # if hasattr(container, 'text'):
    #     print(container, container.text)
    #     if 'en' == container.text:
    #         print(container.value)
    #         print(container.id, hasattr(container, 'id'), container.id == widget_id, container.id == 'scan_when_window_inactive')
    #         print(f'[{widget_id}]')
    #         print(f'[{container.id}]')
    # else:
    #     print(container)
    # 检查当前容器是否是我们正在查找的 Widget
    if hasattr(container, 'id') and container.id == widget_id:
        return container

    # 遍历所有子元素
    if isinstance(container, ScrollContainer):
        found_widget = find_widget_by_id(container.content, widget_id)
        if found_widget:
            return found_widget
    else:
        if hasattr(container, 'children'):
            for child in container.children:
                # 递归调用以查找子元素中的 Widget
                found_widget = find_widget_by_id(child, widget_id)
                if found_widget:
                    return found_widget

    # 如果没有找到，则返回 None
    return None
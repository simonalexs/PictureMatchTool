import os


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
# import win32com.client
import os


# def get_shell():
#     shell = win32com.client.Dispatch("WScript.Shell")
#     return shell
#
#
# def is_shortcut(filepath):
#     """检查文件是否为快捷方式"""
#     return os.path.isfile(filepath) and get_shell().IsLink(filepath)
#
#
# def get_shortcut_target(filepath):
#     """获取快捷方式的目标路径"""
#     link = get_shell().CreateShortCut(filepath)
#     return link.Targetpath

def get_all_file_in_dir(folder_path):
    file_paths = []
    # if os.path.isfile(folder_path):
    #     file_paths.append(folder_path)
    #     return file_paths
    for sub_file_name in os.listdir(folder_path):
        sub_path = os.path.join(folder_path, sub_file_name)
    #     if is_shortcut(sub_path):
    #         target_path = get_shortcut_target(sub_path)
    #         # 递归
    #         sub_files = get_all_file_in_dir(target_path)
    #         file_paths.extend(sub_files)
    #     elif os.path.isdir(sub_path):
    #         # 递归
    #         sub_files = get_all_file_in_dir(sub_path)
    #         file_paths.extend(sub_files)
    #     else:
    #         file_paths.append(sub_path)
        file_paths.append(sub_path)
    return file_paths
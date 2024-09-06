"""
my first
"""
import json
import math
import webbrowser
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from numpy import float32
from pygetwindow import Win32Window
from toga.style import Pack
from toga.style import pack

from .utils import common_utils
import asyncio
import copy
from pathlib import Path

import toga
import sys
import time

from travertino.constants import *
from PIL import ImageGrab
from PIL import Image
import pygetwindow as gw
import os
import cv2
import numpy as np
import shutil
import pyperclip
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import threading
from functools import partial

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata

symbol = '/'
app_main_box = None


def find_widget_by_id(widget_id):
    return common_utils.find_widget_by_id(app_main_box, widget_id)


def cv2Pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil2Cv(img):
    # 直接用 cv2.imread 读取的“中文名称”的图片，在 cv2.matchTemplate 方法中，会报错，所以用pil读取后转化
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def copy_to_clipboard_for_button(widget, *args, content: str, is_show=False, **kwargs):
    """
    toga.button 的 on_press 可以用这种方法传参
    使用方： from functools import partial
           on_press=partial(function_name, content='123')
    https://github.com/beeware/toga/discussions/1987
    """
    copy_to_clipboard(content)
    if is_show:
        log_manager.info(f'已复制到剪切板：【{content}】', 'info')


def copy_to_clipboard(content):
    pyperclip.copy(content)


class LogManager:
    def __init__(self, app_path: Path):
        self.app_path = app_path

    def get_app_folder(self):
        parent_path = self.app_path.parent
        if parent_path.name == 'app':
            # 打包之后自定义的资源文件存放在这个目录下，所以就把这里定义为 app_folder 吧
            return str(parent_path)
        # 开发环境下，再上一级才是根目录
        return str(parent_path.parent)

    def get_app_config_path(self):
        return os.path.join(self.get_app_folder(), 'config', 'config.json5')

    def debug(self, msg, config_name, type='debug'):
        self.__write_log('debug', type, msg, config_name)

    def info(self, msg, config_name, type='info'):
        self.__write_log('info', type, msg, config_name)
        self.debug(msg, config_name, type)

    def warn(self, msg, config_name, type='warn'):
        self.__write_log('warn', type, msg, config_name)
        self.info(msg, config_name, type)

    def error(self, msg, config_name, type='error'):
        self.__write_log('error', type, msg, config_name)
        self.warn(msg, config_name, type)

    def get_log(self, file_name) -> list[str]:
        file_path = self.__get_log_path(file_name)
        result = []
        if not os.path.exists(file_path):
            return result
        with open(file_path, 'r') as file:
            line = file.readline()
            while line:
                if line.endswith('\n'):
                    line = line.replace('\n', '')
                if line == '':
                    continue
                result.append(line)
                line = file.readline()
        return result

    def __write_log(self, file_name, type, content, config_name):
        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        millisecondsInt = int((now - now.replace(microsecond=0)).total_seconds() * 1000)
        real_content = f"{now_str}.{millisecondsInt:03} [{type}] [{config_name}] {content}"

        file_path = self.__get_log_path(file_name)
        with open(file_path, 'a') as file:
            file.write(real_content + '\n')
        # 日志文件过大时，减小文件体积
        if os.path.exists(file_path):
            file_size_byte = os.path.getsize(file_path)
            if file_size_byte >= 10 * 1024 * 1024:
                all_lines: list[str] = self.get_log(file_name)
                with open(file_path, 'w') as file:
                    file.write('\n'.join(all_lines[-2000:]))

    def __get_log_path(self, file_name):
        path = self.get_app_folder() + symbol + 'logs'
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = path + symbol + file_name + '.log'
        return file_path

    def clear_log(self):
        types = ['info', 'warn', 'debug']
        for t in types:
            file_path = self.__get_log_path(t)
            if os.path.exists(file_path):
                os.remove(file_path)


class LockInfo:
    def __init__(self):
        self.is_loading_data = False


log_manager = LogManager(Path())
lock_info = LockInfo()


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float32):
            return float(str(obj))
        if isinstance(obj, Config):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)


class CacheEntity:
    def __init__(self, cache_block_file_name, threadhold_match_rate, contraction_ratio, match_rate,
                 relative_result_picture_path, image=None):
        self.cache_block_file_name: str = cache_block_file_name
        self.threadhold_match_rate: float = threadhold_match_rate
        self.contraction_ratio: float = contraction_ratio
        self.match_rate: float = match_rate
        self.relative_result_picture_path: str = relative_result_picture_path
        # 内存中使用
        self.image = image


class AppConfig:
    def __init__(self, config_dict_list: list[dict], enable_debug_mode=False, scan_interval_seconds=5, enable_cache=True,
                 cache_similar_region=None, scan_when_window_inactive=False, cache_similar_region_match_rate=0.9,
                 show_log_num=10, pixel_error_range_when_judge_full_screen=0):
        if cache_similar_region is None:
            cache_similar_region = [0.05, 0.05, 0.95, 0.95]
        self.enable_debug_mode: bool = enable_debug_mode
        # 开启debug模式，用于debug的截图和日志输出（截游戏窗口和全屏，以及打印坐标日志），用于调试出真正的饰品技能区域

        self.scan_interval_seconds: int = scan_interval_seconds
        # 自动扫描运行周期，单位 秒

        self.scan_when_window_inactive: bool = scan_when_window_inactive
        # 窗口“未激活”时，是否执行扫描操作（未激活是指，在windows中该窗口未在最上方，也就是不是鼠标最新点击的窗口）

        self.enable_cache: bool = enable_cache
        # 是否开启缓存

        self.cache_similar_region: list[float] = cache_similar_region
        # 使用缓存时，用最新截图的多大区域去和缓存库做对比
        # （如果用全图完全相同才能使用缓存，则会导致如果截图时游戏画面稍微抖动，就会有几个像素的偏差，这样就无法使用缓存了）

        self.cache_similar_region_match_rate: float = cache_similar_region_match_rate
        # 截图与缓存库的匹配度阈值，高于此阈值时，才认为命中缓存

        self.show_log_num = show_log_num
        # 日志显示条数

        self.pixel_error_range_when_judge_full_screen: int = pixel_error_range_when_judge_full_screen
        """判断窗口是否是“全屏”时，允许的像素误差范围"""

        self.configs: list[Config] = []
        if config_dict_list is not None:
            for config_dict in config_dict_list:
                self.configs.append(Config(**config_dict))


class Config:
    def __init__(self, **kwargs):
        self.name: str = kwargs.get("name", "unknown_config")
        """配置名称"""

        self.description: str = kwargs.get("description", "")

        self.enable: bool = kwargs.get("enable", False)
        """配置名称"""

        self.window_name: str = kwargs.get("window_name", "unknown_window_name")
        """要监听的应用程序窗口 title ，当可能有多种名称时可使用“,”分隔（例如中文名和英文名） """

        self.region_in_full_screen: list[float] = kwargs.get("region_in_full_screen", [])
        """游戏内的小图（游戏窗口全屏状态下）  [left距离左边, top距离上边, right距离左边, bottom距离上边]"""

        self.region: list[float] = kwargs.get("region", [])
        """游戏内的小图（窗口模式状态下）  [left距离左边, top距离上边, right距离左边, bottom距离上边]"""

        self.db_picture_valid_region: list[float] = kwargs.get("db_picture_valid_region", [0, 0, 1, 1])
        """图片库中图片的有效匹配区域，用于和游戏内的小图匹配时避免无效匹配  [left距离左边, top距离上边, right距离左边, bottom距离上边]"""

        self.is_lock_contraction_ratio: bool = kwargs.get("is_lock_contraction_ratio", False)
        """
        不固定时，会自动在收缩比范围内遍历寻找匹配度最高的结果
        固定时，只会以指定的 fixed_contraction_ratio 去寻找
        """

        self.fixed_contraction_ratio: float = kwargs.get("fixed_contraction_ratio", 0.59)
        """小图与原图的放缩比例"""

        self.auto_contraction_ratio_range: list[float] = kwargs.get("auto_contraction_ratio_range", [0.4, 0.7])
        """
        is_lock_contraction_ratio = False 时，自动以此收缩比范围去遍历寻找结果
        """

        self.auto_contraction_ratio_step: float = kwargs.get("auto_contraction_ratio_step", 0.01)
        """收缩比范围的遍历步长"""

        self.threadhold_match_rate: float = kwargs.get("threadhold_match_rate", 0.8)
        """计算图片匹配度时的阈值，大于这一阈值才认为图片是匹配的"""

        self.result_target_folder: str = kwargs.get("result_target_folder", "data/result")
        """
        结果文件存放的文件夹（默认会以配置名称作为文件名存放）
        """

    def __get_base_folder(self):
        path = os.path.join(log_manager.get_app_folder(), 'data')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_database_folder(self):
        path = os.path.join(self.__get_base_folder(), 'database', self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_cache_folder(self):
        path = os.path.join(self.__get_base_folder(), 'cache', self.name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_temp_folder(self):
        path = os.path.join(self.__get_base_folder(), 'temp')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_result_target_path(self) -> str:
        folder = self.result_target_folder
        if folder.find(":/") == -1 and folder.find(":\\") == -1:
            # 相对路径
            folder = common_utils.get_absolute_path_from_relative(folder, log_manager.get_app_folder())
        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, self.name + '.png')

    def get_real_region_by_config(self, config_region: list[float, float, float, float], window):
        key_left = math.ceil(window.width * config_region[0])
        key_top = math.ceil(window.height * config_region[1])
        key_right = math.floor(window.width * config_region[2])
        key_bottom = math.floor(window.height * config_region[3])
        return window.left + key_left, window.top + key_top, window.left + key_right, window.top + key_bottom

    def add_cache(self, ratio_2_match_rate_2_path: tuple[float, float, str], block_picture_path: str) -> CacheEntity:
        cache_folder = self.get_cache_folder()
        # 复制 block文件到缓存文件夹
        cache_block_file_name = 'block-' + str(datetime.now().timestamp()) + '.png'
        cache_block_path = os.path.join(cache_folder, cache_block_file_name)
        shutil.copy(block_picture_path, cache_block_path)
        # 写入缓存信息到缓存文件里
        cache_datas = self.read_cache_file()
        data = CacheEntity(
            cache_block_file_name,
            self.threadhold_match_rate,
            ratio_2_match_rate_2_path[0],
            ratio_2_match_rate_2_path[1],
            common_utils.get_relative_path(ratio_2_match_rate_2_path[2], log_manager.get_app_folder()))
        cache_datas.append(data)
        self.__write_cache_file(cache_datas)
        return data

    def clear_cache(self):
        cache_folder = self.get_cache_folder()
        if os.path.exists(cache_folder):
            shutil.rmtree(cache_folder)

    def __get_cache_list_file_path(self):
        path = os.path.join(self.get_cache_folder(), 'list.json')
        return path

    def read_cache_file(self) -> list[CacheEntity]:
        file_path = self.__get_cache_list_file_path()
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r') as file:
            content = file.read()
        if content is None or content == '':
            return []
        dicts = json.loads(content)
        res = []
        for dictionary in dicts:
            cache_entity = CacheEntity(**dictionary)
            res.append(cache_entity)
        return res

    def __write_cache_file(self, data: list[CacheEntity]):
        with open(self.__get_cache_list_file_path(), 'w') as file:
            dicts = []
            for entity in data:
                dicts.append(entity.__dict__)
            file.write(json.dumps(dicts, indent=4, ensure_ascii=False, cls=CustomEncoder))


class DbPicture:
    def __init__(self):
        self.path: str = ''
        self.image = None


class PictureMatchManager:
    def __init__(self):
        self.app_config: AppConfig = self.load_app_config()
        self.db_pictures: dict = {}
        self.cache_pictures: dict = {}

    def load_app_config(self):
        app_config_path = log_manager.get_app_config_path()
        if not os.path.exists(app_config_path):
            with open(app_config_path, 'w') as file:
                file.write(json.dumps(AppConfig([]).__dict__, indent=4, ensure_ascii=False))
        with open(app_config_path, 'r') as file:
            app_config_dict: dict = json.loads(file.read())
            app_config = AppConfig(app_config_dict.get("configs"),
                                   app_config_dict.get('enable_debug_mode'),
                                   app_config_dict.get('scan_interval_seconds'),
                                   app_config_dict.get('enable_cache'),
                                   app_config_dict.get('cache_similar_region'),
                                   app_config_dict.get('scan_when_window_inactive'),
                                   app_config_dict.get('cache_similar_region_match_rate'),
                                   app_config_dict.get('show_log_num'),
                                   app_config_dict.get('pixel_error_range_when_judge_full_screen'))
        return app_config

    def load_db_pictures(self):
        if lock_info.is_loading_data:
            log_manager.info(f'已有加载数据的任务，请稍后再试', 'load_db_pictures')
            return
        lock_info.is_loading_data = True
        try:
            log_manager.info(f'开始加载图片库', 'load_db_pictures')
            self.db_pictures.clear()
            start = time.time_ns() // 1000000
            for config in self.app_config.configs:
                pictures: list[DbPicture] = []
                database_folder_path = config.get_database_folder()
                all_file_path = common_utils.get_all_file_in_dir(database_folder_path, log_manager.get_app_folder())
                log_manager.info(f'扫描到 {len(all_file_path)} 个文件', config.name)
                for file_path in all_file_path:
                    if not file_path.endswith('.png'):
                        continue
                    img = pil2Cv(Image.open(file_path))
                    # 获取有效匹配区域
                    valid_region = common_utils.get_image_sub_region(img.shape[:2][0], img.shape[:2][1], config.db_picture_valid_region)
                    valid_picture = img[valid_region[0]:valid_region[2], valid_region[1]:valid_region[3]]
                    # 存入内存
                    picture = DbPicture()
                    picture.path = file_path
                    picture.image = valid_picture
                    pictures.append(picture)
                self.db_pictures[config.name] = pictures
                log_manager.info(f'已加载 {len(pictures)} 个图片', config.name)
            end = time.time_ns() // 1000000
            log_manager.info(f'加载图片库完成，共耗时 {str(end - start)} ms', 'load_db_pictures')
        finally:
            lock_info.is_loading_data = False

    def load_cache_pictures(self):
        if lock_info.is_loading_data:
            log_manager.info(f'已有加载数据的任务，请稍后再试', 'load_cache_pictures')
            return False
        lock_info.is_loading_data = True
        try:
            self.cache_pictures.clear()
            if not self.app_config.enable_cache:
                return False
            log_manager.info(f'开始加载缓存', 'load_cache_pictures')
            start = time.time_ns() // 1000000
            for config in self.app_config.configs:
                cache_pictures: list[CacheEntity] = []
                cache_folder_path = config.get_cache_folder()
                cache_datas = config.read_cache_file()
                for cache_data in cache_datas:
                    cache_file_path = os.path.join(cache_folder_path, cache_data.cache_block_file_name)
                    cache_image = pil2Cv(Image.open(cache_file_path))
                    cache_data.image = cache_image
                    cache_pictures.append(cache_data)
                self.cache_pictures[config.name] = cache_pictures
                log_manager.info(f'已加载 {len(cache_pictures)} 个图片', config.name)
            end = time.time_ns() // 1000000
            log_manager.info(f'加载缓存完成，共耗时 {str(end - start)} ms', 'load_cache_pictures')
            return True
        finally:
            lock_info.is_loading_data = False

    def save_app_config(self, app_config: AppConfig):
        with open(log_manager.get_app_config_path(), 'w') as file:
            file.write(json.dumps(app_config.__dict__, indent=4, ensure_ascii=False, cls=CustomEncoder))

    def save_config(self, config_update: Config):
        app_config = self.app_config
        configs = app_config.configs
        for config in configs:
            if config.name == config_update.name:
                config.__dict__ = copy.deepcopy(config_update.__dict__)
                break
        self.save_app_config(app_config)

    def run_all_configs(self):
        while lock_info.is_loading_data:
            log_manager.info('等待后台数据加载完毕...', 'configs')
            time.sleep(1)
        log_manager.info('==================== 开始识别 ============================', 'configs')
        start = time.time_ns() // 1000000
        for config in self.app_config.configs:
            if config.enable:
                try:
                    t1 = time.time_ns() // 1000000
                    status, message = self.do_run_config(config, self.app_config)
                    t2 = time.time_ns() // 1000000
                    log_manager.info(message + '（耗时：' + str(t2 - t1) + ' ms）', config.name)
                except Exception as e:
                    logging.exception(e)
                    log_manager.error(str(e), config.name)
        end = time.time_ns() // 1000000
        log_manager.info(f"总耗时：{str(end - start)} ms", 'configs')

    def do_run_config(self, config: Config, app_config: AppConfig):
        windows: list[Win32Window] = []
        window_name_split = config.window_name.split(',')
        for window_name in window_name_split:
            windows.extend(gw.getWindowsWithTitle(window_name))
        if len(windows) == 0:
            return False, '未找到窗口[' + config.window_name + ']'
        res_windows = []
        for window in windows:
            if window_name_split.count(window.title) > 0:
                res_windows.append(window)
        if len(res_windows) == 0:
            return False, '未找到窗口[' + config.window_name + ']'
        if len(res_windows) > 1:
            return False, '找到' + str(len(res_windows)) + '个名叫[' + config.window_name + ']的窗口'
        window = res_windows[0]
        if window.isMinimized:
            return False, '窗口已最小化，终止识别'
        if not window.isActive and not self.app_config.scan_when_window_inactive:
            return False, '窗口未激活，终止识别'
        # 截取图片
        png_path = config.get_temp_folder() + symbol + config.name + '.png'
        # 判断当前窗口是否都在一个屏幕里
        screen_index = common_utils.get_window_screen_index_if_window_in_a_screen(window, app_config.pixel_error_range_when_judge_full_screen)
        if screen_index == -1:
            return False, '窗口没有完全处于同一个显示器内，终止识别'
        # 依据当前窗口是否是全屏状态，取用不同的配置信息
        if common_utils.is_window_full_screen(window, screen_index, app_config.pixel_error_range_when_judge_full_screen):
            log_manager.debug('检测到当前窗口为：全屏模式', config.name)
            if len(config.region_in_full_screen) == 0:
                return False, '当前配置未适配全屏模式，终止识别'
            config_region = config.region_in_full_screen
        else:
            log_manager.debug('检测到当前窗口为：窗口模式', config.name)
            if len(config.region) == 0:
                return False, '当前配置未适配窗口模式，终止识别'
            config_region = config.region
        real_region = config.get_real_region_by_config(config_region, window)
        image = common_utils.screenshot_region(real_region)
        image.save(png_path)

        if self.app_config.enable_debug_mode:
            # 当截取的目标区域不是想要的饰品区域时，依据这个游戏窗口截图，和全屏截图，以及打印的坐标参数，来分析是哪里的问题
            # 从 0，0 截取到游戏窗口右下角
            screen_path = config.get_temp_folder() + symbol + 'screenshot_0_0.png'
            common_utils.screenshot_region((0, 0, window.right, window.bottom)).save(screen_path)
            # 截取游戏窗口
            game_window_path = config.get_temp_folder() + symbol + 'screenshot_window.png'
            common_utils.screenshot_region((window.left, window.top, window.right, window.bottom)).save(game_window_path)
            # 打印坐标信息
            log_manager.info(f'window.left={window.left}, window.top={window.top}, window.right={window.right}, window.bottom={window.bottom}', config.name)
            log_manager.info(f'window.width={window.width}, window.height={window.height}', config.name)
            log_manager.info(f'region.left={real_region[0]}, region.top={real_region[1]}, region.right={real_region[2]}, region.bottom={real_region[3]}', config.name)

        # 从数据库中识别
        block = pil2Cv(Image.open(png_path))
        config_name = config.name
        ratio_2_match_rate_2_path = None
        is_result_from_cache = False
        # 先从缓存中获取
        if self.app_config.enable_cache:
            log_manager.debug('查找缓存...', config_name)
            max_rate_res = self.get_max_similar_from_cache(block, self.cache_pictures[config_name], self.app_config.cache_similar_region, config)
            if max_rate_res[0][0] >= self.app_config.cache_similar_region_match_rate:
                # 大于指定阈值
                log_manager.debug(f'缓存中已找到，与缓存图的最高匹配度为：{str(max_rate_res[0][0])}', config_name)
                ratio_2_match_rate_2_path = max_rate_res[1]
                is_result_from_cache = True
            else:
                log_manager.debug(f'缓存中未找到，与缓存库中的最高匹配度：{str(max_rate_res[0][0])}，缓存图名称：{str(max_rate_res[0][1])}，'
                                 f'结果图收缩比：{str(max_rate_res[1][0])}，结果图匹配度：{str(max_rate_res[1][1])}，'
                                 f'结果图路径：{str(max_rate_res[1][2])}', config_name)
        else:
            log_manager.debug('缓存未开启', config_name)
        if ratio_2_match_rate_2_path is None:
            # 缓存中没有，从数据库中遍历
            log_manager.debug('开始遍历数据库', config_name)
            ratio_2_match_rate_2_path = self.find_picture_in_db(block, config)
        if ratio_2_match_rate_2_path is None:
            return False, '未找到结果'
        elif ratio_2_match_rate_2_path[1] < config.threadhold_match_rate:
            # 找到的结果不符合阈值
            return False, f'未找到，图片库中最匹配的是：收缩比：{ratio_2_match_rate_2_path[0]}，匹配度：{ratio_2_match_rate_2_path[1]}' \
                          f'，路径：{ratio_2_match_rate_2_path[2].replace(log_manager.get_app_folder(), "")}'
        else:
            # 找到符合阈值的结果了
            shutil.copyfile(ratio_2_match_rate_2_path[2], config.get_result_target_path())
            message = f'已找到，收缩比：{str(ratio_2_match_rate_2_path[0])}, 匹配度：{str(ratio_2_match_rate_2_path[1])}' \
                      f'，找到的目标：{ratio_2_match_rate_2_path[2].replace(log_manager.get_app_folder(), "")}'
            if self.app_config.enable_cache and not is_result_from_cache:
                # 添加到缓存
                cache_data = config.add_cache(ratio_2_match_rate_2_path, png_path)
                cache_data.image = block
                self.cache_pictures[config_name].append(cache_data)
                # 判断缓存文件夹的大小，过大时清理缓存
                file_size_byte = os.path.getsize(config.get_cache_folder())
                if file_size_byte >= 20 * 1024 * 1024:
                    os.rmdir(config.get_cache_folder())
                    self.load_cache_pictures()
            return True, message

    def get_max_similar_from_cache(self, block, cache_pictures: list[CacheEntity], cache_similar_region: list[float], config: Config):
        # 计算 block 的90%左右区域 在 cache库 中的最高匹配度
        block_sub_region = common_utils.get_image_sub_region(block.shape[:2][0], block.shape[:2][1], cache_similar_region)
        sub_block = block[block_sub_region[1]:block_sub_region[3], block_sub_region[0]:block_sub_region[2]]
        max_rate_res = [(0.6, ''), (0, 0, '')]
        for cache_picture in cache_pictures:
            if cache_picture.threadhold_match_rate != config.threadhold_match_rate:
                # 和缓存图片 匹配阈值 的配置不同，或 不相似
                continue
            match_rate = self.do_cal_picture_match_rate(sub_block, cache_picture.image)
            if match_rate <= max_rate_res[0][0]:
                continue
            # 找到匹配度更高的了
            result_picture_path = common_utils.get_absolute_path_from_relative(cache_picture.relative_result_picture_path, log_manager.get_app_folder())
            if os.path.exists(result_picture_path):
                max_rate_res = [(match_rate, cache_picture.cache_block_file_name),
                                (cache_picture.contraction_ratio, cache_picture.match_rate, result_picture_path)]
            else:
                # 有缓存图片，但文件里没有目标文件，删除此缓存
                os.remove(os.path.join(config.get_cache_folder(), cache_picture.cache_block_file_name))
        return max_rate_res

    def find_picture_in_db(self, block, config: Config):
        config_name = config.name
        if config.is_lock_contraction_ratio:
            log_manager.debug('固定收缩比模式，当前收缩比：' + str(config.fixed_contraction_ratio), config_name)
            ratio_2_match_rate_2_path = self.find_fittest_picture(block,
                                                                  config.fixed_contraction_ratio,
                                                                  config_name)
        else:
            log_manager.debug('自动查找模式，要扫描的收缩比范围：' + str(config.auto_contraction_ratio_range)
                             + ', 步长：' + str(config.auto_contraction_ratio_step), config_name)
            ratio_2_match_rate_2_path = self.find_highest_match_rate_picture(block,
                                                                             config.auto_contraction_ratio_range,
                                                                             config.auto_contraction_ratio_step,
                                                                             config_name)
        return ratio_2_match_rate_2_path

    def do_cal_picture_match_rate(self, block, img, contraction_ratio=1.0):
        """
        计算两个图片的最大匹配度，前者是否是后者的一部分
        """
        # 依据收缩比放缩原图片
        if contraction_ratio is None:
            return -1
        reshaped_height = int(img.shape[:2][0] * contraction_ratio)
        reshaped_width = int(img.shape[:2][1] * contraction_ratio)
        if reshaped_height < block.shape[:2][0] or reshaped_width < block.shape[:2][1]:
            return 0
        new_img = cv2.resize(img, [reshaped_width, reshaped_height])
        res = cv2.matchTemplate(new_img, block, cv2.TM_CCOEFF_NORMED)
        return np.amax(res)

    def find_highest_match_rate_picture(self, block, contraction_ratio_range,
                                        contraction_ratio_step, config_name):
        """
        遍历收缩比，逐一寻找匹配度最高的图片
        """
        start_ratio_int = int(100 * contraction_ratio_range[0])
        end_ratio_int = int(100 * contraction_ratio_range[1])
        step = int(100 * contraction_ratio_step)

        ratio_2_match_rate_2_path = None
        for i in range(start_ratio_int, end_ratio_int + 1, step):
            contraction_ratio = float(i) / 100
            fittest = self.find_fittest_picture(block, contraction_ratio, config_name)
            if fittest is None:
                continue
            if ratio_2_match_rate_2_path is None:
                ratio_2_match_rate_2_path = fittest
            elif fittest[1] > ratio_2_match_rate_2_path[1]:
                ratio_2_match_rate_2_path = fittest
        return ratio_2_match_rate_2_path

    def find_fittest_picture(self, block, contraction_ratio, config_name):
        res_file_path = None
        res_rate = 0
        pictures: list[DbPicture] = self.db_pictures[config_name]
        for db_picture in pictures:
            rate = self.do_cal_picture_match_rate(block, db_picture.image, contraction_ratio)
            if rate > res_rate:
                res_rate = rate
                res_file_path = db_picture.path
        if res_file_path is None:
            return None
        return (contraction_ratio, res_rate, res_file_path)


class Item(toga.Box):
    def __init__(self, text):
        super().__init__(style=Pack(direction=ROW))

        self.label = toga.Label(
            text,
            style=Pack(),
        )
        self.add(self.label)

    def set(self, text):
        self.label.text = text


class PictureMatchTool(toga.App):

    def __init__(self, **options):
        super().__init__(**options)
        log_manager.app_path = self.app.paths.app
        log_manager.clear_log()

        self.releases_url = 'https://github.com/simonalexs/PictureMatchTool/releases'
        self.scheduler = BackgroundScheduler()
        self.picture_match_manager = PictureMatchManager()

        # Find the name of the module that was used to start the app
        app_module = sys.modules['__main__'].__package__
        # Retrieve the app's metadata
        self.metadata = importlib_metadata.metadata(app_module)

    def startup(self):
        self.main_window = toga.MainWindow(title=self.formal_name + '-' + self.version)
        self.main_window.content = self.create_main_box()
        global app_main_box
        app_main_box = self.main_box
        self.main_window.show()

        self.on_running = self.refresh_ui_handler

        self.scheduler.add_job(self.after_started, 'date', id='after_started')
        self.scheduler.start()

    def after_started(self):
        self.check_update()
        self.load_data()

    def get_app_config(self) -> AppConfig:
        return self.picture_match_manager.app_config

    def load_data(self, widget=None, **kwargs):
        if lock_info.is_loading_data:
            log_manager.info(f'已有加载数据的任务，请稍后再试', 'load_data')
            return
        self.picture_match_manager.load_db_pictures()
        self.picture_match_manager.load_cache_pictures()

    def check_update(self, widget=None, **kwargs):
        log_manager.info('开始检查新版本', 'check_update')
        status, info = common_utils.check_update_in_github(self.releases_url, self.version)
        if status == 1:
            log_manager.info('检测到新版本：' + info + '，可点击右上角“下载新版本”按钮前往下载', 'check_update')
        elif status == 0:
            log_manager.info(info, 'check_update')
        else:
            log_manager.warn(info, 'check_update')

    def create_main_box(self):
        self.padding = 10
        self.width = 1200
        self.main_box = toga.Box(style=Pack(direction=COLUMN))
        self.main_box.style.width = self.width
        self.main_box.style.height = 700

        # header
        header_button_box = self.create_header_button_box()
        self.main_box.add(header_button_box)

        # header-2
        header_button_box2 = self.create_header_button_box2()
        self.main_box.add(header_button_box2)

        divider = toga.Label('', style=Pack(background_color='#D3D3D3', height=2, width=self.width - 40,
                                            padding_left=20, padding_right=20))
        self.main_box.add(divider)

        # header-2
        header_button_box3 = self.create_header_button_box3()
        self.main_box.add(header_button_box3)

        # body
        body_scroll_container = toga.ScrollContainer(content=self.create_body_box(), horizontal=False, vertical=True,
                                                     style=Pack(width=self.width,
                                                                height=200))
        self.main_box.add(body_scroll_container)

        divider2 = toga.Label('', style=Pack(background_color='#D3D3D3', height=2, width=self.width - 40,
                                            padding_left=20, padding_right=20))
        self.main_box.add(divider2)

        # footer
        footer_scroll_height = 200
        footer_scroll_container = toga.ScrollContainer(horizontal=False, vertical=True,
                                                       style=Pack(direction=COLUMN, flex=1))
        footer_scroll_container.content = self.create_footer_box(footer_scroll_height)
        self.main_box.add(footer_scroll_container)

        return self.main_box

    def create_header_button_box(self):
        box = toga.Box(style=Pack(padding=self.padding, width=self.width))
        # add_config_button = toga.Button('添加配置', on_press=self.add_config_button_handler())
        # box.add(add_config_button)
        # box.add(toga.Button('识别全部'))

        self.start_all_configs_btn = toga.Button('开始自动识别', on_press=self.start_all_configs_btn_handler)
        box.add(self.start_all_configs_btn)
        self.stop_all_configs_btn = toga.Button('停止自动识别', on_press=self.stop_all_configs_btn_handler)
        self.stop_all_configs_btn.enabled = False
        box.add(self.stop_all_configs_btn)

        box.add(toga.Label('空白，用来占用空间', style=Pack(flex=1, visibility='hidden')))

        # box.add(toga.Label('空白，用来占用空间', style=Pack(flex=1, visibility='hidden')))

        box.add(toga.Button('重新加载图片库', style=Pack(width=100), on_press=self.load_data))

        box.add(toga.Button('检查更新', style=Pack(width=80), on_press=self.check_update))
        self.download_new_version_btn = toga.Button('下载新版本', style=Pack(width=80), on_press=lambda widget: webbrowser.open(self.releases_url))
        box.add(self.download_new_version_btn)

        box.add(toga.Label('空白，用来占用空间', style=Pack(width=50, visibility='hidden')))

        # self.picture_source_btn = toga.Button('图片库来源：旅法师营地@Bennidge', style=Pack(flex=2),
        #                                       on_press=lambda widget: webbrowser.open(
        #                                           'https://www.iyingdi.com/tz/people/55547'))
        # box.add(self.picture_source_btn)
        return box

    def start_all_configs_btn_handler(self, widget, **kwargs):
        if lock_info.is_loading_data:
            log_manager.info(f'正在加载数据，请稍后再试', 'load')
            return
        scan_interval_second = self.picture_match_manager.app_config.scan_interval_seconds
        self.scheduler.add_job(self.picture_match_manager.run_all_configs, 'interval', seconds=scan_interval_second,
                               id='job_all',
                               replace_existing=True)
        if not self.scheduler.running:
            self.scheduler.start()
        self.start_all_configs_btn.enabled = False
        self.stop_all_configs_btn.enabled = True
        log_manager.info(f'开始识别任务（首次执行会在{str(scan_interval_second)}秒后开始）', 'configs')

    def stop_all_configs_btn_handler(self, widget, **kwargs):
        self.scheduler.remove_job('job_all')
        self.start_all_configs_btn.enabled = True
        self.stop_all_configs_btn.enabled = False
        log_manager.info('已停止识别（如果仍看到有新的识别日志，请不用担心，那只是最后一次的识别任务，之后就不会再自动识别了）', 'configs')

    def create_header_button_box2(self):
        box = toga.Box(style=Pack(padding=self.padding, width=self.width))

        result_folder_path = os.path.dirname(self.get_app_config().configs[0].get_result_target_path())
        box.add(toga.Label('匹配结果路径：', style=Pack()))
        box.add(toga.Label(result_folder_path, style=Pack()))
        box.add(toga.Label('空白，用来占用空间', style=Pack(width=10, visibility='hidden')))
        box.add(toga.Button('复制', on_press=partial(copy_to_clipboard_for_button, content=result_folder_path, is_show=True)))
        return box

    def create_header_button_box3(self):
        box = toga.Box(style=Pack(padding=self.padding, width=self.width))

        app_config = self.get_app_config()
        box.add(toga.Switch('允许窗口未激活时识别', id='scan_when_window_inactive', value=app_config.scan_when_window_inactive))

        box.add(toga.Switch('启用缓存', id='enable_cache', value=app_config.enable_cache))

        box.add(toga.Switch('调试模式', id='enable_debug_screenshot', value=app_config.enable_debug_mode))

        box.add(toga.Label('判断是否全屏时允许的误差(像素)：'))
        box.add(toga.TextInput(id='pixel_error_range_when_judge_full_screen', value=str(app_config.pixel_error_range_when_judge_full_screen),
                       style=Pack(width=50))),

        box.add(toga.Label('空白，用来占用空间', style=Pack(flex=1, visibility='hidden')))

        box.add(toga.Button('保存配置', on_press=self.save_config_btn_handler, style=Pack()))

        box.add(toga.Label('空白，用来占用空间', style=Pack(width=50, visibility='hidden')))
        return box

    async def save_config_btn_handler(self, widget, **kwargs):
        confirm_dialog = toga.QuestionDialog('二次确认', '确认要保存当前配置信息？')
        if not await self.dialog(confirm_dialog):
            return
        app_config = self.get_app_config()
        app_config.scan_when_window_inactive = find_widget_by_id('scan_when_window_inactive').value
        app_config.enable_debug_mode = find_widget_by_id('enable_debug_screenshot').value
        app_config.enable_cache = find_widget_by_id('enable_cache').value
        app_config.pixel_error_range_when_judge_full_screen = int(find_widget_by_id('pixel_error_range_when_judge_full_screen').value)

        for config in app_config.configs:
            config_name = config.name
            config.enable = find_widget_by_id(f'{config_name}_enable').value
            config.region = self.get_text_input_value_list(f'{config_name}_region')
            config.region_in_full_screen = self.get_text_input_value_list(f'{config_name}_region_in_full_screen')
            config.db_picture_valid_region = self.get_text_input_value_list(f'{config_name}_db_picture_valid_region')
            config.auto_contraction_ratio_range = self.get_text_input_value_list(f'{config_name}_auto_contraction_ratio_range')
            config.auto_contraction_ratio_step = float(find_widget_by_id(f'{config_name}_auto_contraction_ratio_step').value)
            config.threadhold_match_rate = float(find_widget_by_id(f'{config_name}_threadhold_match_rate').value)
        # 保存配置
        self.picture_match_manager.save_app_config(app_config)
        log_manager.info('保存配置成功（配置已生效，不需要重启本软件。若已开启自动识别，则配置可能会延迟几秒生效，不用担心）', 'save_config')
        # 开启缓存的话重新加载缓存
        self.picture_match_manager.load_cache_pictures()

    def get_text_input_value_list(self, id: str) -> list[float]:
        region_str: str = find_widget_by_id(id).value
        region_splits = region_str.split(',')
        region = []
        for i in range(0, len(region_splits)):
            region.append(float(region_splits[i].strip()))
        return region

    def create_body_box(self):
        body = toga.Box(style=Pack(padding=self.padding))
        body.style.direction = COLUMN

        header_names = ["启用", "配置名称", "窗口名", "窗口截图区域", "全屏截图区域", "截图预览", "图片库图片有效区域", "收缩比范围", "收缩比步长", "匹配度阈值"]
        self.table_box_column_widths = [30, 60, 180, 150, 150, 80, 120, 70, 70, 70]

        self.cell_padding_left = 10
        header_box = toga.Box()
        for i in range(len(header_names)):
            header_box.add(
                toga.Label(header_names[i], style=Pack(font_weight='bold', padding_left=self.cell_padding_left,
                                                       width=self.table_box_column_widths[i])))
        body.add(header_box)

        self.table_box = toga.Box()
        self.table_box.style.direction = COLUMN
        self.refresh_table_box()
        body.add(self.table_box)

        # my_image = toga.Image(self.paths.app / "brutus.png")
        # view = toga.ImageView(my_image)

        return body

    def refresh_table_box(self):
        configs = self.picture_match_manager.app_config.configs
        self.table_box.clear()
        for i in range(len(configs)):
            config = configs[i]
            config_name = config.name
            region_str = ','.join([str(item) for item in config.region])
            region_in_full_screen_str = ','.join([str(item) for item in config.region_in_full_screen])
            db_picture_valid_region_str = ','.join([str(item) for item in config.db_picture_valid_region])
            auto_contraction_ratio_range_str = ','.join([str(item) for item in config.auto_contraction_ratio_range])
            row = toga.Box(
                id=config_name,
                style=Pack(padding_top=10),
                children=[
                    toga.Switch('', id=f'{config_name}_enable', value=config.enable,
                                style=Pack(width=self.table_box_column_widths[0], padding_left=self.cell_padding_left)),
                    toga.Label(config_name, style=Pack(width=self.table_box_column_widths[1], padding_left=self.cell_padding_left)),
                    toga.Label(config.window_name, style=Pack(width=self.table_box_column_widths[2], padding_left=self.cell_padding_left)),
                    toga.TextInput(id=f'{config_name}_region', value=region_str,
                                   style=Pack(width=self.table_box_column_widths[3], padding_left=self.cell_padding_left)),
                    toga.TextInput(id=f'{config_name}_region_in_full_screen', value=region_in_full_screen_str,
                                   style=Pack(width=self.table_box_column_widths[4], padding_left=self.cell_padding_left)),
                    toga.Box(children=[
                            toga.ImageView(id=f'{config_name}_screenshot_view', style=Pack(width=60, height=40))
                        ],
                        style=Pack(width=self.table_box_column_widths[5], padding_left=self.cell_padding_left)),
                    toga.TextInput(id=f'{config_name}_db_picture_valid_region', value=db_picture_valid_region_str,
                                   style=Pack(width=self.table_box_column_widths[6], padding_left=self.cell_padding_left)),
                    toga.TextInput(id=f'{config_name}_auto_contraction_ratio_range', value=auto_contraction_ratio_range_str,
                                   style=Pack(width=self.table_box_column_widths[7], padding_left=self.cell_padding_left)),
                    toga.TextInput(id=f'{config_name}_auto_contraction_ratio_step', value=str(config.auto_contraction_ratio_step),
                                   style=Pack(width=self.table_box_column_widths[8], padding_left=self.cell_padding_left)),
                    toga.TextInput(id=f'{config_name}_threadhold_match_rate', value=str(config.threadhold_match_rate),
                                   style=Pack(width=self.table_box_column_widths[9], padding_left=self.cell_padding_left))
                ]
            )
            self.table_box.add(row)

    def create_footer_box(self, footer_scroll_height):
        self.footer_box = toga.Box(style=Pack(direction=COLUMN, padding=self.padding))
        return self.footer_box

    async def refresh_ui_handler(self, widget=None, **kwargs):
        while True:
            self.do_refresh_footer_log(self.picture_match_manager.app_config.show_log_num)
            self.do_show_screenshot_image()
            await asyncio.sleep(1)

    def do_refresh_footer_log(self, log_nums=20):
        if self.get_app_config().enable_debug_mode:
            file_name = 'debug'
        else:
            file_name = 'info'
        logs = log_manager.get_log(file_name)
        new_show_logs = logs[(-log_nums - 1):]

        items = self.footer_box.children
        item_nums = len(items)

        i = 1
        for log in new_show_logs:
            if i > item_nums:
                label = toga.Label('')
                self.footer_box.add(label)
            else:
                label = items[i - 1]

            if log.find('检测到新版本') != -1:
                color = '#f44336'
                label.style.color = '#ffffff'
            elif log.find('[warn]') != -1:
                color = '#ffeb3b'
            else:
                color = '#ffffff'
            label.text = log
            label.style.background_color = color

            i = i + 1

    def handle_config_enable_switch(self, widget, **kwargs):
        app_config = self.get_app_config()
        for config in app_config.configs:
            if config.name == widget.id:
                config.enable = widget.value
        self.picture_match_manager.save_app_config(app_config)

    def do_show_screenshot_image(self):
        # 显示在软件界面上
        for config in self.get_app_config().configs:
            image_view = find_widget_by_id(f'{config.name}_screenshot_view')
            if image_view is not None:
                png_path = config.get_temp_folder() + symbol + config.name + ".png"
                if os.path.exists(png_path):
                    image_view.image = Image.open(png_path)


def main():
    """
    写代码时，可参考 github源码中的 examples 中的代码示例
    https://github.com/beeware/toga/blob/main/examples/handlers/handlers/app.py
    """
    return PictureMatchTool()

# TODO-high：计算全屏模式数据，测试全屏模式是否通用


# TODO-high：智能适配（全屏、窗口，以及能否 截个小图自动依据窗口截图去找目标区域）：
#            窗口比例不一样怎么办，如何适配不同的电脑（依据“预截图”自动在指定区域内扫描寻找截图位置？但要考虑“等比例放缩，以及不同比例的问题。。。。”）



# TODO-high： 等国服回来后，试试能不能申请调用官方的api获取图片数据  https://develop.battle.net/documentation/hearthstone/game-data-apis 。2024/08/30 09:37:01

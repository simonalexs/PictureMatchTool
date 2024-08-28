"""
my first
"""
from toga.style import Pack

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
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import threading
try:
    from importlib import metadata as importlib_metadata
except ImportError:
    # Backwards compatibility - importlib.metadata was added in Python 3.8
    import importlib_metadata


symbol = '/'


def cv2Pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil2Cv(img):
    # 直接用cv2.imread读取的“中文名称”的图片，在cv2.match_template方法中，会报错，所以用pil读取后转化
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cal_fittest_resize_rate(image_path, block_path):
    """
    临时工具，计算最合适的放缩比例，使模版最匹配（block 是 image 的子图）
    """
    start_factor = 0.1

    image = pil2Cv(Image.open(image_path))
    block = pil2Cv(Image.open(block_path))

    max_match_rate = (0, 0)
    for i in range(int(100 * start_factor), 101):
        factor = float(i) / 100
        height, width = image.shape[:2]
        resized_width = int(factor * width)
        resized_height = int(factor * height)
        if block.shape[:2][0] <= resized_height and block.shape[:2][1] <= resized_width:
            new_img = cv2.resize(image, [resized_width, resized_height])
            res = cv2.matchTemplate(new_img, block, cv2.TM_CCOEFF_NORMED)
            match_rate = np.amax(res)
            if match_rate > max_match_rate[1]:
                max_match_rate = (factor, match_rate)
    return max_match_rate


class LogManager:
    def __init__(self, app_path: Path):
        self.__app_path = app_path

    def get_app_folder(self):
        # self.__app_path.parent.parent 此为项目根目录
        return str(self.__app_path.parent.parent)

    def info(self, msg, config_name):
        self.__write_log('info', msg, config_name)

    def error(self, msg, config_name):
        self.__write_log('error', msg, config_name)
        self.__write_log('info', msg, config_name)

    def get_log_time(self, log_line: str):
        first = log_line.index(' ')
        second = log_line.index(' ', first + 1)
        return log_line[0:second]

    def get_log(self, type) -> list[str]:
        file_path = self.__get_log_path(type)
        result = []
        if not os.path.exists(file_path):
            return result
        with open(file_path, 'r') as file:
            line = file.readline()
            while line:
                line = file.readline()
                if line.endswith('\n'):
                    line = line.replace('\n', '')
                result.append(line)
        return result

    def __write_log(self, type, content, config_name):
        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        millisecondsInt = int((now - now.replace(microsecond=0)).total_seconds() * 1000)
        real_content = f"{now_str}.{millisecondsInt:03} [{threading.current_thread().name}] [{type}] [{config_name}] {content}"

        file_path = self.__get_log_path(type)
        with open(file_path, 'a') as file:
            print(real_content)
            file.write(real_content + '\n')
        # 日志文件过大时，减小文件体积
        if os.path.exists(file_path):
            file_size_byte = os.path.getsize(file_path)
            if file_size_byte >= 10 * 1024 * 1024:
                all_lines: list[str] = self.get_log(type)
                with open(file_path, 'w') as file:
                    file.write('\n'.join(all_lines[-2000:]))

    def __get_log_path(self, type):
        path = self.get_app_folder() + symbol + 'logs'
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = path + symbol + type + '.log'
        return file_path


class Config:
    def __init__(self, name, log_manager: LogManager):
        self.log_manager = log_manager

        self.name: str = name
        """配置名称"""

        self.description: str = ''

        self.enable: bool = False
        """配置名称"""

        self.window_name: str = ''
        """要监听的应用程序窗口 title"""

        self.region: tuple[int, int, int, int] = (0, 0, 0, 0)
        """游戏内的小图  [left距离左边, top距离上边, right距离左边, bottom距离上边]"""

        self.fixed_contraction_ratio: float = 1
        """小图与原图的放缩比例"""

        self.is_lock_contraction_ratio: bool = False
        """
        不固定时，会自动在收缩比范围内遍历寻找匹配度最高的结果
        固定时，只会以指定的 contraction_ratio 去寻找
        """

        self.auto_contraction_ratio_range: tuple[float, float] = (0.40, 1.00)
        """
        is_lock_contraction_ratio = False 时，自动以此收缩比范围去遍历寻找结果
        """

        self.auto_contraction_ratio_step: float = 0.01
        """收缩比范围的遍历步长"""

        self.result_target_folder: str = self.log_manager.get_app_folder() + symbol + 'data' + symbol + 'result'
        """
        结果文件存放的文件夹（默认会以配置名称作为文件名存放）
        """

    def get_cache_folder(self):
        path = self.log_manager.get_app_folder() + symbol + 'data' + symbol + 'cache' + symbol + self.name
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_temp_folder(self):
        path = self.log_manager.get_app_folder() + symbol + 'data' + symbol + 'temp'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_result_target_path(self):
        if not os.path.exists(self.result_target_folder):
            os.makedirs(self.result_target_folder)
        return self.result_target_folder + symbol + self.name + '.png'

    def get_real_region_by_config(self, config_region: tuple[int, int, int, int], window):
        key_left = int(window.width * config_region[0])
        key_top = int(window.height * config_region[1])
        key_right = int(window.width * config_region[2])
        key_bottom = int(window.height * config_region[3])
        return window.left + key_left, window.top + key_top, window.left + key_right, window.top + key_bottom

    def get_mouse_target_by_config(self, config_region: tuple[int, int, int, int], window):
        left, top, right, bottom = self.get_real_region_by_config(config_region, window)
        mouse_left = int((left + right) / 2)
        mouse_top = int((top + bottom) / 2)
        return mouse_left, mouse_top


class PictureMatchManager:
    def __init__(self, logManager: LogManager):
        self.log_manager = logManager

    def run_all_configs(self):
        start = time.time_ns() // 1000000
        for config in self.read_config():
            if config.enable:
                try:
                    t1 = time.time_ns() // 1000000
                    status, message = self.do_run_config(config)
                    t2 = time.time_ns() // 1000000
                    self.log_manager.info(message + '（耗时：' + str(t2 - t1) + ' ms', config.name)
                except Exception as e:
                    self.log_manager.error(str(e), config.name)
        end = time.time_ns() // 1000000
        self.log_manager.info(f"总耗时：{str(end - start)} ms", '所有配置')

    def do_cal_picture_match_rate(self, picture_block_path, img_file_path, contraction_ratio=1.0):
        """
        计算两个图片的最大匹配度，前者是否是后者的一部分
        """
        block = pil2Cv(Image.open(picture_block_path))
        img = pil2Cv(Image.open(img_file_path))
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

    def find_picture_in_db(self, block_picture_path, config: Config, threadhold_match_rate=0.8):
        cache_folder_path = config.get_cache_folder()
        all_file_path = common_utils.get_all_file_in_dir(cache_folder_path)
        if config.is_lock_contraction_ratio:
            self.log_manager.info('固定收缩比模式，当前收缩比：' + str(config.fixed_contraction_ratio), config.name)
            fittest = self.find_fittest_picture(block_picture_path, all_file_path, config.fixed_contraction_ratio, threadhold_match_rate)
            return fittest
        else:
            self.log_manager.info('自动查找模式，当前收缩比范围：' + str(config.auto_contraction_ratio_range)
                                  + ', 步长：' + str(config.auto_contraction_ratio_step), config.name)
            ratio_2_match_rate_2_path = self.find_highest_match_rate_picture(block_picture_path, all_file_path, threadhold_match_rate,
                                                                             config.auto_contraction_ratio_range,
                                                                             config.auto_contraction_ratio_step)
            return ratio_2_match_rate_2_path

    def find_highest_match_rate_picture(self, block_picture_path, all_file_path, threadhold_match_rate, contraction_ratio_range,
                                        contraction_ratio_step):
        """
        遍历收缩比，逐一寻找匹配度最高的图片
        """
        start_ratio_int = int(100 * contraction_ratio_range[0])
        end_ratio_int = int(100 * contraction_ratio_range[1])
        step = int(100 * contraction_ratio_step)

        ratio_2_match_rate_2_path = None
        for i in range(start_ratio_int, end_ratio_int + 1, step):
            contraction_ratio = float(i) / 100
            fittest = self.find_fittest_picture(block_picture_path, all_file_path, contraction_ratio, threadhold_match_rate)
            if fittest is None:
                continue
            if ratio_2_match_rate_2_path is None:
                ratio_2_match_rate_2_path = fittest
            elif fittest[1] > ratio_2_match_rate_2_path[1]:
                ratio_2_match_rate_2_path = fittest
        return ratio_2_match_rate_2_path

    def find_fittest_picture(self, block_picture_path, all_file_path, contraction_ratio, threadhold_match_rate):
        res_file_path = None
        res_rate = 0
        for file_path in all_file_path:
            if not file_path.endswith('.png'):
                continue
            rate = self.do_cal_picture_match_rate(block_picture_path, file_path, contraction_ratio)
            if rate > threadhold_match_rate and rate > res_rate:
                res_rate = rate
                res_file_path = file_path
        if res_file_path is None:
            return None
        return (contraction_ratio, res_rate, res_file_path)

    def read_config(self) -> list[Config]:
        configs = []
        little_config = Config('小饰品', self.log_manager)
        little_config.window_name = '炉石传说'
        little_config.enable = True
        little_config.region = [0.414, 0.798, 0.439, 0.833]
        little_config.fixed_contraction_ratio = 0.58
        # TODO-high: 测试，不锁定时，能否自动找到匹配度最高的图片
        little_config.is_lock_contraction_ratio = True
        little_config.auto_contraction_ratio_range = (0.50, 0.70)

        little_config.result_target_folder = self.log_manager.get_app_folder() + symbol + 'data' + symbol + 'result'
        configs.append(little_config)
        # 从 json 配置文件读取
        return configs

    def save_config(self, config_update: Config):
        configs = self.read_config()
        for config in configs:
            if config.name == config_update.name:
                config.__dict__ = copy.deepcopy(config_update.__dict__)
                break
        self.save_configs(configs)

    def save_configs(self, configs: list[Config]):
        # 写到 json 配置文件里
        pass

    def do_run_config(self, config):
        windows = gw.getWindowsWithTitle(config.window_name)
        if len(windows) == 0:
            return False, '未找到窗口[' + config.window_name + ']'
        res_windows = []
        for window in windows:
            if window.title == config.window_name:
                res_windows.append(window)
        if len(res_windows) == 0:
            return False, '未找到窗口[' + config.window_name + ']'
        if len(res_windows) > 1:
            return False, '找到' + str(len(res_windows)) + '个名叫[' + config.window_name + ']的窗口'
        window = res_windows[0]
        if window.isMinimized:
            return False, '窗口已最小化，无法截图'
        # 截取图片
        png_path = config.get_temp_folder() + symbol + config.name + '.png'
        real_region = config.get_real_region_by_config(config.region, window)
        image = ImageGrab.grab(real_region)
        image.save(png_path)

        # 从数据库中识别
        ratio_2_match_rate_2_path = self.find_picture_in_db(png_path, config, threadhold_match_rate=0.8)
        if ratio_2_match_rate_2_path is None:
            return False, '未找到结果'
        else:
            shutil.copyfile(ratio_2_match_rate_2_path[2], config.get_result_target_path())
            message = '已找到结果' + ratio_2_match_rate_2_path[2] + ', 收缩比：' + str(ratio_2_match_rate_2_path[0]) + ', 匹配度：' + str(ratio_2_match_rate_2_path[1])
            return True, message


class PictureMatchTool(toga.App):

    def __init__(self, **options):
        super().__init__(**options)
        self.scheduler = BackgroundScheduler(timezone='MST')
        self.log_manager = LogManager(self.app.paths.app)
        self.picture_match_manager = PictureMatchManager(self.log_manager)

        # Find the name of the module that was used to start the app
        app_module = sys.modules['__main__'].__package__
        # Retrieve the app's metadata
        self.metadata = importlib_metadata.metadata(app_module)

    def startup(self):
        """
        匹配配置（可以有多项配置）：
            窗口名
            【两种模式结合，且实时截图支持自动扫描，先尝试从缓存中获取结果图，没有的话再通过控制鼠标的方式获取并存入缓存中】
            本地库匹配模式，匹配本地图片库获取结果：
                目标识别区：[距离左边, 目标宽度, 距离上边, 目标宽度]
                    * 自动识别：根据“完整游戏窗口截图、目标位置截图”自动识别出“识别区” 
                    * 手动指定：可以手动指定“识别区”
                本地图片库（文件夹路径）
            实时截图模式，根据实时的悬浮图片获取结果：（只能手动模式，因为会抢夺鼠标的控制，加缓存以后缓存内的结果可以自动）
                鼠标放置区：[距离左边, 目标宽度, 距离上边, 目标宽度]
                    * 自动识别：根据“完整游戏窗口截图、目标位置截图”自动识别出“识别区” 
                    * 手动指定：可以手动指定“识别区”
                结果所在区域：[距离左边, 目标宽度, 距离上边, 目标宽度]
                    * 自动识别：根据“完整游戏窗口截图、结果位置截图”自动识别出“识别区” 
                    * 手动指定：可以手动指定“识别区”
            匹配结果图片存放的全路径（全路径，包括文件名和格式后缀，图片已存在时会自动覆盖）
            手动 or 自动（自动扫描间隔时间）
        """
        self.main_window = toga.MainWindow(title=self.formal_name + '-' + self.version, content=self.create_main_box())
        self.main_window.show()

        self.add_background_task(self.refresh_footer_log_handler)

    def create_main_box(self):
        self.main_box = toga.Box(style=Pack(padding=20))
        self.main_box.style.direction = COLUMN
        self.main_box.style.width = 1000

        # header
        header_button_box = self.create_header_button_box()
        self.main_box.add(header_button_box)

        # body
        body_scroll_container = toga.ScrollContainer(content=self.create_body_box())
        self.main_box.add(body_scroll_container)
        # footer
        footer_scroll_container = toga.ScrollContainer(content=self.create_footer_box())
        footer_scroll_container.style.width = 1000
        footer_scroll_container.style.height = 200
        self.main_box.add(footer_scroll_container)

        return self.main_box

    def create_header_button_box(self):
        box = toga.Box()
        # add_config_button = toga.Button('添加配置', on_press=self.add_config_button_handler())
        # box.add(add_config_button)
        # box.add(toga.Button('识别全部'))

        self.start_all_configs_btn = toga.Button('开始识别', on_press=self.start_all_configs_btn_handler)
        box.add(self.start_all_configs_btn)
        self.stop_all_configs_btn = toga.Button('停止识别', on_press=self.stop_all_configs_btn_handler)
        box.add(self.stop_all_configs_btn)
        return box

    def start_all_configs_btn_handler(self, widget, **kwargs):
        self.scheduler.add_job(self.picture_match_manager.run_all_configs, 'interval', seconds=5, id='job_all',
                               replace_existing=True)
        if not self.scheduler.running:
            self.scheduler.start()
        self.start_all_configs_btn.enabled = False
        self.stop_all_configs_btn.enabled = True
        self.log_manager.info('开始识别', 'configs')

    def stop_all_configs_btn_handler(self, widget, **kwargs):
        self.scheduler.remove_job('job_all')
        self.start_all_configs_btn.enabled = True
        self.stop_all_configs_btn.enabled = False
        self.log_manager.info('已停止识别', 'configs')

    def create_body_box(self):
        box = toga.Box()

        return box

    def create_footer_box(self):
        self.footer_box = toga.Box()
        self.footer_box.style.direction = COLUMN
        return self.footer_box

    async def refresh_footer_log_handler(self, widget, **kwargs):
        while True:
            await self.do_refresh_footer_log()
            await asyncio.sleep(1)

    async def do_refresh_footer_log(self, log_nums=10):
        logs = self.log_manager.get_log('info')
        new_show_logs = logs[(-log_nums - 1):]

        labels = self.footer_box.children
        label_nums = len(labels)

        i = 1
        for log in new_show_logs:
            if i > label_nums:
                self.footer_box.add(toga.Label(log))
            else:
                labels[i - 1].text = log
            i = i + 1

    def add_config_button_handler(self):
        pass


def main():
    """
    写代码时，可参考 github源码中的 examples 中的代码示例
    https://github.com/beeware/toga/blob/main/examples/handlers/handlers/app.py
    """
    return PictureMatchTool()
import json
import os
import sys
import mss
import requests
from PIL import ImageGrab
import pygetwindow as gw
from PIL import Image

from src.picture_match_tool.app import CacheEntity, AppConfig
from src.picture_match_tool.utils import common_utils

def sc(region):
    with mss.mss() as sct:
        width = region[2] - region[0]
        height = region[3] - region[1]
        screenshot = sct.grab({"left": region[0], "top": region[1], "width": width, "height": height})
        pim = Image.new("RGB", screenshot.size)
        pim.frombytes(screenshot.rgb)
        return pim

url = 'https://github.com/simonalexs/PictureMatchTool/releases'

name = '钉钉'
windows = gw.getWindowsWithTitle(name)
if len(windows) == 0:
    print ('未找到窗口[' + name + ']')
res_windows = []
for window in windows:
    if name == window.title:
        res_windows.append(window)
window = res_windows[0]
print(window.left, window.top, window.right, window.bottom)
print(window.width, window.height)
# ImageGrab.grab((0, 0, window.right, window.bottom)).show('0-0-right-bottom')
# ImageGrab.grab((window.left, window.top, window.right, window.bottom)).show('0-0-right-bottom')

# sc((window.left, window.top, window.right, window.bottom)).show('0-0-right-bottom')



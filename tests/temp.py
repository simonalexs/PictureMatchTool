import json
import os
import sys

from src.picture_match_tool.app import CacheEntity, AppConfig
from src.picture_match_tool.utils import common_utils

# print(os.path.isfile('C:/Users/SimonAlexs/Desktop'))

# path = os.readlink('C:/Users/SimonAlexs/Desktop/项目-4-上融.lnk')
# print(path)
# print(os.listdir(path))
a = [0,1,2,3,4,5,6]
print(a[-3:])
print(a[-7:])
print(a[-10:])


for i in range(3):
    print(i)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, CacheEntity):
            return obj.__dict__
        return json.JSONEncoder.default(self, obj)

p: list[CacheEntity] = [CacheEntity('1', 1, 1, 1, '2')]
print(json.dumps(p, cls=CustomEncoder))
# print(json.dumps(p))


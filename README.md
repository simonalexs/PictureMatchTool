# PictureMatchTool
图片匹配工具（windows软件），依据指定区域的截图从匹配库中找到相应的图片，并存放到指定位置（最初的想法是为主播做的，用于炉石传说酒馆战棋模式中，自动识别出所选技能或饰品的图片详情）

## 数据库图片
### 图片格式必须为 png
### 图片处理
从旅法师营地的文章中保存饰品原图片，用 tests/utils/cut_pictures_util.py 裁剪图片，得到最终有效的数据库中的图片。

## 项目开发步骤（开发人员使用）
### 开发环境
* python 3.11（更高版本应该也可以）
* wix toolset 3.11 (使用命令安装：dotnet tool install --global wix)
* git
* pip install briefcase
### 命令
* briefcase dev 本地运行
* briefcase create 当新引入了依赖包时，使用此命令从而使打包时能把新依赖包打进去
* briefcase update
* briefcase build
* 打包：
  - msi格式：briefcase package，或者 briefcase package windows -p msi
  - zip格式：briefcase package windows -p zip
### 相关网址
* 打包说明
https://github.com/beeware/briefcase/blob/main/docs/reference/platforms/windows/app.rst
* 软件icon说明
https://docs.beeware.org/zh-cn/latest/tutorial/topics/custom-icons.html
* 代码示例
https://github.com/beeware/toga/blob/main/examples/handlers/handlers/app.py
* toga 组件样式介绍
https://toga.readthedocs.io/en/latest/reference/api/containers/scrollcontainer.html
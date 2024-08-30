# PictureMatchTool
图片匹配工具（windows软件），依据指定区域的截图从图片库中找到相应的图片，并存放到指定位置（最初的想法是为主播做的，用于炉石传说酒馆战棋模式中，自动识别出所选技能或饰品的图片详情）

## 一、使用说明
### 1.1 功能介绍
* 根据指定的电脑上软件窗口中的矩形区域，对其截图并从图片库中找到对应的原图，并将找到的结果存放到指定位置
  - 【已适配炉石传说酒馆战棋的“小饰品、大饰品”】

### 1.2 数据库图片来源
由于没有找到炉石官方的渠道获取图片，所以目前的饰品图是从 [旅法师营地@Bennidge](https://www.iyingdi.com/tz/people/55547) 的“饰品一览”中手动保存来的。

**特别说明：目前软件功能应该问题不大，难解决的是图片库的自动维护问题，因为饰品效果设计师会不断调整，目前只能手动维护
（[暴雪官网](https://develop.battle.net/documentation/hearthstone/game-data-apis)虽然有api介绍，但我用账号登录时
好像提示什么需要验证器，估计是需要梯子之类的吧，也找了找其它网站，貌似没有这种api，不知道国服回来能不能有api可以使用）**

## 二、项目开发步骤（开发人员使用）
### 2.1 开发环境
* python 3.11（更高版本应该也可以）
* wix toolset 3.11 (使用命令安装：dotnet tool install --global wix)
* git
* pip install briefcase
### 2.2 命令
* briefcase dev 本地运行
* briefcase create 当新引入了依赖包时，使用此命令从而使打包时能把新依赖包打进去
* briefcase update
* briefcase build
* 打包：
  - msi格式：briefcase package，或者 briefcase package windows -p msi
  - zip格式：briefcase package windows -p zip

### 2.3 图片处理
若数据库图片需要裁剪等处理，可使用 tests/utils/cut_pictures_util.py 裁剪图片，得到最终有效的数据库中的图片。
### 2.4 相关网址
* 打包说明
https://github.com/beeware/briefcase/blob/main/docs/reference/platforms/windows/app.rst
* 软件icon说明
https://docs.beeware.org/zh-cn/latest/tutorial/topics/custom-icons.html
* 代码示例
https://github.com/beeware/toga/blob/main/examples/handlers/handlers/app.py
* toga 组件样式介绍
https://toga.readthedocs.io/en/latest/reference/api/containers/scrollcontainer.html
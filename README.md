# PictureMatchTool
图片匹配工具（windows软件），依据指定区域的截图从图片库中找到相应的图片，并存放到指定位置（最初的想法是为主播做的，用于炉石传说酒馆战棋模式中，
自动识别出所选技能或饰品的图片详情，再配合直播软件添加“图片源”，就可以实现“游戏内选择了小饰品、大饰品后自动在obs中显示饰品介绍”的功能）

## 一、使用说明
### 1.1 功能介绍
* 根据指定的电脑上软件窗口中的矩形区域，对其截图并从图片库中找到对应的原图，并将找到的结果存放到指定位置
  - 【已适配炉石传说酒馆战棋的“小饰品、大饰品”】
### 1.2 使用说明（应该只有主播会用到吧）
* 下载右方“Release”处的最新版本（zip格式），解压后双击“PictureMatchTool.exe”打开软件；
* 点击左上方“开始自动识别”按钮；
* 在obs中（其它直播软件或许也可以吧）添加“图片源”，并选择“本软件界面中所显示的图片路径”所对应的图片。

然后就可以打开炉石开始游玩了，本软件会每隔几秒自动对“饰品技能所在区域”截图，并尝试匹配“饰品”所对应的原图然后存放到指定位置（也就是obs选择的路径，
obs会自动实时显示更新后的图片）。
### 1.3 软件安全性说明
本软件所做的操作不访问任何用户数据，也从未访问或修改任何游戏数据，请放心使用。

本软件所做的操作：
* **获取窗口状态**：通过python的windows工具包获取“炉石传说”窗口的状态（是否有此窗口、该窗口是否最小化、该窗口是否已激活）；
* **截图**：依据配置好的“饰品”技能所在区域，用python工具包对指定区域截图并保存在软件目录的“/app/data/temp/”下；
* **图片匹配**：将截图与图片库中的图片做匹配，筛选出匹配度最高的图片，存放在“/app/data/result”下，供用户使用。
* **检查软件更新**：软件启动时会自动访问一次本github的release页面，检查是否有新的版本。

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

### 2.5 数据库图片来源
由于没有找到炉石官方的渠道获取图片，所以目前的饰品图是从 [旅法师营地@Bennidge](https://www.iyingdi.com/tz/people/55547) 的“饰品一览”中手动保存来的。

**特别说明：目前软件功能应该问题不大，难解决的是图片库的自动维护问题，因为饰品效果设计师会不断调整，目前只能手动维护
（[暴雪官网](https://develop.battle.net/documentation/hearthstone/game-data-apis)虽然有api介绍，但我用账号登录时
好像提示什么需要验证器，估计是需要梯子之类的吧，也找了找其它网站，貌似没有这种api，不知道国服回来能不能有api可以使用）**
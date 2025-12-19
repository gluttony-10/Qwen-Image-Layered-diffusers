# Qwen-Image-Layered-diffusers
Qwen-Image-Layered的diffusers版本

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 使用需求
1.显卡支持BF16

2.显存大于4G

## 安装依赖
```
git clone https://github.com/gluttony-10/Qwen-Image-Layered-diffusers
cd Qwen-Image-Layered-diffusers
conda create -n Qwen-Image-Layered-diffusers python=3.12
conda activate Qwen-Image-Layered-diffusers
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
## 下载模型
```
modelscope download --model Gluttony10/Qwen-Image-Layered-diffusers --local_dir ./models
```
## 开始运行
```
python glut.py
```
## 参考项目
https://modelscope.cn/models/Qwen/Qwen-Image-Layered



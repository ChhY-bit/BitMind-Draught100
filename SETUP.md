# 环境配置指南

本项目只有一个外部 Python 依赖：**PyTorch**。其他全部使用标准库（`ctypes`、`tkinter`、`json`、`pathlib` 等）。

## 方式一：Conda（推荐）

```bash
# 从 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate draughts

# 编译 C 共享库
make interface
```

## 方式二：pip + venv

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 编译 C 共享库
make interface
```

## 方式三：系统 Python（Linux）

```bash
# 安装 PyTorch（CPU 版）
pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# 编译
make interface
```

## 验证

```bash
# 接口测试（不依赖 torch）
make interface-test

# 全部 Python 测试（需要 torch）
make draughts-test
make trainer-test
make scripts-test

# 启动 GUI
PYTHONPATH=interface:. python3 gui/simple_gui.py
```

## 依赖清单

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | ≥2.0 | 神经网络模型、训练 |
| C 编译器 (gcc/clang) | — | 编译 C 引擎为 `libdraughts.so` |
| `tkinter` | — | GUI（通常随 Python 自带） |

`tkinter` 在少数 Linux 发行版上需要单独安装：

```bash
# Ubuntu/Debian
sudo apt install python3-tk

# CentOS/Fedora
sudo dnf install python3-tkinter
```

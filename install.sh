#!/bin/bash

# 设置目标目录
INSTALL_DIR="/usr/local/sbin"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 确保目标目录存在
sudo mkdir -p $INSTALL_DIR

# 复制所有相关的 Python 文件
sudo cp "$SCRIPT_DIR/paper2md.py" "$INSTALL_DIR/paper2md"
sudo cp "$SCRIPT_DIR/arxiv_extract.py" "$INSTALL_DIR/"
sudo cp "$SCRIPT_DIR/ieee_extract.py" "$INSTALL_DIR/"

# 设置执行权限
sudo chmod +x "$INSTALL_DIR/paper2md"
sudo chmod 644 "$INSTALL_DIR/arxiv_extract.py"
sudo chmod 644 "$INSTALL_DIR/ieee_extract.py"

echo "安装完成！现在可以在任何目录使用 'paper2md' 命令" 
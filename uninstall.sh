#!/bin/bash

# 设置目标目录
INSTALL_DIR="/usr/local/sbin"

# 删除所有相关文件
sudo rm -f "$INSTALL_DIR/paper2md"
sudo rm -f "$INSTALL_DIR/arxiv_extract.py"
sudo rm -f "$INSTALL_DIR/ieee_extract.py"

echo "卸载完成！" 
#!/usr/bin/env python3

import argparse
from datetime import datetime
import re
from urllib.parse import urlparse
import os
import sys

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 将脚本目录添加到系统路径
sys.path.append(SCRIPT_DIR)

from arxiv_extract import extract_arxiv_paper
from ieee_extract import extract_ieee_paper

def get_output_filename():
    """生成基于当前时间的输出文件名"""
    now = datetime.now()
    return now.strftime('%Y%m%d_%H%M%S.md')

def is_arxiv_url(url):
    """判断是否为 arXiv URL"""
    parsed = urlparse(url)
    return 'arxiv' in parsed.netloc.lower() or 'ar5iv' in parsed.netloc.lower()

def is_ieee_url(url):
    """判断是否为 IEEE URL"""
    parsed = urlparse(url)
    return 'ieee' in parsed.netloc.lower()

def convert_paper(url):
    """根据URL类型选择相应的转换器"""
    output_file = get_output_filename()
    
    try:
        if is_arxiv_url(url):
            print(f"检测到 arXiv 论文，开始转换...")
            extract_arxiv_paper(url, output_file)
        elif is_ieee_url(url):
            print(f"检测到 IEEE 论文，开始转换...")
            extract_ieee_paper(url, output_file)
        else:
            raise ValueError("不支持的URL格式。目前仅支持 arXiv 和 IEEE 论文。")
        
        print(f"转换完成！文件已保存为: {output_file}")
        
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='将学术论文转换为Markdown格式')
    parser.add_argument('url', help='论文URL (支持 arXiv 和 IEEE)')
    args = parser.parse_args()
    
    convert_paper(args.url)

if __name__ == "__main__":
    main() 
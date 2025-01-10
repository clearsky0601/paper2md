from bs4 import BeautifulSoup
import requests
import re

def process_math_formula(element):
    """处理数学公式，将其转换为LaTeX格式"""
    try:
        # 获取 LaTeX 代码
        tex = element.find('annotation', {'encoding': 'application/x-tex'})
        if tex:
            latex_code = tex.text.strip()
            # 检查是否是行间公式
            if element.get('display') == 'block' or 'ltx_equation' in element.parent.get('class', []):
                return f"\n$${latex_code}$$\n\n"
            else:
                return f"${latex_code}$"
    except Exception as e:
        print(f"处理公式时出错: {str(e)}")
        return element.get_text()

def process_section_title(element):
    """处理章节标题，转换为对应级别的markdown标题"""
    try:
        # 获取标题级别
        classes = element.get('class', [])
        level = 1  # 默认级别
        
        if 'ltx_title_document' in classes:
            level = 1
        elif 'ltx_title_section' in classes:
            level = 2
        elif 'ltx_title_subsection' in classes:
            level = 3
        elif 'ltx_title_subsubsection' in classes:
            level = 4
        elif 'ltx_title_paragraph' in classes:
            level = 5
            
        title_text = element.get_text().strip()
        return f"{'#' * level} {title_text}\n\n"
    except Exception as e:
        print(f"处理标题时出错: {str(e)}")
        return ""

def process_paragraph(element):
    """处理段落内容，包括文本和公式"""
    try:
        content = []
        # 使用集合记录已处理的公式
        processed_formulas = set()
        
        for child in element.children:
            if child.name == 'math':
                # 只处理行内公式，行间公式由外层处理
                if child.get('display') != 'block':
                    formula_id = child.get('id', '')
                    if formula_id not in processed_formulas:
                        processed_formulas.add(formula_id)
                        content.append(process_math_formula(child))
            elif child.name == 'h4' and 'ltx_title_paragraph' in child.get('class', []):
                content.append('\n' + process_section_title(child))
            elif isinstance(child, str):
                content.append(child.strip())
        return ' '.join(filter(None, content)).strip()
    except Exception as e:
        print(f"处理段落时出错: {str(e)}")
        return ""

def process_equation(element):
    """处理独立的数学公式"""
    try:
        # 获取公式内容
        math = element.find('math')
        if math:
            tex = math.find('annotation', {'encoding': 'application/x-tex'})
            if tex:
                latex_code = tex.text.strip()
                return f"\n$${latex_code}$$\n\n"
    except Exception as e:
        print(f"处理公式时出错: {str(e)}")
    return ""

def extract_arxiv_paper(url):
    """从ar5iv提取论文内容并转换为markdown格式"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content = []
        # 使用集合记录已处理的公式
        processed_equations = set()
        
        # 提取标题
        title = soup.find('h1', class_='ltx_title_document')
        if title:
            content.append(process_section_title(title))
            
        # 提取作者和机构信息
        authors = soup.find_all('div', class_='ltx_authors')
        if authors:
            for author in authors:
                content.append(author.get_text().strip() + "\n\n")
                
        # 处理主要内容
        main_content = soup.find('div', class_='ltx_page_main')
        if main_content:
            for element in main_content.find_all(['div', 'p', 'table', 'h2', 'h3', 'h4']):
                if 'ltx_title' in element.get('class', []):
                    content.append('\n' + process_section_title(element))
                elif element.name == 'table' and 'ltx_equation' in element.get('class', []):
                    # 处理行间公式（equation环境）
                    equation_id = element.get('id', '')
                    if equation_id not in processed_equations:
                        processed_equations.add(equation_id)
                        equation_content = process_equation(element)
                        if equation_content:
                            content.append(equation_content)
                elif element.name == 'p':
                    text = process_paragraph(element)
                    if text:
                        content.append(text + "\n\n")
                elif element.find('img'):
                    img = element.find('img')
                    caption = element.find('figcaption')
                    caption_text = caption.get_text().strip() if caption else ''
                    content.append(f"\n![{caption_text}]({img['src']})\n\n")
        
        # 保存为markdown文件
        with open('paper.md', 'w', encoding='utf-8') as f:
            f.writelines(content)
        print("内容已保存到 paper.md")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    url = "https://ar5iv.labs.arxiv.org/html/2105.00243"
    extract_arxiv_paper(url) 
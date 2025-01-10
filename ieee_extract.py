from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import re

def process_math_formula(element):
    """处理数学公式，将其转换为LaTeX格式"""
    try:
        # 检查公式类型
        if element.find_elements(By.CSS_SELECTOR, "script[type='math/tex; mode=display']"):
            # 行间公式
            script = element.find_element(By.CSS_SELECTOR, "script[type='math/tex; mode=display']")
            tex = script.get_attribute('innerHTML').strip()
            return f"\n$${tex}$$\n"
        elif element.find_elements(By.CSS_SELECTOR, "script[type='math/tex']"):
            # 行内公式
            script = element.find_element(By.CSS_SELECTOR, "script[type='math/tex']")
            tex = script.get_attribute('innerHTML').strip()
            return f"${tex}$"
    except Exception as e:
        print(f"处理公式时出错: {str(e)}")
    return element.text

def process_image(element):
    """处理图片，保留原始URL"""
    try:
        img = element.find_element(By.CSS_SELECTOR, "img.img-responsive")
        src = img.get_attribute('src')
        alt = img.get_attribute('alt') or ''
        
        # 尝试获取图片标题
        try:
            caption = element.find_element(By.CSS_SELECTOR, "figcaption").text
            return f"\n![{alt}]({src})\n*{caption}*\n"
        except:
            return f"\n![{alt}]({src})\n"
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return ''

def process_paragraph(element):
    """处理段落内容，包括文本和公式"""
    try:
        # 使用JavaScript获取所有子节点
        script = """
        function getTextAndFormulas(element) {
            const result = [];
            for (const node of element.childNodes) {
                if (node.nodeType === Node.TEXT_NODE) {
                    if (node.textContent.trim()) {
                        result.push({type: 'text', content: node.textContent});
                    }
                } else if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.classList.contains('inline-formula') || node.classList.contains('display-formula')) {
                        result.push({type: 'formula', element: node});
                    } else {
                        result.push({type: 'text', content: node.textContent});
                    }
                }
            }
            return result;
        }
        return getTextAndFormulas(arguments[0]);
        """
        nodes = element.parent.execute_script(script, element)
        
        text_parts = []
        for node in nodes:
            if node['type'] == 'text':
                text_parts.append(node['content'].strip())
            elif node['type'] == 'formula':
                formula_text = process_math_formula(node['element'])
                text_parts.append(formula_text)
        
        return ' '.join(filter(None, text_parts))
    except Exception as e:
        print(f"处理段落时出错: {str(e)}")
        return element.text

def extract_ieee_paper(url, output_file='paper.md'):
    """从IEEE提取论文内容并转换为markdown格式"""
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    try:
        print("正在初始化 ChromeDriver...")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        print("正在访问页面...")
        driver.get(url)
        time.sleep(5)  # 等待页面加载
        
        # 等待页面主要内容加载完成
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.document-main"))
        )
        
        content = []
        
        # 提取标题
        title = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h1.document-title"))
        ).text
        content.append(f"# {title}\n\n")
        print(f"已提取标题: {title}")
        
        # 提取摘要
        try:
            abstract = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.abstract-text"))
            )
            abstract_text = process_paragraph(abstract)
            content.append(f"## Abstract\n\n{abstract_text}\n\n")
            print("已提取摘要")
        except Exception as e:
            print(f"提取摘要时出错: {str(e)}")
        
        # 提取正文
        sections = driver.find_elements(By.CSS_SELECTOR, "div.section")
        for section in sections:
            try:
                # 提取章节标题
                section_title = section.find_element(By.CSS_SELECTOR, "h2").text
                content.append(f"\n## {section_title}\n\n")
                
                # 提取段落、公式、图片等
                for element in section.find_elements(By.XPATH, ".//*"):
                    try:
                        if element.tag_name == 'p':
                            text = process_paragraph(element)
                            if text:
                                content.append(f"{text}\n\n")
                        elif 'image' in (element.get_attribute('class') or ''):
                            img_text = process_image(element)
                            if img_text:
                                content.append(img_text)
                    except:
                        continue
            except Exception as e:
                print(f"处理章节时出错: {str(e)}")
                continue
        
        # 保存内容
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(content)
        print(f"内容已保存到 {output_file}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        driver.quit()

if __name__ == "__main__":
    url = "https://ieeexplore.ieee.org/document/10484206"
    # url = "https://ieeexplore.ieee.org/document/10484206"
    extract_ieee_paper(url)

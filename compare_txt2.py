import numpy as np
from scipy.spatial.distance import cosine
import argparse

def load_numbers(file_path):
    """
    加载文本文件中的数字，处理行排布和列排布
    """
    with open(file_path, 'r') as f:
        # 读取所有内容并分割
        content = f.read().strip()
        # 尝试不同的分隔符
        if '\n' in content:
            numbers = content.split('\n')
        elif '\t' in content:
            numbers = content.split('\t')
        else:
            numbers = content.split()
        
        # 转换为浮点数
        return np.array([float(n) for n in numbers if n.strip()])

def calculate_cosine_similarity(file1_path, file2_path):
    """
    计算两个文件中数字的余弦相似度
    """
    try:
        # 加载两个文件的数字
        vector1 = load_numbers(file1_path)
        vector2 = load_numbers(file2_path)
        
        # 检查维度
        print(f"\n向量维度:")
        print(f"文件1 ({file1_path}): {len(vector1)}维")
        print(f"文件2 ({file2_path}): {len(vector2)}维")
        
        if len(vector1) != len(vector2):
            print(f"警告：向量维度不匹配！")
            return None
        
        # 计算余弦相似度
        similarity = cosine(vector1, vector2)
        return similarity
        
    except Exception as e:
        print(f"错误：{str(e)}")
        return None

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='计算两个向量文件的余弦相似度')
    parser.add_argument('file1', type=str, help='第一个向量文件的路径')
    parser.add_argument('file2', type=str, help='第二个向量文件的路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 计算相似度
    similarity = calculate_cosine_similarity(args.file1, args.file2)
    
    if similarity is not None:
        print(f"\n余弦相似度: {similarity:.6f}")
        print(f"相似度百分比: {similarity * 100:.2f}%")

if __name__ == "__main__":
    main()
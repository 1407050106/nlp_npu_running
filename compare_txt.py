import numpy as np
from scipy.spatial.distance import cosine
import sys

def load_txt(file_path, last_n=64):
    """加载txt文件并返回最后last_n个元素"""
    try:
        # 加载数据
        data = np.loadtxt(file_path, dtype=np.float32)
        # 展平并获取最后last_n个元素
        flattened_data = data.flatten()
        if len(flattened_data) < last_n:
            print(f"警告: 数据长度({len(flattened_data)})小于要求的维度({last_n})")
            return flattened_data
        return flattened_data[-last_n:]
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def compare_files(file1, file2, last_n=64):
    """比较两个文件中最后last_n个数据的余弦相似度"""
    # 加载数据
    data1 = load_txt(file1, last_n)
    data2 = load_txt(file2, last_n)
    
    # 检查维度
    if len(data1) != len(data2):
        print(f"维度不匹配: file1={len(data1)}, file2={len(data2)}")
        return

    similarity = cosine(data1, data2)
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print(f"数据维度： {last_n}")
    print(f"余弦相似度: {similarity:.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("使用方法: python compare_txt.py file1.txt file2.txt")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    compare_files(file1, file2, last_n=64)
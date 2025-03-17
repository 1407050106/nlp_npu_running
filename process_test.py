import numpy as np
import torch
import onnxruntime as ort
import time
from typing import List, Tuple, Optional

def process_sequence(
    data: np.ndarray, 
    model_path: str, 
    chunk_size: int = 16, 
    batch_size: int = 8
) -> Tuple[np.ndarray, float]:
    """处理输入序列
    
    Args:
        data: 输入数据, shape为[seq_len, features]
        model_path: ONNX模型路径
        chunk_size: 每个chunk的大小
        batch_size: 批处理大小
    
    Returns:
        results: 处理结果
        process_time: 处理时间
    """
    try:
        # 初始化ONNX运行时
        ort_session = ort.InferenceSession(model_path)
        start_time = time.time()
        
        # 基础参数计算
        total_length = data.shape[0]
        num_full_chunks = total_length // chunk_size
        remaining = total_length % chunk_size
        
        chunks = []
        results = []
        
        # 处理完整的chunks
        for i in range(num_full_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            chunk = data[start_idx:end_idx].T[np.newaxis, :, :]  # [1, 40, 16]
            chunks.append(chunk)
            
            # 当累积到batch_size或是最后一个chunk时进行批处理
            if len(chunks) == batch_size or i == num_full_chunks - 1:
                batch = np.concatenate(chunks, axis=0)
                output = ort_session.run(None, {'input': batch.astype(np.float32)})[0]
                results.extend(output)
                chunks = []
        
        # 处理剩余的数据
        if remaining > 0:
            last_chunk = np.zeros((chunk_size, data.shape[1]))
            last_chunk[:remaining] = data[-remaining:]
            last_chunk = last_chunk.T[np.newaxis, :, :]
            output = ort_session.run(None, {'input': last_chunk.astype(np.float32)})[0]
            results.append(output[0])
        
        process_time = time.time() - start_time
        return np.vstack(results), process_time
    
    except Exception as e:
        print(f"Error in process_sequence: {str(e)}")
        raise

def save_results(
    results: np.ndarray, 
    output_path: str, 
    process_time: float
) -> None:
    """保存处理结果
    
    Args:
        results: 处理结果
        output_path: 输出文件路径
        process_time: 处理时间
    """
    try:
        np.savetxt(
            output_path,
            results,
            fmt='%.6f',
            delimiter='\n',
            # header=f'Processing time: {process_time:.4f} seconds'
        )
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # 配置参数
        INPUT_PATH = '/Users/wangyonglin/Desktop/nnlp/input_data.txt'
        MODEL_PATH = '/Users/wangyonglin/Desktop/nnlp/gather_test/simplified_model.onnx'
        OUTPUT_PATH = '/Users/wangyonglin/Desktop/nnlp/output_results.txt'
        CHUNK_SIZE = 16
        BATCH_SIZE = 8
        
        # 加载数据
        data = np.loadtxt(INPUT_PATH, dtype=np.float32)
        print(f"Input data shape: {data.shape}")
        
        # 处理序列
        results, process_time = process_sequence(
            data, 
            MODEL_PATH, 
            CHUNK_SIZE, 
            BATCH_SIZE
        )
        print(f"Processing time: {process_time:.4f} seconds")
        print(f"Output shape: {results.shape}")
        
        # 保存结果
        save_results(results, OUTPUT_PATH, process_time)
        print(f"Results saved to: {OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
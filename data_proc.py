import torch

torch.manual_seed(0)

num_channels = 40
seq_length = 500

input_tensor = torch.rand(seq_length, num_channels) * 10  # 生成 0-10 之间的随机浮点数

with open('/Users/wangyonglin/Desktop/nnlp/input_data.txt', 'w') as f:
    for i in range(seq_length):
        data_line = ' '.join(f'{val:.6f}' for val in input_tensor[i].tolist())
        f.write(data_line + '\n')  # 每行数据输出

print("Input data has been saved to 'input_data.txt'.")
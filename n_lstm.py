import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify
from scipy.spatial.distance import cosine

class SampleNetwork(nn.Module):
    def __init__(self):
        super(SampleNetwork, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=40, out_channels=192, kernel_size=3, padding=2, stride=1)

        # # 1st LSTM 层
        self.lstm1 = nn.LSTM(input_size=192, hidden_size=128, batch_first=True, bidirectional=False)       
        # # # 1st LayerNorm
        self.layer_norm1 = nn.LayerNorm(128, eps=1e-05, elementwise_affine=True)       
        # # # 2nd LSTM 层
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)        
        # # # 2nd LayerNorm
        self.layer_norm2 = nn.LayerNorm(128, eps=1e-05, elementwise_affine=True)        
        # # # 3rd LSTM 层
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)       
        # # # 3rd LayerNorm
        self.layer_norm3 = nn.LayerNorm(128, eps=1e-05, elementwise_affine=True)

        self.fc = nn.Linear(in_features=128, out_features=64, bias=True)

    def forward(self, x):
        x = self.conv1d(x)     #  x shape: (batch_size, 40, seq_length) 
        # x = x.reshape(x.shape[0], -1)  
        x = x.transpose(1, 2)  #  (batch_size, seq_length, 192)

        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)
        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)
        x, _ = self.lstm3(x)
        x = self.layer_norm3(x)
        x = x[:, -1, :]  # last time step
        
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = SampleNetwork()
    torch.save(model, '/Users/wangyonglin/Desktop/nnlp/gather_test/sample_network.pth')
    model.eval()

    # input_tensor = torch.randn(1, 40, 16)
    # data = np.loadtxt('/Users/wangyonglin/Desktop/nnlp/input_data.txt', dtype=np.float32)  # [500, 40]
    # data = data.T[np.newaxis, :, :]  # [1, 40, 500]
    # data = data[:, :, :16]  # [1, 40, 16]
    # data = np.repeat(data, 5, axis=0)
    # input_tensor = torch.from_numpy(data)
    data_list = []
    for i in range(5):
        start_idx = i * 16  # 每个批次读取16个时间步
        data = np.loadtxt('/Users/wangyonglin/Desktop/nnlp/input_data.txt', dtype=np.float32)  # [500, 40]
        data = data.T[np.newaxis, :, start_idx:start_idx+16]  # [1, 40, 16]
        data_list.append(data)
    
    # 合并5个批次
    data = np.concatenate(data_list, axis=0)  # [5, 40, 16]
    input_tensor = torch.from_numpy(data)
    output_pytorch = model(input_tensor)
    print("PyTorch output shape:", output_pytorch.shape)

    onnx_file_path = "/Users/wangyonglin/Desktop/nnlp/gather_test/sample_network.onnx"
    torch.onnx.export(model, 
                      input_tensor,
                      onnx_file_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={
                          'input': {0: 'batch_size'},  # 第0维为动态批次
                          'output': {0: 'batch_size'}  # 输出的第0维也对应动态批次
                      })

    print(f"Model saved to {onnx_file_path}")

    # ort_session = ort.InferenceSession(onnx_file_path)
    ort_session = ort.InferenceSession("/Users/wangyonglin/Desktop/nnlp/gather_test/sample_network.onnx")
    input_onnx = input_tensor.numpy()
    output_onnx = ort_session.run(None, {ort_session.get_inputs()[0].name: input_onnx})[0]

    output_pytorch_sample = output_pytorch[0].detach().numpy()
    output_onnx_sample = output_onnx[0]

    output_file = "/Users/wangyonglin/Desktop/nnlp/output_onnx.txt"
    np.savetxt(
        output_file, 
        output_onnx[0],
        fmt='%.6f',
        delimiter='\n'
    )
    print(f"ONNX output saved to {output_file}")
    
    pytorch_output_file = "/Users/wangyonglin/Desktop/nnlp/output_pytorch.txt"
    np.savetxt(
        pytorch_output_file,
        output_pytorch_sample,
        fmt='%.6f',
        delimiter='\n'
    )
    print(f"PyTorch output saved to {pytorch_output_file}")
    
    cosine_similarity = 1 - cosine(output_pytorch_sample, output_onnx_sample)
    print(f"Cosine Similarity: {cosine_similarity:.4f}")



    onnx_model = onnx.load("/Users/wangyonglin/Desktop/nnlp/gather_test/sample_network.onnx")
    onnx_model_simplified, _ = simplify(onnx_model)
    simplified_file_path = "/Users/wangyonglin/Desktop/nnlp/gather_test/simplified_model.onnx"
    onnx.save(onnx_model_simplified, simplified_file_path)
    print(f"Simplified model saved to {simplified_file_path}")
    ort_session = ort.InferenceSession("/Users/wangyonglin/Desktop/nnlp/gather_test/simplified_model.onnx")
    input_onnx2 = input_tensor.numpy()
    output_onnx2 = ort_session.run(None, {ort_session.get_inputs()[0].name: input_onnx2})[0]
    output_onnx_sample2 = output_onnx2[0]
    cosine_similarity2 = 1 - cosine(output_onnx_sample2, output_onnx_sample)
    print(f"Cosine Similarity: {cosine_similarity2:.4f}")
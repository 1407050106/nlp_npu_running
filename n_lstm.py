import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from scipy.spatial.distance import cosine

class SampleNetwork(nn.Module):
    def __init__(self):
        super(SampleNetwork, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=40, out_channels=192, kernel_size=3, padding=2, stride=1)
        
        # 1st LSTM 层
        self.lstm1 = nn.LSTM(input_size=192, hidden_size=128, batch_first=True, bidirectional=False)
        
        # 1st LayerNorm
        self.layer_norm1 = nn.LayerNorm(128, eps=1e-05, elementwise_affine=True)
        
        # 2nd LSTM 层
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)
        
        # 2nd LayerNorm
        self.layer_norm2 = nn.LayerNorm(128, eps=1e-05, elementwise_affine=True)
        
        # 3rd LSTM 层
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bidirectional=False)
        
        # 3rd LayerNorm
        self.layer_norm3 = nn.LayerNorm(128, eps=1e-05, elementwise_affine=True)

        self.fc = nn.Linear(in_features=128, out_features=64, bias=True)

    def forward(self, x):
        x = self.conv1d(x)     #  x shape: (batch_size, 40, seq_length)
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
    model.eval()

    input_tensor = torch.randn(16, 40, 500)
    output_pytorch = model(input_tensor)
    # output = output[:, :, :500]
    # avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    # output = avg_pool(output)
    print(output_pytorch.shape)

    onnx_file_path = "/Users/wangyonglin/Desktop/nnlp/sample_network.onnx"
    torch.onnx.export(model, input_tensor, onnx_file_path,
                      export_params=True,       # 传递模型参数
                      opset_version=11,         # ONNX 的操作集版本
                      do_constant_folding=True, # 优化常量
                      input_names=['input'],    # 输入名称
                      output_names=['output'])  # 输出名称

    print(f"Model saved to {onnx_file_path}")


    ort_session = ort.InferenceSession(onnx_file_path)
    input_onnx = input_tensor.numpy()
    output_onnx = ort_session.run(None, {ort_session.get_inputs()[0].name: input_onnx})[0]

    output_pytorch_sample = output_pytorch[0].detach().numpy()
    output_onnx_sample = output_onnx[0]
    
    cosine_similarity = 1 - cosine(output_pytorch_sample, output_onnx_sample)
    print(f"Cosine Similarity: {cosine_similarity:.4f}")
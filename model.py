import os
import torch
import torch.nn as nn
import time

torch.manual_seed(1)

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


def Benchmark(data_type, suffix):
    seq_len = 188
    batch_size = 8
    input_size = 128
    hidden_size = 256

    standard_model = StandardLSTM(input_size, hidden_size).cuda().to(data_type)
    standard_model.eval()

    x = torch.randn((seq_len, batch_size, input_size), dtype=data_type, device=torch.device('cuda:0'))

    with torch.no_grad():
        for i in range(5):
            t1 = time.time()
            y = standard_model(x)
            torch.cuda.synchronize()
            t2 = time.time()
            print(f'{(t2-t1)*1000} ms')

    # save data
    with open(f'./data/input.{suffix}.data', 'wb') as f:
        f.write(x.cpu().numpy().tobytes())

    with open(f'./data/output.{suffix}.data', 'wb') as f:
        f.write(y.cpu().numpy().tobytes())

    state_dict = standard_model.state_dict()

    weight_map = {
            '00': state_dict['lstm.weight_ih_l0'], 
            '01': state_dict['lstm.weight_hh_l0'], 
            '10': state_dict['lstm.weight_ih_l0_reverse'], 
            '11': state_dict['lstm.weight_hh_l0_reverse'],
            }

    bias_map = {
            '00': state_dict['lstm.bias_ih_l0'],
            '01': state_dict['lstm.bias_hh_l0'],
            '10': state_dict['lstm.bias_ih_l0_reverse'],
            '11': state_dict['lstm.bias_hh_l0_reverse'],
            }


    for direction in range(2):
        for wtype in range(2):
            weight = weight_map[f'{direction}{wtype}']
            for i in range(4):
                w = weight[:, i * hidden_size : (i+1) * hidden_size]
                with open(f'./data/{direction}{wtype*4+i}0.{suffix}.data', 'wb') as f:
                    f.write(w.cpu().detach().numpy().tobytes())

            bias = bias_map[f'{direction}{wtype}']
            for i in range(4):
                b = bias[i * hidden_size : (i+1) * hidden_size]
                with open(f'./data/{direction}{wtype*4+i}1.{suffix}.data', 'wb') as f:
                    f.write(b.cpu().detach().numpy().tobytes())


if __name__ == '__main__':
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    print("using float precision!")
    Benchmark(torch.float32, 0)
    print()

    print("using half precision!")
    Benchmark(torch.float16, 1)

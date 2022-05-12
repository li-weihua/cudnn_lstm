import argparse
import numpy as np

for data_type, suffix in  zip([np.float32, np.float16], ['0', '1']):

    with open(f'./output.{suffix}.data', 'rb') as f:
        x = np.frombuffer(f.read(), dtype=data_type)

    with open(f'./output_cudnn.{suffix}.data', 'rb') as f:
        y = np.frombuffer(f.read(), dtype=data_type)

    d = np.abs(x-y)

    if suffix == '0':
        print('check float32 precision!')
    else:
        print('check half precision!')
    print(f'ref  : {x.min()}, {x.max()}')
    print(f'cudnn: {y.min()}, {y.max()}')
    print(f'diff: {d.max()}')
    print()

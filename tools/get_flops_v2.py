# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine.analysis import get_model_complexity_info

from mmpretrain import get_model

from fvcore.nn import FlopCountAnalysis
import torch

# v2: 使用fvcore计算FLOPs, 并计算运行时

def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    # parser.add_argument('--config', default='configs/BCFC-329k/AGSA_distilled.py')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    model = get_model(args.config)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total_params/1e6} MB ')
    # 统计可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {trainable_params/1e6} MB ')

    # 假设 model 和 input_data 已经准备好
    device = torch.device('cuda:0')
    model.to(device)
    model.backbone.to(device)
    model.head.to(device)
    input_data = torch.randn(1, 3, 224, 224, device=device)

    # --- 预热 ---
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    # --- 正式测量 ---
    num_iterations = 100
    # 创建事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录开始事件
    start_event.record()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)

    # 记录结束事件
    end_event.record()

    # 等待 GPU 完成所有操作，确保时间计算准确
    torch.cuda.synchronize()

    # 计算总时间（单位是毫秒）
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / num_iterations

    print(f"平均推理时间 (GPU): {avg_time_ms:.2f} ms")


if __name__ == '__main__':
    main()

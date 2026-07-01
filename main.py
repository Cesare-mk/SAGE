import argparse
import os


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model', type=str, default='SAGE', help='Model name'
  )
  parser.add_argument(
      '--dataset', type=str, default='AmazonReviews2014', help='Dataset name'
  )
  return parser.parse_known_args()


if __name__ == '__main__':
  args, unparsed_args = parse_args()
  command_line_configs = parse_command_line_args(unparsed_args)

  pipeline = Pipeline(
      model_name=args.model,
      dataset_name=args.dataset,
      config_dict=command_line_configs,
  )
  pipeline.run()

import numpy as np
import sys


def main():
    # 读取标准输入
    data_line = sys.stdin.readline().strip()
    param_line = sys.stdin.readline().strip()

    # 使用eval解析输入（题目允许）
    data = eval(data_line)
    params = eval(param_line)

    epochs = int(params[0])
    lr = float(params[1])

    # 提取特征和标签
    X_list = [item[0] for item in data]  # [(x1,x2), ...]
    y_list = [item[1] for item in data]  # [1, -1, ...]

    # 转换为numpy数组，方便向量计算
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    n_samples = X.shape[0]
    n_features = X.shape[1]

    # 初始化参数
    w = np.zeros(n_features, dtype=float)  # w = [0, 0]
    b = 0.0

    # 随机梯度下降训练
    for _ in range(epochs):
        for i in range(n_samples):
            xi = X[i]
            yi = y[i]

            # 计算预测值（函数间隔）
            distance = yi * (np.dot(w, xi) + b)

            # 判断是否为错误分类（包括恰好在超平面上的情况）
            if distance <= 0:
                # 更新权重和偏置
                w = w + lr * yi * xi
                b = b + lr * yi

    # 保留3位小数，处理浮点精度
    w1 = round(float(w[0]), 3)
    w2 = round(float(w[1]), 3)
    b_val = round(float(b), 3)

    # 处理 -0.0 情况，转为 0.0（美观性处理，可选）
    if abs(w1) == 0.0:
        w1 = 0.0
    if abs(w2) == 0.0:
        w2 = 0.0
    if abs(b_val) == 0.0:
        b_val = 0.0

    # 按列表格式输出
    result = [w1, w2, b_val]
    print(result)


if __name__ == "__main__":
    main()
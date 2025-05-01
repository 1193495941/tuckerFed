import os
import flgo
import json
import tucker
import fedbuff
import flgo.benchmark.femnist_classification.model.cnn as cnn

import flgo.benchmark.partition as fbp
import numpy as np


class MyIIDPartitioner(fbp.BasicPartitioner):
    def __init__(self, samples_per_client=[4000] * 20):
        self.samples_per_client = samples_per_client
        self.num_clients = len(samples_per_client)  # 该属性用于可视化划分结果图

    def __call__(self, data):
        # 1.1 打乱所有样本的索引
        d_idxs = np.random.permutation(len(data))
        # 1.2 将样本所以划分成num_clients份，并保证返回结果的类型为List[List[int]]
        local_datas = np.split(d_idxs, np.cumsum(self.samples_per_client))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        return local_datas


# 将列表保存为 JSON 文件
def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


# 从 JSON 文件读取列表
def load_from_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)


task = './femnist_result'
gen_config = {
    'benchmark': {'name': 'flgo.benchmark.femnist_classification'},
    'partitioner': {
        'name': MyIIDPartitioner,
    }
}

if not os.path.exists(task):
    data_columns, _ = flgo.gen_task(gen_config, task_path=task)
    # 保存列表到 JSON 文件
    save_to_json(data_columns, 'femnist_data_columns.json')

else:
    data_columns = load_from_json('femnist_data_columns.json')

compress = 'tucker'
n_bit = 8
k = 0.3
gamma = 2
lr = 1e-5
num_rounds = 300
device = 0

if compress == 'tucker':
    algorithm = tucker
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'tucker', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns}

elif compress == 'qsgd':
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'qsgd', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns}

elif compress == 'topk':
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'topk', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns}

else:
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'none', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns}

runner_fedavg1 = flgo.init(task, algorithm=algorithm, option=option, model=cnn, scene='horizontal')
runner_fedavg1.run()

# 对比实验FedPAQ:
# runner_fedavg2 = flgo.init(task, algorithm=fedbuff,
#                            option={'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
#                                    'num_epochs': 5, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
#                                    'responsiveness': 'UNI', 'compress': 'qsgd', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
#                                    'data_columns': data_columns}, model=cnn, scene='horizontal')
# runner_fedavg2.run()

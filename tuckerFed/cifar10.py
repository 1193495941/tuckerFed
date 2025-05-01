import os
import flgo
import json
import tucker
import tucker_test2
import fedbuff
import flgo.benchmark.cifar10_classification.model.cnn as cnn
import flgo.benchmark.cifar10_classification.model.resnet18 as resnet18
import flgo.benchmark.cifar10_classification.model.resnet50 as resnet50


# 将列表保存为 JSON 文件
def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


# 从 JSON 文件读取列表
def load_from_json(filename):
    with open(filename, 'r') as json_file:
        return json.load(json_file)


num_clients = 20
task = f'./cifar10_result_{num_clients}'
data_columns_name = f'cifar10_data_columns_{num_clients}.json'
gen_config = {
    'benchmark': {'name': 'flgo.benchmark.cifar10_classification'},
    'partitioner': {
        'name': 'IIDPartitioner',
        'para': {
            'num_clients': num_clients,
        }
    }
}

if not os.path.exists(task):
    data_columns, _ = flgo.gen_task(gen_config, task_path=task)
    # 保存列表到 JSON 文件
    save_to_json(data_columns, data_columns_name)

else:
    data_columns = load_from_json(data_columns_name)

compress = 'tucker'
rank = 0.01
n_bit = 1
k = 0.01
gamma = 1.1
lr = 5e-3
num_rounds = 300
device = 0
model = resnet18

list_bit = []
if model == resnet18:
    for param in model.Model().parameters():
        list_bit.append(param.numel())
print("sum_list_bit:", sum(list_bit))
print("sum_list_bit:", sum(list_bit))


if compress == 'tucker':
    algorithm = tucker
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'tucker', 'rank': rank, 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns, 'list_bit': list_bit}

elif compress == 'qsgd':
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'qsgd', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns, 'list_bit': list_bit}

elif compress == 'topk':
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'topk', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns, 'list_bit': list_bit}

elif compress == 'fedpaq':
    num_rounds = 60
    n_bit = 16
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 5, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'fedpaq', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns, 'list_bit': list_bit}

else:
    algorithm = fedbuff
    option = {'gpu': [device], 'num_rounds': num_rounds, 'log_file': True, 'sample': 'full',
              'num_epochs': 1, 'num_parallels': 1, 'optimizer': 'Adam', 'learning_rate': lr,
              'responsiveness': 'UNI', 'compress': 'none', 'gamma': gamma, 'n_bit': n_bit, 'k': k,
              'data_columns': data_columns, 'list_bit': list_bit}

runner_fedavg1 = flgo.init(task, algorithm=algorithm, option=option, model=model, scene='horizontal')
runner_fedavg1.run()


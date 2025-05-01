from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy
import torch


class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 1.0, 'eta': 1.0})
        self.buffer = []  # 设置缓存区


    def package_handler(self, received_packages: dict):
        if self.is_package_empty(received_packages): return False
        # 将接收到的包裹添加进缓存，并记录用户的模型延迟
        received_updates = received_packages['model']
        received_client_taus = [u._round for u in received_updates]
        for cdelta, ctau in zip(received_updates, received_client_taus):
            self.buffer.append((cdelta, ctau))
        # 缓存区模型数目大于K时，进行模型更新
        if len(self.buffer) >= int(self.buffer_ratio * self.num_clients):
            if self.compress == 'qsgd' or self.compress == 'fedpaq':
                self.expense += self.bit * (self.n_bit / 32) * self.num_clients
            elif self.compress == 'topk':
                self.expense += self.bit * self.k * self.num_clients
            else:
                self.expense += self.bit * self.num_clients
            # aggregate and clear updates in buffer
            taus_bf = [b[1] for b in self.buffer]
            updates_bf = [b[0] for b in self.buffer]
            weights_bf = [(1 + self.current_round - ctau) ** (-0.5) for ctau in taus_bf]  # 计算每个模型的权重
            model_delta = fmodule._model_average(updates_bf, weights_bf) / len(self.buffer)
            self.model = self.model + self.eta * model_delta
            # clear buffer
            self.buffer = []
            return True
        return False


class Client(BasicClient):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        update = model - global_model
        update._round = model._round

        cpkg = self.pack(update)
        return cpkg

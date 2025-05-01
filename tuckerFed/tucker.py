from flgo.algorithm.asyncbase import AsyncServer
from flgo.algorithm.fedbase import BasicClient
import flgo.utils.fmodule as fmodule
import copy
import torch
import numpy as np
import tensorly as tl

tl.set_backend('pytorch')
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor


class PIDController:
    def __init__(self, Kp, Ki, Kd, gamma):
        self.Kp = Kp  # Proportional coefficient
        self.Ki = Ki  # Integral coefficient
        self.Kd = Kd  # Derivative coefficient
        self.gamma = gamma
        # 初始化积分和前一误差，用于计算微分项
        self.integral = 0
        self.prev_error = None
        self.last_update_output = None  # 用于存储最近更新的控制输出 z(t0 + 1)

    def compute_output(self, current_error):
        """
        计算当前的控制输出 z(t)
        """
        # 计算比例项
        proportional = self.Kp * current_error

        # 计算积分项（累计误差）
        self.integral += current_error
        integral = self.Ki * self.integral

        # 计算微分项（误差的变化率）
        derivative = 0
        if self.prev_error is not None:
            derivative = self.Kd * (current_error - self.prev_error)

        # 更新前一误差
        self.prev_error = current_error

        # 计算控制输出 z(t)
        control_output = proportional + integral + derivative
        return control_output

    def should_update_basis(self, control_output):
        """
        根据控制输出决定是否更新基向量
        """
        if self.last_update_output is None:
            # 首次更新基向量
            self.last_update_output = control_output
            return True

        # 相对阈值判断
        if control_output > self.gamma * self.last_update_output:
            self.last_update_output = control_output
            return True
        return False


class Server(AsyncServer):
    def initialize(self):
        self.init_algo_para({'buffer_ratio': 1, 'eta': 1.0})
        self.buffer = []  # 设置缓存区
        self.cores = []
        self.global_factors = None
        self.is_update_basic = True
        self.re_update = None
        self.pid_controller = PIDController(Kp=1, Ki=1, Kd=1, gamma=self.gamma)
        self.rank = self.option['rank']
        self.trans_type = 0
        self.alpha_bit = 0

    def package_handler(self, received_packages: dict):
        if self.is_package_empty(received_packages):
            return False
        # 将接收到的包裹添加进缓存，并记录用户的模型延迟
        received_update = received_packages['update']
        received_alphas = received_packages['alphas']
        received_sum_errors = received_packages['sum_errors']
        # print("received_sum_errors",received_sum_errors)
        for update, alphas, sum_errors in zip(received_update, received_alphas, received_sum_errors):
            self.buffer.append((update, alphas, sum_errors))

        # 缓存区模型数目大于K时，进行模型更新
        if len(self.buffer) >= int(self.buffer_ratio * self.num_clients):
            updates_bf = [b[0] for b in self.buffer]
            weights_bf = [1] * len(self.buffer)  # 计算每个模型的权重
            update_delta = fmodule._model_average(updates_bf, weights_bf) / len(self.buffer)

            if self.is_update_basic is True:
                self.trans_type = 0
                self.model = self.model + update_delta

                self.expense += self.bit * self.num_clients * (self.n_bit / 32)
                self.cores = []
                self.global_factors = []
                for param in update_delta.parameters():
                    if param.dim() == 4:
                        lay_factors = []
                        ranks = [max(1, int(dim)) for dim in param.shape]  # 确保秩至少为1
                        core, factors = tucker(param.data, rank=ranks)
                        for factor in factors:
                            U, S, Vh = torch.linalg.svd(factor)
                            # 截断 SVD：保留前 k 个奇异值
                            k = max(1, int(self.rank * len(S)))  # 保证至少保留一个奇异值
                            U = U[:, :k]
                            S = S[:k]
                            Vh = Vh[:k, :]
                            lay_factors.append(U)
                        self.global_factors.append(lay_factors)
                        self.cores.append(core)
                # 设置flag为False
                self.is_update_basic = False
            else:
                self.trans_type = 1
                self.alpha_bit = 0
                conv_updates = []
                for param in update_delta.parameters():
                    if param.dim() == 4:
                        conv_updates.append(torch.zeros_like(param.data))

                for b in self.buffer:
                    for lay_alpha, lay_factors, core, conv_update in zip(b[1], self.global_factors, self.cores,
                                                                         conv_updates):
                        update_factors = []
                        for alpha, factor in zip(lay_alpha, lay_factors):
                            self.expense += 8 * alpha.numel()
                            self.alpha_bit += 8 * alpha.numel() / len(self.buffer)
                            update_factor = factor @ alpha
                            update_factors.append(update_factor)

                        re_param = tl.tucker_to_tensor((core, update_factors))
                        conv_update += re_param / len(self.buffer)
                i = 0
                for param in update_delta.parameters():
                    if param.dim() == 4:
                        param.data = conv_updates[i]
                        i += 1
                    else:
                        # self.expense += 8 * param.numel() * self.num_clients
                        self.alpha_bit += 8 * param.numel()

                self.model = self.model + update_delta

                errors_bf = [b[2] for b in self.buffer]
                control_output = self.pid_controller.compute_output(sum(errors_bf))
                if self.pid_controller.should_update_basis(control_output):
                    self.is_update_basic = True
                    # print(f"next round{self.current_round + 1} update basic")
                    self.gv.logger.info(f"next round{self.current_round + 1} update basic")
                else:
                    self.is_update_basic = False

            self.buffer = []
            return True
        return False

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {
            "model": copy.deepcopy(self.model),
            "cores": self.cores,
            "global_basics": self.global_factors,
            "is_update_basic": self.is_update_basic,
        }


class Client(BasicClient):
    def initialize(self):
        self.cores = []
        self.tucker_basic = []
        self.is_update_basic = None
        self.sum_errors = None
        self.alphas = None

    def compute_alpha(self, update, cores, global_basic):
        self.alphas = []
        errors = []
        i = 0
        for param in update.parameters():
            if param.dim() == 4:
                ranks = [max(1, int(dim)) for dim in param.shape]  # 确保秩至少为1
                core, factors = tucker(param.data, rank=ranks)
                errors.append(torch.norm(param.data - tucker_to_tensor((cores[i], factors))).item())
                lay_alpha = []
                for j, factor in enumerate(factors):
                    alpha = global_basic[i][j].T @ factor
                    lay_alpha.append(alpha)
                self.alphas.append(lay_alpha)
                i += 1
        self.sum_errors = sum(errors)

    def reply(self, svr_pkg):
        model, cores, global_basics, is_update_basic = self.unpack(svr_pkg)
        global_model = copy.deepcopy(model)
        self.train(model)
        update = model - global_model

        if is_update_basic is True:
            # 在服务器聚合梯度
            pass
        else:
            # 传输系数alpha
            self.compute_alpha(update, cores, global_basics)

        cpkg = self.pack(update)
        return cpkg

    def pack(self, update, *args, **kwargs):
        zero_model = copy.deepcopy(update)
        for param in zero_model.parameters():
            param.data.zero_()

        compressed_updates = {}
        n_bit = self.option['n_bit']
        for name, param in update.named_parameters():
            compressed_updates[name] = self.qsgd_compress(param.data, n_bit=n_bit, random=True)

        decompress_model = copy.deepcopy(zero_model)
        for name, signature in compressed_updates.items():
            decompressed_param = self.qsgd_decompress(signature, n_bit=n_bit)
            decompress_model.state_dict()[name].copy_(decompressed_param)


        return {
            "update": update,
            "alphas": self.alphas,
            "sum_errors": self.sum_errors,
        }

    def unpack(self, received_pkg):
        return (received_pkg['model'],
                received_pkg['cores'],
                received_pkg['global_basics'],
                received_pkg['is_update_basic'])

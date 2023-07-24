#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
@Project :kaiwu-fwk 
@File    :torch_dqn_model.py
@Author  :kaiwu
@Date    :2022/12/15 22:50 

'''


import numpy as np
import torch
from torch import nn
import re
import os
from framework.common.config.config_control import CONFIG
from framework.common.utils.common_func import get_first_line_and_last_line_from_file, get_last_two_line_from_file
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from conf.gorge_walk_v1.config import DimConfig, DQNConfig, Config
from app.gorge_walk.env.sample_processor.gorge_walk_sample_processor import Frame
from copy import deepcopy


class DQNModel(object):
    """
        dqn算法模型的实现: 包括神经网络、模型预测、模型训练、模型保存、模型恢复
    """

    def __init__(self, network, name, role='actor'):
        """
            Parameters
            ----------
            network : torch_network.BaseNetwork
                神经网络通过参数传入
            name : str
                该模型的名字，用于标识
            role : str
                适配框架, 用于区分当前模型的使用场景(actor或learner), 当前模型不进行区分
        """
        super().__init__()
        self.model = network
        self.optim = torch.optim.Adam(self.model.parameters(), lr=DQNConfig.START_LR)
        self._eps = np.finfo(np.float32).eps.item()
        self._gamma = DQNConfig.GAMMA
        self.name = name

        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.file_queue = []

    # 更新目标模型的参数
    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self, g_data):
        """
            Description: 该方法实现了dqn算法和模型的训练过程
            ----------

            Return: 训练过程中产生的数据, 用于统计
            ----------

            Parameters
            ----------
            g_data: list
                由reverb传送过来的一个batch的原始训练数据

        """

        # 将reverb传入的数据转换成可以训练的数据
        t_data = self.__rdata2tdata(g_data)

        # 提取训练需要用到的数据
        obs = [frame.obs for frame in t_data]
        action = torch.LongTensor([frame.act for frame in t_data]).view(-1, 1).long().to(self.model.device)
        ret = torch.tensor([frame.ret for frame in t_data], device=self.model.device)
        _obs = [frame._obs for frame in t_data]
        not_done = torch.tensor([0 if frame.done == 1 else 1 for frame in t_data], device=self.model.device)

        model = getattr(self, 'target_model')
        model.eval()
        with torch.no_grad():
            q, h = model(_obs, state=None)
            q_max = q.max(dim=1).values.detach()  # .cpu()

        target_q = ret + self._gamma * q_max * not_done

        self.optim.zero_grad()
        frames = self(obs, model_selector="model", model_mode="train")
        loss = torch.square(target_q - frames.logits.gather(1, action).view(-1)).sum()
        loss.backward()
        self.optim.step()

        self.train_step += 1
        # 更新target网络
        if self.train_step % DQNConfig.TARGET_UPDATE_FREQ == 0:
            self.update_target_q()

        # 返回统计数据
        loss_value = loss.detach().item()
        return loss_value, loss_value, target_q.mean().detach().item()

    def get_action(self, *kargs, **kwargs):
        return self.predict(*kargs, **kwargs)

    def predict(self, obs, state=None, types="prob", model_selector="model"):
        """
            Description: 该方法实现了模型的预测
            ----------

            Return: 
            ----------
            format_action: list
                预测得到的动作序列
            network_sample_info: list
                返回的其他信息，该算法无需返回有效信息
            lstm_info: list
                返回的lstm相关信息, 该网络没有使用lstm, 则返回None

            Parameters:
            ----------
            obs: dict
                由aisvr传送过来的一个observation数据

        """
        model = getattr(self, model_selector)
        model.eval()
        obs = obs["observation"]
        with torch.no_grad():
            if types == "max":
                logits, h = model(obs, state=state)
                act = logits.argmax(dim=1).view(-1,1).numpy()
            elif types == "prob":
                if np.random.rand(1) >= DQNConfig.EPSLION:    # epslion greedy
                    act = np.random.choice(range(DimConfig.DIM_OF_ACTION), len(obs)).reshape(-1,1) 
                else:
                    logits, h = model(obs, state=state)
                    act = logits.argmax(dim=1).view(-1,1).numpy()
            else:
                raise AssertionError
        
        format_action = [instance.tolist() for instance in act]
        network_sample_info = [(None, None)] * len(format_action)
        lstm_info = [None] * len(format_action)
        return format_action, network_sample_info, lstm_info

    def __call__(self, obs, state=None, model_selector="model", model_mode="train"):
        model = getattr(self, model_selector)
        getattr(model, model_mode)()            # model.train() or model.eval()
        logits, h = model(obs, state=state)
        return Frame(logits=logits)

    def should_stop(self):
        return False

    def stop(self):
        return True

    def load_last_new_model(self, models_path):
        """
            Description: 根据传入的模型路径，载入最新模型
        """
        checkpoint_file = f'{models_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

        _, last_line = get_first_line_and_last_line_from_file(checkpoint_file)

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = last_line.split(
            f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-')[1]
        checkpoint_id = re.findall(r'\d+\.?\d*', checkpoint_id)[0]

        self.load_param(path=models_path, id=checkpoint_id)

    def load_specific_model(self, models_path):
        """
            Description: 根据传入的模型，载入指定模型
        """
        checkpoint_id = models_path.split(
            f'{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-')[1][:-4]
        models_path = os.path.dirname(models_path)

        self.load_param(path=models_path, id=checkpoint_id)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def __rdata2tdata(self, r_data):
        """
            Description: 该方法将reverb传入的数据转换成可以训练的数据
            ----------

            Return: 
            ----------
            t_data: list
                训练数据

            Parameters
            ----------
            r_data: list
                由reverb传入的原始数据
        """

        t_data = list(r_data)
        return [Frame(obs=i[:DimConfig.observation_shape],
                      _obs=i[DimConfig.observation_shape:2 *
                             DimConfig.observation_shape],
                      act=i[-4], rew=i[-3], ret=i[-2], done=i[-1]
                      )for i in t_data]

    def save_param(self, path=None, id='1'):
        """
            Description: 保存模型的方法
            ----------

            Parameters
            ----------
            path: str
                保存模型的路径
            id: int
                保存模型的id
        """
        path = f'{CONFIG.restore_dir}/{self.name}/'

        torch.save(self.model.state_dict(),
                   f"{str(path)}/model.ckpt-{str(id)}.pkl")
        file_exist_flag = os.path.exists(f"{str(path)}/checkpoint")
        with open(f"{str(path)}/checkpoint", mode='a') as fp:
            if not file_exist_flag:
                fp.writelines([
                    f"checkpoints list\n"
                ])
            fp.writelines([
                f"all_model_checkpoint_paths: \"{str(path)}/model.ckpt-{str(id)}\"\n"
            ])
        self.add_file_to_queue(f"{str(path)}/model.ckpt-{str(id)}.pkl")

    def add_file_to_queue(self, file_path):
        self.file_queue.append(file_path)
        if len(self.file_queue) > Config.MAX_FILE_KEEP_CNT:
            os.remove(self.file_queue.pop(0))

    # 加载模型文件并且更新参数
    def load_param(self, path='/tmp/pyt-model', id='1'):
        self.model.load_state_dict(torch.load(f"{str(path)}/model.ckpt-{str(id)}.pkl",
                                   map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        self.update_target_q()

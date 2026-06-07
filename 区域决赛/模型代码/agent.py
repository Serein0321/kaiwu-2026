#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import math
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.conf.conf import Config, GameConfig
from agent_ppo.feature.reward_manager import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm

# DIY
from agent_ppo.feature.unpack_state_dict import Info
from agent_ppo.feature.obs_builder import ObsBuilder
from agent_ppo.utils.list_rearrange import ListRearrange

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        # DIY
        self.info = Info()
        self.obs_builder = ObsBuilder()
        self.last_action = None

        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def init_config(self, lineups=None, camp=None):
        """Choose summoner skill before env.reset for the 112/133 competition protocol.

        The workflow passes the current camp explicitly. The fallback keeps this method usable
        when called by the platform directly with only lineups.
        """
        if isinstance(lineups, dict) and "my_heroes" in lineups:
            my_heroes = lineups.get("my_heroes", [])
            opponent_heroes = lineups.get("opponent_heroes", [])
            enemy_hero_id = opponent_heroes[0] if opponent_heroes else None
            select_skills = {}
            for hero_id in my_heroes:
                select_skills[hero_id] = GameConfig.select_summoner_skill(hero_id, enemy_hero_id)
            if self.logger:
                self.logger.info(
                    f"init_config my_heroes={my_heroes}, opponent_heroes={opponent_heroes}, select_skills={select_skills}"
                )
            return select_skills

        hero_id, enemy_hero_id = self._resolve_lineup_heroes(lineups, camp)
        summoner_skill_id = GameConfig.select_summoner_skill(hero_id, enemy_hero_id)
        if self.logger:
            self.logger.info(
                f"init_config hero_id={hero_id}, enemy_hero_id={enemy_hero_id}, summoner_skill_id={summoner_skill_id}"
            )
        return summoner_skill_id

    def _resolve_lineup_heroes(self, lineups=None, camp=None):
        if camp is None:
            camp = self.hero_camp if self.hero_camp in [0, 1] else 0
        if isinstance(camp, str):
            camp = 0 if camp in ["blue", "blue_camp", "PLAYERCAMP_1"] else 1

        default_blue = GameConfig.CAMP_HEROES[0][0]
        default_red = GameConfig.CAMP_HEROES[-1][0]
        if lineups is None:
            return (default_blue, default_red) if camp == 0 else (default_red, default_blue)

        if isinstance(lineups, dict):
            lineup_dict = lineups.get("lineups", lineups)
            blue_id = lineup_dict.get("blue_camp", [{"hero_id": default_blue}])[0].get("hero_id", default_blue)
            red_id = lineup_dict.get("red_camp", [{"hero_id": default_red}])[0].get("hero_id", default_red)
            return (blue_id, red_id) if camp == 0 else (red_id, blue_id)

        if isinstance(lineups, (list, tuple)) and len(lineups) >= 2:
            hero_id = self._lineup_item_to_hero_id(lineups[camp], default_blue if camp == 0 else default_red)
            enemy_hero_id = self._lineup_item_to_hero_id(lineups[1 - camp], default_red if camp == 0 else default_blue)
            return hero_id, enemy_hero_id

        return (default_blue, default_red) if camp == 0 else (default_red, default_blue)

    def _lineup_item_to_hero_id(self, item, default):
        if isinstance(item, dict):
            return item.get("hero_id", default)
        if isinstance(item, (list, tuple)) and item:
            return item[0]
        return item if item is not None else default

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = self._normalize_camp(observation.get("player_camp", observation.get("camp", 0)))
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id, self.hero_camp)

        # DIY
        self.info.reset()
        self.obs_builder.reset()
        self.last_action = None

    def _normalize_camp(self, camp):
        if isinstance(camp, str):
            if camp.startswith("PLAYERCAMP_"):
                return int(camp[-1]) - 1
            return int(camp) - 1 if camp in ["1", "2"] else int(camp)
        return int(camp) - 1 if int(camp) in [1, 2] else int(camp)

    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            biased_logits = self._apply_combo_skill_bias(logits[i], legal_action[i])
            prob, d_prob, action, d_action = self._sample_masked_action(biased_logits, legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        self.last_action = action
        # Current-frame action feeds next-frame observation features.
        self.obs_builder.record_action(action)
        return action

    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        self.last_action = d_action
        # Current-frame action feeds next-frame observation features.
        self.obs_builder.record_action(d_action)
        return d_action

    def observation_process(self, observation):
        self.info.update(observation)
        feature = self.obs_builder.build_observation(self.info)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def action_process(self, observation, act_data, is_stochastic):
        action = act_data.action if is_stochastic else act_data.d_action
        if self.hero_camp == 1:
            action = list(action)
            for i in range(1, 5):
                if action[i] > 0:
                    action[i] = 16 - action[i]
        return action

    def _apply_combo_skill_bias(self, logits, legal_action):
        preferred_button = self._preferred_combo_skill_button(legal_action)
        if preferred_button is None:
            return logits
        biased_logits = logits.copy()
        biased_logits[preferred_button] += GameConfig.SKILL_BUTTON_LOGIT_BONUS

        target_offset = sum(self.label_size_list[:-1])
        target_legal_action = np.array(legal_action)[target_offset:]
        target_legal_action = target_legal_action.reshape(self.legal_action_size[0], -1)
        if target_legal_action[preferred_button][1] > 0:
            biased_logits[target_offset + 1] += GameConfig.SKILL_TARGET_LOGIT_BONUS
        return biased_logits

    def _preferred_combo_skill_button(self, legal_action):
        if not hasattr(self, "info") or self.info is None:
            return None
        our = getattr(self.info, "hero_our", None)
        enemy = getattr(self.info, "hero_enemy", None)
        if our is None or enemy is None:
            return None
        our_pos = getattr(our.info, "position", None)
        enemy_pos = getattr(enemy.info, "position", None)
        if self._unseen_position(our_pos) or self._unseen_position(enemy_pos):
            return None
        enemy_hp_max = getattr(enemy.info, "hp_max", 0)
        enemy_alive = enemy_hp_max > 0 and getattr(enemy.info, "hp", 0) > 0
        if not enemy_alive or math.dist(our_pos, enemy_pos) > GameConfig.SKILL_URGENCY_DISTANCE:
            return None
        if self._inside_alive_enemy_tower_range(our_pos):
            return None

        legal_action = np.array(legal_action)
        button_legal = legal_action[: self.label_size_list[0]]
        target_legal_action = legal_action[sum(self.label_size_list[:-1]) :].reshape(self.legal_action_size[0], -1)
        hero_id = getattr(our.info, "config_id", 0)
        for button in GameConfig.SKILL_BUTTON_PRIORITY.get(hero_id, [6, 4, 5]):
            if button_legal[button] <= 0 or target_legal_action[button][1] <= 0:
                continue
            if self._skill_button_usable(our, button):
                return button
        return None

    def _inside_alive_enemy_tower_range(self, our_pos):
        organ_enemy = getattr(self.info, "organ_enemy", None)
        enemy_tower = getattr(organ_enemy, "sub_tower", None)
        if enemy_tower is None:
            return False
        tower_pos = getattr(enemy_tower, "position", None)
        if self._unseen_position(our_pos) or self._unseen_position(tower_pos):
            return False
        tower_hp = getattr(enemy_tower, "hp", 0)
        tower_range = getattr(enemy_tower, "attack_range", 0) or 0
        return tower_hp > 0 and tower_range > 0 and math.dist(our_pos, tower_pos) <= tower_range

    def _skill_button_usable(self, hero, button):
        slot_map = {
            4: getattr(hero.skill, "first", None),
            5: getattr(hero.skill, "second", None),
            6: getattr(hero.skill, "thrid", None),
            8: getattr(hero.skill, "summoner", None),
        }
        slot = slot_map.get(button)
        if slot is None:
            return False
        return bool(getattr(slot, "usable", False)) and (button == 8 or getattr(slot, "level", 0) > 0)

    def _unseen_position(self, position):
        return position is None or len(position) < 2 or position[0] == Info.UNSEEN_PADDING

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            load_dict = torch.load(model_file_path, map_location=self.device)
            try:
                self.model.load_state_dict(load_dict)
            except RuntimeError as err:
                loaded, total = self._load_compatible_state_dict(load_dict)
                if self.logger:
                    self.logger.warning(
                        f"partial load model {model_file_path}: {loaded}/{total} tensors matched after 112/133 dim change; {err}"
                    )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def _load_compatible_state_dict(self, load_dict):
        current = self.model.state_dict()
        compatible = {
            key: value for key, value in load_dict.items()
            if key in current and current[key].shape == value.shape
        }
        current.update(compatible)
        self.model.load_state_dict(current)
        return len(compatible), len(current)

    def load_opponent_agent(self, id="1"):
        # Framework provides loading opponent agent function, no need to implement function content
        # 框架提供的加载对手模型功能，无需实现函数内容
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional list
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))

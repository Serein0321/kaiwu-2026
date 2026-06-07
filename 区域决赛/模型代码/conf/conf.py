#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    BASE_REWARD_WEIGHT_DICT = {
        "hp_point": 2.0,
        "tower_hp_point": 10.0,
        "money": 4e-3,
        "exp": 4e-3,
        "ep_rate": 0.75,
        "death": -1.0,
        "kill": -0.6,
        "last_hit": 0.5,
        "forward": 0.01,
    }

    TACTIC_REWARD_WEIGHT_DICT = {
        "frenzy_trade": 0.18,
        "lane_clear": 0.10,
        "lane_push": 0.08,
        "river_crab_control": 0.12,
        "combo_luban": 0.16,
        "combo_direnjie": 0.16,
        "skill_urgency": 0.35,
        "hero_target_priority": 0.25,
        "attack_accuracy": 0.14,
        "attack_spacing": 0.22,
        "grass_ambush": 0.10,
        "tower_safety": 0.45,
        "tower_push": 0.18,
        "recovery_choice": 0.10,
        "safe_recall": 0.08,
        "cake_recovery": 0.15,
        "enemy_luban_ult_safety": 0.30,
    }

    REWARD_WEIGHT_DICT = {
        **BASE_REWARD_WEIGHT_DICT,
        **TACTIC_REWARD_WEIGHT_DICT,
    }
    REMOVE_FORWARD_AFTER = 1000  # 一定帧数后删除forward奖励
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 8000
    # 跳过衰减的奖励
    REWARD_WITHOUT_TIME_SCALE = {
        # "hp_point",
        # "tower_hp_point",
        # "death",
        # "kill",
    }
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800
    # 2026峡谷追猎官方阵容: 112=鲁班七号, 133=狄仁杰。训练轮换会覆盖2x2全组合。
    CAMP_HEROES = [
        [112],  # 鲁班七号
        [133],  # 狄仁杰
    ]

    SUMMONER_SKILLS = {
        "heal": 80102,
        "stun": 80103,
        "smite": 80104,
        "interfere": 80105,
        "purify": 80107,
        "execute": 80108,
        "sprint": 80109,
        "frenzy": 80110,
        "flash": 80115,
        "weaken": 80121,
    }
    # reset前由Agent.init_config调用。当前112/133全阵容统一选择狂暴。
    SUMMONER_POLICY = {
        (112, 112): SUMMONER_SKILLS["frenzy"],
        (112, 133): SUMMONER_SKILLS["frenzy"],
        (133, 112): SUMMONER_SKILLS["frenzy"],
        (133, 133): SUMMONER_SKILLS["frenzy"],
    }
    SKILL_BUTTON_PRIORITY = {
        112: [6, 4, 5],
        133: [6, 4],
    }
    SKILL_BUTTON_LOGIT_BONUS = 2.4
    SKILL_TARGET_LOGIT_BONUS = 1.2
    SKILL_URGENCY_DISTANCE = 8500
    DEFAULT_ATTACK_RANGE = 8500
    ATTACK_SPACING_SAFE_RATIO = 0.65
    ATTACK_SPACING_IDEAL_RATIO = 0.90
    TOWER_AGGRO_RISK_PENALTY = -1.6
    TOWER_AGGRO_TARGETED_PENALTY = -2.0
    TOWER_LOW_HP_EXTRA_PENALTY = -0.4
    CAKE_LOW_HP_THRESHOLD = 0.35
    CAKE_MID_HP_THRESHOLD = 0.55
    CAKE_SEEK_MAX_DISTANCE = 12000
    CAKE_NEAR_DISTANCE = 1800
    CAKE_CONSUME_DISTANCE = 2500
    CAKE_HP_GAIN_THRESHOLD = 0.08
    CAKE_APPROACH_DELTA = 500
    CAKE_LOW_APPROACH_REWARD = 0.3
    CAKE_LOW_NEAR_REWARD = 0.8
    CAKE_LOW_CONSUME_REWARD = 1.8
    CAKE_LOW_RETREAT_PENALTY = -0.3
    CAKE_LOW_BAD_COMBAT_PENALTY = -0.5
    CAKE_MID_APPROACH_REWARD = 0.15
    CAKE_MID_NEAR_REWARD = 0.4
    CAKE_MID_CONSUME_REWARD = 1.0
    RECALL_ENEMY_ALIVE_PENALTY = -1.2
    RECALL_LANE_UNCLEARED_PENALTY = -0.6
    RECALL_HEALTHY_PENALTY = -0.5
    LUBAN_HERO_ID = 112
    LUBAN_ULT_SLOT_TYPE = "SLOT_SKILL_3"
    LUBAN_ULT_DANGER_RADIUS = 3200
    LUBAN_ULT_WARNING_RADIUS = 5400
    LUBAN_ULT_ESCAPE_DELTA = 450
    LUBAN_ULT_STAY_PENALTY = -1.0
    LUBAN_ULT_LOW_HP_EXTRA_PENALTY = -0.35
    LUBAN_ULT_APPROACH_PENALTY = -0.35
    LUBAN_ULT_ESCAPE_REWARD = 0.35
    LUBAN_ULT_EXIT_REWARD = 1.0

    @classmethod
    def select_summoner_skill(cls, hero_id, enemy_hero_id=None):
        """Select frenzy for all matchups."""
        return cls.SUMMONER_SKILLS["frenzy"]

    """ DEBUG """
    # 是否使用固定动作的agent调试obs
    debug_agent: bool = False
    debug_max_run_episodes: int = 1
    # 是否保存obs信息到json文件
    debug_frames: bool = False
    # 从56开始一个step为6帧
    debug_max_save_frame_no: int = 56 + 6 * 8000
    # debug总帧数 (4800升到4级, 2300升到3级)
    debug_total_frames = 4000

class Args:
    ### Observation Builder 配置 ###
    # 单位通用特征
    # 视野/位置说明:
    # - 官方动作方向仍为16x16, 对应Config.LABEL_SIZE_LIST的move/skill二维离散方向。
    # - 这里的43不是动作方向, 而是上一年度结构化观测中的相对位置bucket数;
    #   24600/600得到41个视野内bucket, 两端各补一个视野外溢出bucket, 合计43。
    ACTION_DIRECTION_GRID_SIZE = 16
    RELATIVE_DISTANCE_UNIT_SIZE = 600  # 相对距离单位大小 (向上取整)
    RELATIVE_DISTANCE_MAX_SIZE = 24600  # 相对距离最大大小
    RELATIVE_POSITION_BUCKET_DIM = RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2
    DIM_RELATIVE_DISTANCE = (RELATIVE_DISTANCE_MAX_SIZE // RELATIVE_DISTANCE_UNIT_SIZE + 2) * 2 + 1
    WHOLE_DISTANCE_UNIT_SIZE = 5000  # 全局距离单位大小 (向下取整)
    WHOLE_DISTANCE_MAX_SIZE = int(9e4)  # 全局距离最大大小
    DIM_WHOLE_DISTANCE = (WHOLE_DISTANCE_MAX_SIZE // WHOLE_DISTANCE_UNIT_SIZE) * 2 + 2
    DIM_DISTANCE = DIM_RELATIVE_DISTANCE + DIM_WHOLE_DISTANCE
    HP_UNIT_SIZE = 100  # 生命值单位大小
    HP_MAX_SIZE = 2400  # 生命值最大大小 (向上取整)
    HERO_CONFIGS = {
        112: {
            "name": "luban7",
            "cn_name": "鲁班七号",
            "feature_value": -1.0,
            "skills": {
                "passive": 11200,
                "skill_1": 11210,
                "skill_2": 11220,
                "skill_3": 11230,
            },
            "play_style": "burst_and_long_range_poke",
        },
        133: {
            "name": "direnjie",
            "cn_name": "狄仁杰",
            "feature_value": 1.0,
            "skills": {
                "passive": 13300,
                "skill_1": 13310,
                "skill_2": 13320,
                "skill_3": 13330,
            },
            "play_style": "kite_and_cleanse_duel",
        },
    }
    HERO_CONFIG_ID = list(HERO_CONFIGS.keys())
    HERO_FEATURE_VALUE = {112: -1.0, 133: 1.0}
    COMBO_PHASES = [
        "idle",
        "opener",
        "after_skill",
        "expect_normal_attack",
        "follow_up_skill",
        "finish",
        "failed_timeout",
    ]
    COMBO_PHASE_DIM = len(COMBO_PHASES)
    ATTACK_WINDOW_SIZE = 12
    COMBO_TIMEOUT_FRAMES = 90
    USE_FRENZY_BUTTON = 8
    NORMAL_ATTACK_BUTTON = 3
    RECOVER_BUTTON = 7
    RECALL_BUTTON = 9
    MARK_ID_LAYERS = {  # 每个mark的最大层数; 未观测到的新mark会落入未知位
        11200: 4,  # 鲁班七号被动火力压制: 5次普攻前的计数层
        11201: 1,  # 鲁班七号扫射/强化普攻状态
        13300: 5,  # 狄仁杰被动迅捷: 最多5层
        13301: 1,  # 狄仁杰强化令牌/随机强化普攻状态
    }
    DIM_MARK = sum([v + 1 for v in MARK_ID_LAYERS.values()]) + 1  # 所有层数都加上0层, 以及一个未知空位
    DIM_UNIT = int(
        DIM_DISTANCE +
        HP_MAX_SIZE // HP_UNIT_SIZE + 3 +
        DIM_MARK
    )
    # 英雄专属特征
    HERO_BEHAVE = ['State_Dead', 'State_Idle',
    'Direction_Move', 'Normal_Attack', 'State_Revive',
    'UseSkill_1', 'UseSkill_2', 'UseSkill_3']  # 映射behave到编号
    EP_UNIT_SIZE = 30  # 法术单位大小
    EP_MAX_SIZE = 240  # 法术最大大小 (向下取整)
    CD_UNIT_SIZE = 1  # 冷却单位大小
    CD_MAX_SIZE = 10  # 冷却最大大小 (向上取整)
    LEVEL_MAX = 15  # 最大等级
    MONEY_UNIT_SIZE = 20  # 金币获得单位大小
    MONEY_MAX_SIZE = 300  # 金币获得最大大小 (向下取整)
    BUFFS = [
        90015,  # 可能是泉水的回复buff
        10000,  # 点回复技能时候产生的buff (1.2s先消失)
        10010,  # 回复技能产生的恢复buff (5.8s)
        11001,  # 可能是加速buff
        11002,  # 可能是减速buff
        11010,  # 可能是净化buff
        11111,  # 通用异常/英雄状态buff兜底
        911220, 911290, 914110, 914210, 914211, 914250,  # 一些未知buff (6,)
        # 鲁班七号: 被动扫射、一技能手雷、二技能火箭炮、三技能飞艇照明/减速。
        112000, 112001, 112010, 112020, 112040,
        112100, 112110, 112120, 112150,
        112200, 112210, 112220, 112250,
        112300, 112310, 112320, 112330, 112350,
        112900, 112910, 112920, 112950, 112990,
        # 狄仁杰: 被动迅捷、一技能强化令牌、二技能解控无敌、三技能眩晕/破防。
        133000, 133001, 133010, 133020, 133040,
        133100, 133101, 133110, 133120, 133150,
        133200, 133210, 133220, 133250,
        133300, 133310, 133320, 133350,
        133900, 133910, 133920, 133950, 133990,
    ]
    DIM_BUFF = len(BUFFS) + 1
    DIM_HERO = (
        DIM_UNIT + 1 + len(HERO_BEHAVE) + 1 +
        EP_MAX_SIZE // EP_UNIT_SIZE + 2 +
        (CD_MAX_SIZE // CD_UNIT_SIZE + 4) * 5 +  # 1/2/3技能、召唤师技能、回复技能
        LEVEL_MAX + 
        MONEY_MAX_SIZE // MONEY_UNIT_SIZE + 3 +  # 离散化金钱变化, 金钱变化是否在0~20, 总金钱
        1 +  # 是否在草丛
        2 +  # 是否在塔攻击范围内, 是否为塔的攻击目标
        DIM_BUFF
    )
    # 小兵专属特征
    SOLDIER_MAX_NUM = 4  # 考虑的最大小兵数目
    SOLDIER_BEHAVE = ['State_Dead', 'Attack_Path']  # 映射behave到编号
    SOLDIER_CONFIG_ID = [[6801, 6804], [6800, 6803], [6802, 6805]]  # 近战, 远程, 炮车
    DIM_SOLDIER = (
        DIM_UNIT + len(SOLDIER_BEHAVE) + 1 + len(SOLDIER_CONFIG_ID) +
        2  # 是否在塔的攻击范围内, 是否为塔的攻击目标
    )
    DIM_SOLDIERS = DIM_SOLDIER * SOLDIER_MAX_NUM
    # 河蟹专属特征
    RIVER_CRAB_BEHAVE = ['State_Dead', 'State_Auto', 'State_Revive', 'State_Born']
    DIM_RIVER_CRAB = DIM_UNIT + len(RIVER_CRAB_BEHAVE) + 1
    # 防御塔专属特征
    DIM_ORGAN = DIM_UNIT + 3 + 2  # (3,)攻击目标, (2,)塔后是否有血包, 血包生成剩余时间
    # 全部单位特征
    DIM_ALL_UNITS = (
        DIM_HERO * 2 + DIM_SOLDIERS * 2 +
        DIM_RIVER_CRAB + DIM_ORGAN * 2
    )
    # 子弹特征
    BULLET_MAX_NUM = 10  # 最大子弹数量, 9个英雄子弹, 1个防御塔子弹
    BULLET_SLOT = ['SLOT_SKILL_0', 'SLOT_SKILL_1', 'SLOT_SKILL_2', 'SLOT_SKILL_3', 'SLOT_SKILL_VALID']
    DIM_BULLET = (
        len(BULLET_SLOT) +
        DIM_DISTANCE
    )
    DIM_BULLETS = DIM_BULLET * BULLET_MAX_NUM
    DIM_ENTITY = DIM_ALL_UNITS + DIM_BULLETS

    DIM_TACTIC_COMBAT = 22
    DIM_TACTIC_COMBO = 33
    DIM_TACTIC_LANE = 17
    DIM_TACTIC_CRAB = 10
    DIM_TACTIC_TOWER = 10
    DIM_TACTIC_RECOVERY = 11
    DIM_TACTIC_ACCURACY = 22
    DIM_TACTIC_AMBUSH = 8
    DIM_TACTIC_LUBAN_ULT = 10
    DIM_TACTIC = (
        DIM_TACTIC_COMBAT +
        DIM_TACTIC_COMBO +
        DIM_TACTIC_LANE +
        DIM_TACTIC_CRAB +
        DIM_TACTIC_TOWER +
        DIM_TACTIC_RECOVERY +
        DIM_TACTIC_ACCURACY +
        DIM_TACTIC_AMBUSH +
        DIM_TACTIC_LUBAN_ULT
    )
    # 全部特征
    DIM_ALL = DIM_ENTITY + DIM_TACTIC

# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    # main camp hero
    DIM_OF_HERO_FRD = [Args.DIM_HERO]
    # enemy camp hero
    DIM_OF_HERO_EMY = [Args.DIM_HERO]
    # main camp soldier
    DIM_OF_SOLDIER_1_4 = [Args.DIM_SOLDIER] * 4
    # enemy camp soldier
    DIM_OF_SOLDIER_5_8 = [Args.DIM_SOLDIER] * 4
    # river crab
    DIM_OF_RIVER_CRAB = [Args.DIM_RIVER_CRAB]
    # main camp organ
    DIM_OF_ORGAN_1 = [Args.DIM_ORGAN]
    # enemy camp organ
    DIM_OF_ORGAN_2 = [Args.DIM_ORGAN]
    # bullet
    DIM_OF_BULLET_1_9 = [Args.DIM_BULLET] * 9
    DIM_OF_BULLET_10 = [Args.DIM_BULLET]

# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_DROPOUT = 0
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DIM_PUBLIC = 512  # 将LSTM和旁路MLP的结果合并后执行MLP到512维
    MULTI_HEAD = False  # 今年仅112/133两英雄, 主线沿用共享单模型以保证训练/评估速度

    DATA_SPLIT_SHAPE = [
        Args.DIM_ALL + 85,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]
    SERI_VEC_SPLIT_SHAPE = [(Args.DIM_ALL,), (85,)]
    INIT_LEARNING_RATE_START = 1e-5
    BETA_START = 0
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(Args.DIM_ALL + 85) * 16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.995
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])

if __name__ == '__main__':
    print(Config.SAMPLE_DIM)
    print(Args.DIM_ALL)
    print(Args.DIM_DISTANCE)
    print(Args.DIM_UNIT - Args.DIM_DISTANCE)

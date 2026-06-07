#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
2024年奖励设计
"""

import math
from collections import deque
from agent_ppo.conf.conf import Args, GameConfig


def get_first(data, *keys, default=0):
    for key in keys:
        if isinstance(data, dict) and key in data:
            return data[key]
    return default


def normalize_camp(camp):
    if isinstance(camp, int):
        return camp - 1 if camp in [1, 2] else camp
    if camp == "PLAYERCAMP_MID":
        return -1
    if isinstance(camp, str) and camp.startswith("PLAYERCAMP_"):
        return int(camp[-1]) - 1
    return int(camp) - 1 if str(camp) in ["1", "2"] else int(camp)


def actor_state_of(unit):
    if unit is None:
        return {}
    return unit.get("actor_state", unit)


def unit_location(unit):
    return get_first(actor_state_of(unit), "location", "position", "pos", default={"x": 0, "z": 0})


def unit_runtime_id(unit):
    return get_first(actor_state_of(unit), "runtime_id", "player_id", default=0)


def unit_identity_values(unit):
    if unit is None:
        return []
    actor_state = actor_state_of(unit)
    values = [
        get_first(unit, "runtime_id", default=None),
        get_first(unit, "player_id", default=None),
        get_first(actor_state, "runtime_id", default=None),
        get_first(actor_state, "player_id", default=None),
    ]
    return [value for value in values if value is not None]


def same_identity(left, right):
    return left is not None and right is not None and str(left) == str(right)


def hp_rate_of(unit):
    actor_state = actor_state_of(unit)
    max_hp = get_first(actor_state, "max_hp", "hp_max", default=0)
    return get_first(actor_state, "hp", default=0) / max_hp if max_hp else 0.0


def _location_pair(unit):
    location = unit_location(unit)
    if isinstance(location, dict):
        return (
            get_first(location, "x", default=0),
            get_first(location, "z", "y", default=0),
        )
    if isinstance(location, (list, tuple)) and len(location) >= 2:
        return location[0], location[1]
    return 0, 0


def dist_between(left, right):
    if left is None or right is None:
        return 999999.0
    return math.dist(_location_pair(left), _location_pair(right))


def cake_unit_for_camp(frame_data, camp):
    if camp not in [0, 1]:
        return None
    for cake in frame_data.get("cakes") or []:
        collider = get_first(cake, "collider", default={})
        location = get_first(collider, "location", default=get_first(cake, "location", "position", "pos", default=None))
        if location is None:
            continue
        raw_x = get_first(location, "x", default=location[0] if isinstance(location, (list, tuple)) and location else 0)
        if int(raw_x > 0) == camp:
            return {"location": location}
    return None


def attack_range_of(unit):
    actor_state = actor_state_of(unit)
    attack_range = get_first(actor_state, "attack_range", default=GameConfig.DEFAULT_ATTACK_RANGE)
    return max(float(attack_range or GameConfig.DEFAULT_ATTACK_RANGE), 1.0)


def attack_spacing_score(distance, attack_range):
    if distance > attack_range:
        return 0.0
    ratio = distance / attack_range
    safe_ratio = GameConfig.ATTACK_SPACING_SAFE_RATIO
    ideal_ratio = GameConfig.ATTACK_SPACING_IDEAL_RATIO
    if ratio >= ideal_ratio:
        return 1.0
    if ratio >= safe_ratio:
        return (ratio - safe_ratio) / (ideal_ratio - safe_ratio)
    return -min((safe_ratio - ratio) / safe_ratio, 1.0)


def _int_or_default(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def top_button(last_action):
    if last_action is None:
        return 0
    if isinstance(last_action, dict):
        return _int_or_default(get_first(last_action, "top_button", "button", "action", default=0))
    if hasattr(last_action, "tolist"):
        last_action = last_action.tolist()
    if isinstance(last_action, (list, tuple)) and last_action:
        return _int_or_default(last_action[0])
    return 0


def target_index(last_action):
    if last_action is None:
        return 0
    if isinstance(last_action, dict):
        return _int_or_default(get_first(last_action, "target", "target_index", default=0))
    if hasattr(last_action, "tolist"):
        last_action = last_action.tolist()
    if isinstance(last_action, (list, tuple)) and last_action:
        return _int_or_default(last_action[-1])
    return 0


def skill_state_of(hero):
    actor_state = actor_state_of(hero)
    return get_first(hero, "skill_state", default=get_first(actor_state, "skill_state", default={}))


def skill_slot(hero, button):
    slot_index_map = {4: 1, 5: 2, 6: 3, 8: 5}
    slot_index = slot_index_map.get(button)
    if slot_index is None:
        return {}
    slots = get_first(skill_state_of(hero), "slot_states", default=[])
    return slots[slot_index] if isinstance(slots, list) and len(slots) > slot_index else {}


def is_skill_ready(hero, button):
    slot = skill_slot(hero, button)
    if not isinstance(slot, dict):
        return False
    level = get_first(slot, "level", default=1)
    return bool(get_first(slot, "usable", default=False)) and (button == 8 or level > 0)


def ready_combo_skill_buttons(hero, hero_id):
    priority = GameConfig.SKILL_BUTTON_PRIORITY.get(hero_id, [6, 4, 5])
    return [button for button in priority if is_skill_ready(hero, button)]


def is_skill_3_slot(slot_type):
    if isinstance(slot_type, int):
        return slot_type == 3
    return str(slot_type) in {GameConfig.LUBAN_ULT_SLOT_TYPE, "3"}


def bullet_location_unit(bullet):
    return {"location": get_first(bullet, "location", "position", "pos", default={"x": 0, "z": 0})}

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map(include_tactic=False):
    calc_frame_map = {}
    reward_weight_dict = GameConfig.REWARD_WEIGHT_DICT if include_tactic else GameConfig.BASE_REWARD_WEIGHT_DICT
    for key, weight in reward_weight_dict.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id, main_hero_camp=None):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = normalize_camp(main_hero_camp) if main_hero_camp is not None else -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.attack_results = deque(maxlen=Args.ATTACK_WINDOW_SIZE)
        self.empty_attack_streak = 0
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}
        self.last_cake_recovery_facts = None
        self.last_enemy_luban_ult_facts = None
        self.init_max_exp_of_each_hero()

    def is_main_hero(self, hero):
        return any(same_identity(value, self.main_hero_player_id) for value in unit_identity_values(hero))

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data, last_action=None):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value, last_action=last_action)
        return self.m_reward_value

    def _is_marked_main_hero(self, hero):
        actor_state = actor_state_of(hero)
        return bool(get_first(actor_state, "is_main_hero", "isMainHero", "main_hero", default=False))

    def _get_main_enemy_units(self, frame_data):
        heroes = frame_data.get("hero_states", [])
        main_hero, enemy_hero = None, None
        for hero in heroes:
            if self._is_marked_main_hero(hero) or self.is_main_hero(hero):
                main_hero = hero
                break
        if main_hero is None and self.main_hero_camp in [0, 1]:
            for hero in heroes:
                hero_camp = normalize_camp(actor_state_of(hero).get("camp", -1))
                if hero_camp == self.main_hero_camp:
                    main_hero = hero
                    break
        if main_hero is None and heroes:
            main_hero = heroes[0]
        for hero in heroes:
            if hero is not main_hero:
                enemy_hero = hero
                break

        main_camp = normalize_camp(actor_state_of(main_hero).get("camp", self.main_hero_camp)) if main_hero else self.main_hero_camp
        enemy_camp = normalize_camp(actor_state_of(enemy_hero).get("camp", 1 - main_camp)) if enemy_hero else 1 - main_camp

        units = {
            "main_hero": main_hero,
            "enemy_hero": enemy_hero,
            "main_tower": None,
            "enemy_tower": None,
            "our_soldiers": [],
            "enemy_soldiers": [],
            "crab": None,
            "our_cake": None,
            "enemy_cake": None,
            "id2type": {},
        }
        for hero, unit_type in [(main_hero, "hero"), (enemy_hero, "hero")]:
            if hero is not None:
                units["id2type"][unit_runtime_id(hero)] = unit_type

        for npc in frame_data.get("npc_states", []):
            actor_state = actor_state_of(npc)
            npc_camp = normalize_camp(actor_state.get("camp", -1))
            sub_type = actor_state.get("sub_type", None)
            config_id = get_first(actor_state, "config_id", default=get_first(npc, "config_id", default=0))
            runtime_id = unit_runtime_id(npc)
            is_tower = sub_type in ["ACTOR_SUB_TOWER", 21] or config_id == 21
            is_soldier = sub_type in ["ACTOR_SUB_SOLDIER", 1] or config_id in [6800, 6801, 6802, 6803, 6804, 6805]
            is_crab = config_id == 6827 or sub_type in ["ACTOR_SUB_RIVER_CRAB", "ACTOR_SUB_MONSTER", "ACTOR_SUB_WILD_MONSTER"]
            if is_crab:
                units["crab"] = npc
                units["id2type"][runtime_id] = "crab"
            elif is_tower:
                units["id2type"][runtime_id] = "tower"
                if npc_camp == main_camp:
                    units["main_tower"] = npc
                elif npc_camp == enemy_camp:
                    units["enemy_tower"] = npc
            elif is_soldier:
                units["id2type"][runtime_id] = "soldier"
                if npc_camp == main_camp:
                    units["our_soldiers"].append(npc)
                elif npc_camp == enemy_camp:
                    units["enemy_soldiers"].append(npc)
        units["our_cake"] = cake_unit_for_camp(frame_data, main_camp)
        units["enemy_cake"] = cake_unit_for_camp(frame_data, enemy_camp)
        return units

    def _settle_attack_accuracy(self, is_attack_button, target_is_valid):
        if not is_attack_button:
            return 0.0
        self.attack_results.append(bool(target_is_valid))
        if target_is_valid:
            self.empty_attack_streak = 0
        else:
            self.empty_attack_streak += 1

        min_attempts = min(4, self.attack_results.maxlen or 4)
        if len(self.attack_results) < min_attempts:
            return 0.0
        hit_rate = sum(1 for result in self.attack_results if result) / len(self.attack_results)
        if hit_rate >= 0.65:
            return 0.4
        if hit_rate <= 0.25 or self.empty_attack_streak >= min_attempts:
            return -0.3
        return 0.0

    def _settle_cake_recovery(
        self,
        cake_available,
        cake_distance,
        self_hp,
        enemy_hp,
        enemy_distance,
        enemy_hero_in_attack_range,
        button,
        target,
        is_attack_button,
        tower_danger,
        tower_aggro_risk,
    ):
        reward = 0.0
        last_facts = self.last_cake_recovery_facts

        if last_facts is not None and last_facts["cake_available"] and not cake_available:
            hp_gain = self_hp - last_facts["self_hp"]
            if last_facts["cake_distance"] <= GameConfig.CAKE_CONSUME_DISTANCE and hp_gain >= GameConfig.CAKE_HP_GAIN_THRESHOLD:
                if last_facts["self_hp"] <= GameConfig.CAKE_LOW_HP_THRESHOLD:
                    reward = max(reward, GameConfig.CAKE_LOW_CONSUME_REWARD)
                elif last_facts["self_hp"] <= GameConfig.CAKE_MID_HP_THRESHOLD:
                    reward = max(reward, GameConfig.CAKE_MID_CONSUME_REWARD)

        kill_window = enemy_hp <= 0.20 and enemy_hero_in_attack_range and self_hp >= 0.45
        if cake_available and not tower_danger and not tower_aggro_risk and not kill_window:
            low_hp_seek = self_hp <= GameConfig.CAKE_LOW_HP_THRESHOLD and cake_distance <= GameConfig.CAKE_SEEK_MAX_DISTANCE
            mid_hp_seek = (
                GameConfig.CAKE_LOW_HP_THRESHOLD < self_hp <= GameConfig.CAKE_MID_HP_THRESHOLD
                and cake_distance <= 7000
                and (enemy_hp <= 0.0 or enemy_distance > 6500)
            )
            if last_facts is not None and last_facts["cake_available"]:
                distance_delta = last_facts["cake_distance"] - cake_distance
                if low_hp_seek:
                    if distance_delta >= GameConfig.CAKE_APPROACH_DELTA:
                        reward += GameConfig.CAKE_LOW_APPROACH_REWARD
                    elif distance_delta <= -GameConfig.CAKE_APPROACH_DELTA:
                        reward += GameConfig.CAKE_LOW_RETREAT_PENALTY
                elif mid_hp_seek and distance_delta >= GameConfig.CAKE_APPROACH_DELTA:
                    reward += GameConfig.CAKE_MID_APPROACH_REWARD

            if low_hp_seek and cake_distance <= GameConfig.CAKE_NEAR_DISTANCE:
                reward += GameConfig.CAKE_LOW_NEAR_REWARD
            elif mid_hp_seek and cake_distance <= GameConfig.CAKE_NEAR_DISTANCE:
                reward += GameConfig.CAKE_MID_NEAR_REWARD

            if low_hp_seek and is_attack_button and target in [1, 7]:
                reward += GameConfig.CAKE_LOW_BAD_COMBAT_PENALTY

        self.last_cake_recovery_facts = {
            "cake_available": cake_available,
            "cake_distance": cake_distance,
            "self_hp": self_hp,
        }
        return reward

    def _enemy_luban_ult_distance(self, frame_data, main_hero, enemy_hero, enemy_camp):
        enemy_actor = actor_state_of(enemy_hero)
        if get_first(enemy_actor, "config_id", default=0) != GameConfig.LUBAN_HERO_ID:
            return False, 999999.0
        enemy_runtime_id = unit_runtime_id(enemy_hero)
        distances = []
        for bullet in frame_data.get("bullets") or []:
            if normalize_camp(get_first(bullet, "camp", default=enemy_camp)) != enemy_camp:
                continue
            source_actor = get_first(bullet, "source_actor", "sourceActor", "source_id", default=enemy_runtime_id)
            if enemy_runtime_id and not same_identity(source_actor, enemy_runtime_id):
                continue
            if not is_skill_3_slot(get_first(bullet, "slot_type", "slotType", default=None)):
                continue
            distances.append(dist_between(main_hero, bullet_location_unit(bullet)))
        return bool(distances), min(distances) if distances else 999999.0

    def _settle_enemy_luban_ult_safety(self, active, danger_distance, self_hp):
        reward = 0.0
        last_facts = self.last_enemy_luban_ult_facts
        inside = active and danger_distance <= GameConfig.LUBAN_ULT_DANGER_RADIUS
        near = active and danger_distance <= GameConfig.LUBAN_ULT_WARNING_RADIUS

        if inside:
            reward += GameConfig.LUBAN_ULT_STAY_PENALTY
            if self_hp <= GameConfig.CAKE_LOW_HP_THRESHOLD:
                reward += GameConfig.LUBAN_ULT_LOW_HP_EXTRA_PENALTY

        if near and last_facts is not None and last_facts["active"]:
            distance_delta = danger_distance - last_facts["distance"]
            if distance_delta >= GameConfig.LUBAN_ULT_ESCAPE_DELTA:
                reward += GameConfig.LUBAN_ULT_ESCAPE_REWARD
            elif distance_delta <= -GameConfig.LUBAN_ULT_ESCAPE_DELTA:
                reward += GameConfig.LUBAN_ULT_APPROACH_PENALTY

        if active and not inside and last_facts is not None and last_facts["inside"]:
            reward += GameConfig.LUBAN_ULT_EXIT_REWARD

        self.last_enemy_luban_ult_facts = {
            "active": active,
            "inside": inside,
            "distance": danger_distance,
        }
        return reward

    def calculate_tactic_rewards(self, frame_data, last_action=None):
        tactic_rewards = {name: 0.0 for name in GameConfig.TACTIC_REWARD_WEIGHT_DICT}
        units = self._get_main_enemy_units(frame_data)
        main_hero = units["main_hero"]
        enemy_hero = units["enemy_hero"]
        enemy_tower = units["enemy_tower"]
        our_cake = units["our_cake"]
        crab = units["crab"]
        enemy_soldiers = [unit for unit in units["enemy_soldiers"] if hp_rate_of(unit) > 0]
        our_soldiers = [unit for unit in units["our_soldiers"] if hp_rate_of(unit) > 0]

        enemy_distance = dist_between(main_hero, enemy_hero)
        self_hp = hp_rate_of(main_hero)
        enemy_hp = hp_rate_of(enemy_hero)
        main_actor = actor_state_of(main_hero)
        enemy_actor = actor_state_of(enemy_hero)
        enemy_camp = normalize_camp(get_first(enemy_actor, "camp", default=1 - self.main_hero_camp))
        lane_cleared = len(enemy_soldiers) == 0
        self_attack_range = attack_range_of(main_hero)
        enemy_hero_in_attack_range = self_hp > 0 and enemy_hp > 0 and enemy_distance <= self_attack_range
        trade_active = enemy_hero_in_attack_range
        enemy_soldier_in_attack_range = any(
            dist_between(main_hero, soldier) <= self_attack_range for soldier in enemy_soldiers
        )
        hero_priority_window = enemy_hero_in_attack_range and enemy_soldier_in_attack_range
        hero_id = get_first(main_actor, "config_id", default=0)
        ready_skill_buttons = ready_combo_skill_buttons(main_hero, hero_id)
        skill_window = self_hp > 0 and enemy_hp > 0 and enemy_distance <= GameConfig.SKILL_URGENCY_DISTANCE
        enemy_dead = enemy_hp <= 0.0
        enemy_dead_or_far = enemy_dead or enemy_distance > 9000
        self_low_hp = self_hp <= GameConfig.CAKE_LOW_HP_THRESHOLD
        crab_safe_to_attack = crab is not None and hp_rate_of(crab) > 0 and self_hp > 0.45 and (enemy_dead_or_far or enemy_distance > 6500)
        cake_available = our_cake is not None
        cake_distance = dist_between(main_hero, our_cake) if cake_available else 999999.0

        enemy_tower_actor = actor_state_of(enemy_tower) if enemy_tower is not None else {}
        enemy_tower_hp = get_first(enemy_tower_actor, "hp", default=0)
        enemy_tower_alive = enemy_tower is not None and enemy_tower_hp > 0
        tower_attack_range = get_first(enemy_tower_actor, "attack_range", default=8500)
        attack_target = get_first(enemy_tower_actor, "attack_target", "attack_target_id", default=0)
        tower_target_type = units["id2type"].get(attack_target, "none")
        our_soldier_in_enemy_tower = any(dist_between(soldier, enemy_tower) <= tower_attack_range for soldier in our_soldiers)
        self_in_enemy_tower_range = dist_between(main_hero, enemy_tower) <= tower_attack_range
        tower_attack_window = our_soldier_in_enemy_tower and tower_target_type == "soldier" and enemy_dead_or_far
        tower_danger = enemy_tower_alive and self_in_enemy_tower_range and tower_target_type == "hero"
        no_minion_tower_dive = enemy_tower_alive and self_in_enemy_tower_range and not our_soldier_in_enemy_tower
        safe_recall_window = lane_cleared and enemy_dead and self_low_hp

        self_in_grass = bool(get_first(main_actor, "flag_in_grass", "in_grass", "is_in_grass", default=False))
        enemy_near_grass = self_in_grass and enemy_distance <= 6500 and enemy_hp > 0
        button = top_button(last_action)
        target = target_index(last_action)
        is_skill_button = button in [4, 5, 6]
        is_attack_button = button in [Args.NORMAL_ATTACK_BUTTON, 4, 5, 6]
        tower_aggro_risk = (
            enemy_tower_alive
            and self_in_enemy_tower_range
            and enemy_hp > 0
            and is_attack_button
            and target == 1
        )
        normal_attack_target_valid = button == Args.NORMAL_ATTACK_BUTTON and target == 1 and enemy_hero_in_attack_range
        skill_target_valid = is_skill_button and target == 1 and enemy_hp > 0 and enemy_distance <= GameConfig.SKILL_URGENCY_DISTANCE
        target_is_valid = (
            normal_attack_target_valid
            or skill_target_valid
            or (target in [3, 4, 5, 6] and bool(enemy_soldiers))
            or (target == 7 and enemy_tower is not None)
            or (target == 8 and crab is not None and hp_rate_of(crab) > 0)
        )
        preferred_skill_button = ready_skill_buttons[0] if ready_skill_buttons else None
        luban_ult_active, luban_ult_distance = self._enemy_luban_ult_distance(
            frame_data,
            main_hero,
            enemy_hero,
            enemy_camp,
        )

        if button == Args.USE_FRENZY_BUTTON and trade_active:
            tactic_rewards["frenzy_trade"] = 1.0
        if is_attack_button and target in [3, 4, 5, 6] and enemy_soldiers and not hero_priority_window:
            tactic_rewards["lane_clear"] = 1.0
        if button == Args.NORMAL_ATTACK_BUTTON and hero_priority_window and not tower_aggro_risk:
            if target == 1:
                tactic_rewards["hero_target_priority"] = 1.0
            elif target in [3, 4, 5, 6]:
                tactic_rewards["hero_target_priority"] = -1.0
        if lane_cleared and is_attack_button and target in [1, 7] and not no_minion_tower_dive and not tower_aggro_risk and not tower_danger:
            tactic_rewards["lane_push"] = 1.0
        if is_attack_button and target == 8:
            tactic_rewards["river_crab_control"] = 1.0 if crab_safe_to_attack else -0.5
        if button == Args.NORMAL_ATTACK_BUTTON and target == 1 and enemy_hp > 0:
            tactic_rewards["attack_spacing"] = -1.0 if tower_aggro_risk else attack_spacing_score(enemy_distance, self_attack_range)
        combo_wrong_target = target in [0, 2, 7, 8] or not target_is_valid
        if skill_window and ready_skill_buttons:
            if tower_aggro_risk:
                tactic_rewards["skill_urgency"] = -1.0
            elif button == preferred_skill_button and target == 1:
                tactic_rewards["skill_urgency"] = 1.4 if button == 6 else 1.0
            elif is_skill_button and button in ready_skill_buttons and target == 1:
                tactic_rewards["skill_urgency"] = 0.7
            elif button == Args.NORMAL_ATTACK_BUTTON and target == 1:
                tactic_rewards["skill_urgency"] = -0.6
            elif button in [1, 2] or (is_skill_button and combo_wrong_target):
                tactic_rewards["skill_urgency"] = -0.5
        if hero_id == 112 and trade_active and is_attack_button:
            if tower_aggro_risk:
                tactic_rewards["combo_luban"] = -0.8
            elif is_skill_button and target == 1:
                tactic_rewards["combo_luban"] = 1.0 if button == 6 else 0.8
            elif button == Args.NORMAL_ATTACK_BUTTON and target == 1:
                tactic_rewards["combo_luban"] = 0.25 if ready_skill_buttons else 0.6
            elif combo_wrong_target:
                tactic_rewards["combo_luban"] = -0.5
        if hero_id == 133 and trade_active and is_attack_button:
            if tower_aggro_risk:
                tactic_rewards["combo_direnjie"] = -0.8
            elif button == 6 and target == 1:
                tactic_rewards["combo_direnjie"] = 1.4
            elif button == 4 and target == 1:
                tactic_rewards["combo_direnjie"] = 0.9
            elif button == Args.NORMAL_ATTACK_BUTTON and target == 1:
                tactic_rewards["combo_direnjie"] = 0.15 if ready_skill_buttons else 0.5
            elif combo_wrong_target:
                tactic_rewards["combo_direnjie"] = -0.5
        tactic_rewards["attack_accuracy"] = self._settle_attack_accuracy(is_attack_button, target_is_valid)
        if enemy_near_grass and is_attack_button and target == 1 and not tower_aggro_risk:
            tactic_rewards["grass_ambush"] = 1.0
        if tower_danger:
            tactic_rewards["tower_safety"] = GameConfig.TOWER_AGGRO_TARGETED_PENALTY
        elif tower_aggro_risk:
            tactic_rewards["tower_safety"] = GameConfig.TOWER_AGGRO_RISK_PENALTY
        elif no_minion_tower_dive:
            tactic_rewards["tower_safety"] = -1.2 if tower_danger else -1.0
        if tactic_rewards["tower_safety"] < 0 and self_low_hp:
            tactic_rewards["tower_safety"] += GameConfig.TOWER_LOW_HP_EXTRA_PENALTY
        if button == Args.NORMAL_ATTACK_BUTTON and target == 7:
            if tower_attack_window:
                tactic_rewards["tower_push"] = 1.0
            elif no_minion_tower_dive:
                tactic_rewards["tower_push"] = -1.0
            elif tower_danger:
                tactic_rewards["tower_push"] = -0.5
        if button == Args.RECOVER_BUTTON:
            tactic_rewards["recovery_choice"] = 1.0 if self_low_hp and not safe_recall_window else 0.2
        if button == Args.RECALL_BUTTON:
            if safe_recall_window:
                tactic_rewards["safe_recall"] = 1.0
            elif enemy_hp > 0.0:
                tactic_rewards["safe_recall"] = GameConfig.RECALL_ENEMY_ALIVE_PENALTY
            elif not lane_cleared:
                tactic_rewards["safe_recall"] = GameConfig.RECALL_LANE_UNCLEARED_PENALTY
            else:
                tactic_rewards["safe_recall"] = GameConfig.RECALL_HEALTHY_PENALTY
        tactic_rewards["cake_recovery"] = self._settle_cake_recovery(
            cake_available,
            cake_distance,
            self_hp,
            enemy_hp,
            enemy_distance,
            enemy_hero_in_attack_range,
            button,
            target,
            is_attack_button,
            tower_danger,
            tower_aggro_risk,
        )
        tactic_rewards["enemy_luban_ult_safety"] = self._settle_enemy_luban_ult_safety(
            luban_ult_active,
            luban_ult_distance,
            self_hp,
        )
        return tactic_rewards

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = normalize_camp(actor_state_of(hero)["camp"])
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_actor = actor_state_of(main_hero)
        main_hero_hp = get_first(main_hero_actor, "hp", default=0)
        main_hero_max_hp = get_first(main_hero_actor, "max_hp", default=0)
        main_hero_values = get_first(main_hero_actor, "values", default=main_hero_actor)
        main_hero_ep = get_first(main_hero_values, "ep", default=0)
        main_hero_max_ep = get_first(main_hero_values, "max_ep", default=0)

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_actor = actor_state_of(organ)
            organ_camp = normalize_camp(organ_actor["camp"])
            organ_subtype = organ_actor["sub_type"]
            if organ_camp == camp:
                if organ_subtype in ["ACTOR_SUB_TOWER", 21]:  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype in ["ACTOR_SUB_CRYSTAL", 24]:  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype in ["ACTOR_SUB_TOWER", 21]:  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype in ["ACTOR_SUB_CRYSTAL", 24]:  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                reward_struct.cur_frame_value = get_first(main_hero, "moneyCnt", "money_cnt", "money", default=0)
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp)) if main_hero_max_hp else 0
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = get_first(main_hero, "killCnt", "kill_cnt")
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = get_first(main_hero, "deadCnt", "dead_cnt")
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                main_tower_actor = actor_state_of(main_tower) if main_tower is not None else {}
                tower_max_hp = get_first(main_tower_actor, "max_hp", default=0)
                reward_struct.cur_frame_value = 1.0 * get_first(main_tower_actor, "hp", default=0) / tower_max_hp if tower_max_hp else 0
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data.get("frame_action", {})
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == unit_runtime_id(main_hero)
                            and dead_action["death"]["sub_type"] in ["ACTOR_SUB_SOLDIER", 1]
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == unit_runtime_id(enemy_hero)
                            and dead_action["death"]["sub_type"] in ["ACTOR_SUB_SOLDIER", 1]
                        ):
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, get_first(this_hero_info, "level", default=1)):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += get_first(this_hero_info, "exp", default=0)
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        if main_tower is None or enemy_tower is None:
            return 0
        main_tower_location = unit_location(main_tower)
        enemy_tower_location = unit_location(enemy_tower)
        hero_location = unit_location(main_hero)
        main_tower_pos = (main_tower_location["x"], main_tower_location["z"])
        enemy_tower_pos = (enemy_tower_location["x"], enemy_tower_location["z"])
        hero_pos = (hero_location["x"], hero_location["z"])
        forward_value = 0
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        main_hero_actor = actor_state_of(main_hero)
        hero_max_hp = get_first(main_hero_actor, "max_hp", default=0)
        hero_hp_rate = get_first(main_hero_actor, "hp", default=0) / hero_max_hp if hero_max_hp else 0
        if hero_hp_rate > 0.99 and dist_hero2emy > dist_main2emy:
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy
        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            hero_actor = actor_state_of(hero)
            hero_camp = normalize_camp(hero_actor["camp"])
            if self.is_main_hero(hero) or (self.main_hero_camp in [0, 1] and hero_camp == self.main_hero_camp):
                main_camp = normalize_camp(hero_actor["camp"])
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero_camp
        if enemy_camp == -1 and main_camp in [0, 1]:
            enemy_camp = 1 - main_camp
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict, last_action=None):
        reward_dict.clear()
        frame_no = get_first(frame_data, "frameNo", "frame_no")
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if self.is_main_hero(hero):
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                if GameConfig.REMOVE_FORWARD_AFTER is not None:
                    reward_struct.value *= (frame_no <= GameConfig.REMOVE_FORWARD_AFTER)
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight

            time_scale = 1.0
            
            if self.time_scale_arg > 0:
                if reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                    time_scale = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

            reward_dict[reward_name+"_origin"] = reward_struct.value
            reward_dict[reward_name+"_weight"] = reward_struct.value * reward_struct.weight * time_scale
            reward_sum += reward_dict[reward_name+"_weight"]
            
        tactic_rewards = self.calculate_tactic_rewards(frame_data, last_action=last_action)
        for reward_name, reward_value in tactic_rewards.items():
            time_scale = 1.0
            if self.time_scale_arg > 0:
                if reward_name not in GameConfig.REWARD_WITHOUT_TIME_SCALE:
                    time_scale = math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
            reward_dict[reward_name + "_origin"] = reward_value
            reward_dict[reward_name + "_weight"] = (
                reward_value * GameConfig.TACTIC_REWARD_WEIGHT_DICT[reward_name] * time_scale
            )
            reward_sum += reward_dict[reward_name + "_weight"]

        reward_dict["reward_sum"] = reward_sum

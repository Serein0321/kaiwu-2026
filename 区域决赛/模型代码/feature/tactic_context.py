import math
from collections import deque

from agent_ppo.conf.conf import Args, GameConfig
from agent_ppo.feature.unpack_state_dict import Info


UNSEEN_DISTANCE = 999999.0


def safe_dist(left, right):
    if left is None or right is None:
        return UNSEEN_DISTANCE
    if left[0] == Info.UNSEEN_PADDING or right[0] == Info.UNSEEN_PADDING:
        return UNSEEN_DISTANCE
    return math.dist(left, right)


def hp_rate(unit):
    hp_max = getattr(unit, "hp_max", 0)
    return getattr(unit, "hp", 0) / hp_max if hp_max else 0.0


def one_hot(index, size):
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def bucket(value, cuts):
    for index, cut in enumerate(cuts):
        if value <= cut:
            return index
    return len(cuts)


def hp_bucket(rate):
    return bucket(rate, [0.25, 0.5, 0.75])


def distance_bucket(distance):
    if distance >= UNSEEN_DISTANCE:
        return 4
    return bucket(distance, [1400, 4500, 8500, 16000])


def compact_distance_bucket(distance):
    if distance >= UNSEEN_DISTANCE:
        return 3
    return bucket(distance, [1800, 6500, 14000])


def is_skill_3_slot(slot_type):
    if isinstance(slot_type, int):
        return slot_type == 3
    return str(slot_type) in {GameConfig.LUBAN_ULT_SLOT_TYPE, "3"}


class TacticContext:
    COMBO_PHASES = Args.COMBO_PHASES

    def __init__(self, window_size=None):
        self.window_size = window_size or Args.ATTACK_WINDOW_SIZE
        self.reset()

    def reset(self):
        self.last_action = None
        self.last_frame = -1
        self.combo_phase = "idle"
        self.combo_started_frame = -1
        self.grass_wait_frames = 0
        self.ambush_exit_frames = 0
        self.normal_results = deque(maxlen=self.window_size)
        self.skill_results = deque(maxlen=self.window_size)
        self.empty_attack_streak = 0
        self.last_facts = {}

    @property
    def normal_attack_hit_rate(self):
        return self._hit_rate(self.normal_results)

    @property
    def skill_hit_rate(self):
        return self._hit_rate(self.skill_results)

    def _hit_rate(self, results):
        if not results:
            return 1.0
        return sum(1 for item in results if item) / len(results)

    def record_action(self, action):
        self.last_action = list(action) if action is not None else None

    def _last_button(self):
        if isinstance(self.last_action, (list, tuple)) and self.last_action:
            return self.last_action[0]
        return 0

    def record_attack_result(self, hit, slot_type):
        if slot_type == "SLOT_SKILL_0":
            self.normal_results.append(bool(hit))
        else:
            self.skill_results.append(bool(hit))
        if hit:
            self.empty_attack_streak = 0
        else:
            self.empty_attack_streak += 1

    def build_features(self, info):
        facts = self._collect_facts(info)
        self._update_combo(info, facts)
        features = []
        features.extend(self._combat_features(facts))
        features.extend(self._combo_features(info, facts))
        features.extend(self._lane_features(facts))
        features.extend(self._crab_features(facts))
        features.extend(self._tower_features(facts))
        features.extend(self._recovery_features(facts))
        features.extend(self._accuracy_features())
        features.extend(self._ambush_features(facts))
        features.extend(self._luban_ult_features(facts))
        if len(features) != Args.DIM_TACTIC:
            raise ValueError(f"tactic feature length mismatch: {len(features)} != {Args.DIM_TACTIC}")
        self.last_facts = facts
        return features

    def _enemy_luban_ult_distance(self, info, our_actor, enemy_actor):
        if getattr(enemy_actor, "config_id", 0) != GameConfig.LUBAN_HERO_ID:
            return UNSEEN_DISTANCE
        hero_bullets = getattr(getattr(info, "bullets_enemy", None), "hero", [])
        distances = [
            safe_dist(our_actor.position, bullet.position)
            for bullet in hero_bullets
            if is_skill_3_slot(getattr(bullet, "slot_type", None))
        ]
        return min(distances) if distances else UNSEEN_DISTANCE

    def _collect_facts(self, info):
        our = info.hero_our
        enemy = info.hero_enemy
        our_actor = our.info
        enemy_actor = enemy.info
        enemy_distance = safe_dist(our_actor.position, enemy_actor.position)
        enemy_visible = enemy_distance < UNSEEN_DISTANCE
        self_rate = hp_rate(our_actor)
        enemy_rate = hp_rate(enemy_actor)
        self_attack_range = max(float(getattr(our_actor, "attack_range", 0) or GameConfig.DEFAULT_ATTACK_RANGE), 1.0)
        enemy_in_attack_range = enemy_visible and enemy_distance <= self_attack_range and self_rate > 0 and enemy_rate > 0
        enemy_soldiers = list(getattr(info.soldiers_enemy, "merge", []))
        our_soldiers = list(getattr(info.soldiers_our, "merge", []))
        nearest_enemy_hp_rate = 0.0
        if enemy_soldiers:
            nearest = min(enemy_soldiers, key=lambda unit: safe_dist(our_actor.position, unit.position))
            nearest_enemy_hp_rate = hp_rate(nearest)
        crab = getattr(info, "river_crab", None)
        crab_position = getattr(crab, "position", None) if crab is not None else None
        crab_distance = safe_dist(our_actor.position, crab_position) if crab is not None else UNSEEN_DISTANCE
        enemy_tower = info.organ_enemy.sub_tower
        self_in_enemy_tower = safe_dist(our_actor.position, enemy_tower.position) <= getattr(enemy_tower, "attack_range", 0)
        tower_target_type = getattr(info, "id2type", {}).get(getattr(enemy_tower, "attack_target", 0), "none")
        our_soldier_in_enemy_tower = any(
            safe_dist(unit.position, enemy_tower.position) <= getattr(enemy_tower, "attack_range", 0)
            for unit in our_soldiers
        )
        low_hp = self_rate <= 0.35
        lane_cleared = len(enemy_soldiers) == 0
        enemy_dead_or_far = enemy_rate <= 0.0 or enemy_distance > 9000
        frenzy_slot = getattr(our.skill, "summoner", getattr(our.skill, "flash", None))
        luban_ult_distance = self._enemy_luban_ult_distance(info, our_actor, enemy_actor)
        luban_ult_active = luban_ult_distance < UNSEEN_DISTANCE
        luban_ult_inside = luban_ult_active and luban_ult_distance <= GameConfig.LUBAN_ULT_DANGER_RADIUS
        luban_ult_near = luban_ult_active and luban_ult_distance <= GameConfig.LUBAN_ULT_WARNING_RADIUS
        last_luban_distance = self.last_facts.get("enemy_luban_ult_distance", luban_ult_distance)
        luban_ult_moving_away = (
            luban_ult_active
            and self.last_facts.get("enemy_luban_ult_active", False)
            and luban_ult_distance - last_luban_distance >= GameConfig.LUBAN_ULT_ESCAPE_DELTA
        )
        facts = {
            "frame": getattr(info, "n_frame", 0),
            "hero_id": getattr(our_actor, "config_id", 0),
            "enemy_visible": enemy_visible,
            "enemy_distance": enemy_distance,
            "self_hp_rate": self_rate,
            "enemy_hp_rate": enemy_rate,
            "hp_advantage": self_rate - enemy_rate,
            "trade_active": enemy_in_attack_range,
            "frenzy_ready": bool(getattr(frenzy_slot, "usable", False)),
            "frenzy_active": GameConfig.SUMMONER_SKILLS["frenzy"] in getattr(our_actor.buff, "skill_ids", []),
            "enemy_soldier_count": len(enemy_soldiers),
            "our_soldier_count": len(our_soldiers),
            "nearest_enemy_hp_rate": nearest_enemy_hp_rate,
            "lane_cleared": lane_cleared,
            "lane_advancing": our_soldier_in_enemy_tower or (len(our_soldiers) > len(enemy_soldiers)),
            "last_hit_window": 0 < nearest_enemy_hp_rate <= 0.25,
            "crab_visible": crab is not None and crab_distance < UNSEEN_DISTANCE,
            "crab_distance": crab_distance,
            "crab_hp_rate": hp_rate(crab) if crab is not None else 0.0,
            "crab_safe_to_attack": crab is not None and self_rate > 0.45 and (enemy_dead_or_far or enemy_distance > 6500),
            "our_soldier_in_enemy_tower": our_soldier_in_enemy_tower,
            "enemy_tower_target_type": tower_target_type,
            "self_in_enemy_tower_range": self_in_enemy_tower,
            "tower_attack_window": our_soldier_in_enemy_tower and tower_target_type == "soldier" and enemy_dead_or_far,
            "tower_danger": self_in_enemy_tower and tower_target_type == "hero",
            "self_low_hp": low_hp,
            "self_low_ep": getattr(our_actor, "ep", 0) <= 60,
            "recover_ready": bool(getattr(our.skill.recover, "usable", False)),
            "recall_ready": bool(getattr(our.skill.back, "usable", False)),
            "our_cake_available": getattr(info, "cake_our", None) is not None,
            "cake_distance": safe_dist(our_actor.position, getattr(getattr(info, "cake_our", None), "position", None)),
            "enemy_dead_or_far": enemy_dead_or_far,
            "safe_recall_window": lane_cleared and enemy_dead_or_far and low_hp,
            "self_in_grass": bool(getattr(our, "flag_in_grass", False)),
            "enemy_near_grass": bool(getattr(our, "flag_in_grass", False)) and enemy_visible and enemy_distance <= 6500,
            "enemy_luban_ult_active": luban_ult_active,
            "enemy_luban_ult_distance": luban_ult_distance,
            "enemy_luban_ult_inside": luban_ult_inside,
            "enemy_luban_ult_near": luban_ult_near,
            "enemy_luban_ult_moving_away": luban_ult_moving_away,
        }
        return facts

    def _update_combo(self, info, facts):
        frame = facts["frame"]
        if self.combo_started_frame >= 0 and frame - self.combo_started_frame > Args.COMBO_TIMEOUT_FRAMES:
            self.combo_phase = "failed_timeout"
        if self.last_action is None:
            return
        button = self._last_button()
        if button == 0:
            return
        hero_id = facts["hero_id"]
        skill_buttons = {4, 5, 6}
        if hero_id == 112:
            if button == 6 and facts["trade_active"]:
                self.combo_phase = "opener"
                self.combo_started_frame = frame
            elif button in skill_buttons:
                self.combo_phase = "expect_normal_attack"
            elif button == Args.NORMAL_ATTACK_BUTTON and self.combo_phase in {"opener", "expect_normal_attack"}:
                self.combo_phase = "follow_up_skill"
        elif hero_id == 133:
            if button == 6 and facts["trade_active"]:
                self.combo_phase = "opener"
                self.combo_started_frame = frame
            elif button == 4 and self.combo_phase == "opener":
                self.combo_phase = "follow_up_skill"
            elif button == Args.NORMAL_ATTACK_BUTTON and self.combo_phase in {"opener", "follow_up_skill"}:
                self.combo_phase = "finish"

    def _combat_features(self, facts):
        values = [float(facts["enemy_visible"])]
        values.extend(one_hot(distance_bucket(facts["enemy_distance"]), 5))
        values.extend(one_hot(hp_bucket(facts["self_hp_rate"]), 4))
        values.extend(one_hot(hp_bucket(facts["enemy_hp_rate"]), 4))
        values.extend(one_hot(bucket(facts["hp_advantage"], [-0.5, -0.15, 0.15, 0.5]), 5))
        values.extend([float(facts["trade_active"]), float(facts["frenzy_ready"]), float(facts["frenzy_active"])])
        return values

    def _combo_features(self, info, facts):
        hero_value = Args.HERO_FEATURE_VALUE.get(facts["hero_id"], 0.0)
        phase_index = self.COMBO_PHASES.index(self.combo_phase) if self.combo_phase in self.COMBO_PHASES else 0
        button = self._last_button()
        skills = info.hero_our.skill
        skill_flags = [
            bool(skills.first.usable),
            bool(skills.second.usable),
            bool(skills.thrid.usable),
            bool(skills.recover.usable),
            bool(skills.summoner.usable),
            bool(skills.back.usable),
        ]
        passive_layer = 0
        marks_ids = getattr(info.hero_our.info.buff, "marks_ids", [])
        marks_layers = getattr(info.hero_our.info.buff, "marks_layers", [])
        for mark_id, layer in zip(marks_ids, marks_layers):
            if mark_id in Args.MARK_ID_LAYERS:
                passive_layer = min(max(layer, 0), Args.MARK_ID_LAYERS[mark_id])
                passive_layer = min(passive_layer, 5)
                break
        values = [hero_value]
        values.extend(one_hot(phase_index, Args.COMBO_PHASE_DIM))
        values.extend(one_hot(button, 12))
        values.extend([float(flag) for flag in skill_flags])
        values.extend(one_hot(passive_layer, 6))
        values.append(float(self.normal_attack_hit_rate < 1.0 or self.skill_hit_rate < 1.0))
        return values

    def _lane_features(self, facts):
        values = []
        values.extend(one_hot(min(facts["enemy_soldier_count"], 4), 5))
        values.extend(one_hot(min(facts["our_soldier_count"], 4), 5))
        values.extend(one_hot(hp_bucket(facts["nearest_enemy_hp_rate"]), 4))
        values.extend([float(facts["lane_cleared"]), float(facts["lane_advancing"]), float(facts["last_hit_window"])])
        return values

    def _crab_features(self, facts):
        values = [float(facts["crab_visible"])]
        values.extend(one_hot(compact_distance_bucket(facts["crab_distance"]), 4))
        values.extend(one_hot(hp_bucket(facts["crab_hp_rate"]), 4))
        values.append(float(facts["crab_safe_to_attack"]))
        return values

    def _tower_features(self, facts):
        target_map = {"none": 0, "hero": 1, "soldier": 2}
        target_index = target_map.get(facts["enemy_tower_target_type"], 3)
        danger_index = 2 if facts["tower_danger"] else (0 if facts["tower_attack_window"] else 1)
        values = [float(facts["our_soldier_in_enemy_tower"])]
        values.extend(one_hot(target_index, 4))
        values.append(float(facts["self_in_enemy_tower_range"]))
        values.extend(one_hot(danger_index, 3))
        values.append(float(facts["tower_attack_window"]))
        return values

    def _recovery_features(self, facts):
        values = [
            float(facts["self_low_hp"]),
            float(facts["self_low_ep"]),
            float(facts["recover_ready"]),
            float(facts["recall_ready"]),
            float(facts["our_cake_available"]),
        ]
        values.extend(one_hot(compact_distance_bucket(facts["cake_distance"]), 4))
        values.extend([float(facts["enemy_dead_or_far"]), float(facts["safe_recall_window"])])
        return values

    def _accuracy_features(self):
        normal_count = len(self.normal_results)
        skill_count = len(self.skill_results)
        values = []
        values.extend(one_hot(min(normal_count, 3), 4))
        values.extend(one_hot(bucket(self.normal_attack_hit_rate, [0.2, 0.4, 0.6, 0.8]), 5))
        values.extend(one_hot(min(skill_count, 3), 4))
        values.extend(one_hot(bucket(self.skill_hit_rate, [0.2, 0.4, 0.6, 0.8]), 5))
        values.extend(one_hot(min(self.empty_attack_streak, 3), 4))
        return values

    def _ambush_features(self, facts):
        if facts["self_in_grass"]:
            self.grass_wait_frames += 1
        else:
            if self.grass_wait_frames > 0:
                self.ambush_exit_frames = 12
            self.grass_wait_frames = 0
        if self.ambush_exit_frames > 0:
            self.ambush_exit_frames -= 1
        wait_bucket = bucket(self.grass_wait_frames, [15, 45, 90])
        ambush_ready = facts["self_in_grass"] and facts["enemy_near_grass"]
        values = [float(facts["self_in_grass"]), float(facts["enemy_near_grass"])]
        values.extend(one_hot(wait_bucket, 4))
        values.extend([float(ambush_ready), float(self.ambush_exit_frames > 0)])
        return values

    def _luban_ult_features(self, facts):
        distance = facts["enemy_luban_ult_distance"]
        distance_ratio = 0.0
        if facts["enemy_luban_ult_active"]:
            distance_ratio = min(distance / GameConfig.LUBAN_ULT_WARNING_RADIUS, 1.0)
        values = [
            float(facts["enemy_luban_ult_active"]),
            float(facts["enemy_luban_ult_inside"]),
            float(facts["enemy_luban_ult_near"]),
            float(facts["enemy_luban_ult_moving_away"]),
        ]
        values.extend(one_hot(compact_distance_bucket(distance), 4))
        values.extend([
            distance_ratio,
            float(facts["enemy_luban_ult_inside"] and facts["self_low_hp"]),
        ])
        return values
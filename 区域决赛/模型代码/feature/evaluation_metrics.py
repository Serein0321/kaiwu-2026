class EvaluationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.attack_attempts = 0
        self.attack_hits = 0
        self.empty_attacks = 0
        self.skill_attempts = 0
        self.skill_hits = 0
        self.frenzy_windows = 0
        self.frenzy_uses = 0
        self.safe_recall = 0
        self.bad_recall = 0
        self.tower_push_windows = 0
        self.tower_push_uses = 0
        self.grass_ambush_windows = 0
        self.grass_ambush_success = 0
        self.combo_luban_starts = 0
        self.combo_luban_finishes = 0
        self.combo_direnjie_starts = 0
        self.combo_direnjie_finishes = 0

    def observe_reward(self, reward):
        frenzy_trade = reward.get("frenzy_trade_origin", 0)
        if frenzy_trade > 0:
            self.frenzy_windows += 1
            self.frenzy_uses += 1
        elif frenzy_trade < 0:
            self.frenzy_windows += 1

        attack_accuracy = reward.get("attack_accuracy_origin", 0)
        if attack_accuracy != 0:
            self.attack_attempts += 1
            if attack_accuracy > 0:
                self.attack_hits += 1
            elif attack_accuracy < 0:
                self.empty_attacks += 1

        skill_urgency = reward.get("skill_urgency_origin", 0)
        if skill_urgency != 0:
            self.skill_attempts += 1
            if skill_urgency > 0:
                self.skill_hits += 1

        safe_recall = reward.get("safe_recall_origin", 0)
        if safe_recall > 0:
            self.safe_recall += 1
        elif safe_recall < 0:
            self.bad_recall += 1

        tower_push = reward.get("tower_push_origin", 0)
        if tower_push != 0:
            self.tower_push_windows += 1
            if tower_push > 0:
                self.tower_push_uses += 1

        grass_ambush = reward.get("grass_ambush_origin", 0)
        if grass_ambush != 0:
            self.grass_ambush_windows += 1
            if grass_ambush > 0:
                self.grass_ambush_success += 1

        combo_luban = reward.get("combo_luban_origin", 0)
        if combo_luban != 0:
            self.combo_luban_starts += 1
            if combo_luban > 0:
                self.combo_luban_finishes += 1

        combo_direnjie = reward.get("combo_direnjie_origin", 0)
        if combo_direnjie != 0:
            self.combo_direnjie_starts += 1
            if combo_direnjie > 0:
                self.combo_direnjie_finishes += 1

    def ratio(self, numerator, denominator):
        return numerator / denominator if denominator else 0.0

    def as_monitor_data(self):
        return {
            "empty_attack_rate": round(self.ratio(self.empty_attacks, self.attack_attempts), 3),
            "normal_attack_hit_rate": round(self.ratio(self.attack_hits, self.attack_attempts), 3),
            "skill_hit_rate": round(self.ratio(self.skill_hits, self.skill_attempts), 3),
            "frenzy_trade_rate": round(self.ratio(self.frenzy_uses, self.frenzy_windows), 3),
            "safe_recall_rate": round(self.ratio(self.safe_recall, self.safe_recall + self.bad_recall), 3),
            "bad_recall_count": self.bad_recall,
            "tower_push_window_use_rate": round(self.ratio(self.tower_push_uses, self.tower_push_windows), 3),
            "grass_ambush_success_rate": round(self.ratio(self.grass_ambush_success, self.grass_ambush_windows), 3),
            "combo_completion_rate_luban": round(self.ratio(self.combo_luban_finishes, self.combo_luban_starts), 3),
            "combo_completion_rate_direnjie": round(self.ratio(self.combo_direnjie_finishes, self.combo_direnjie_starts), 3),
        }
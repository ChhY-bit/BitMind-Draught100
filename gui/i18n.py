from __future__ import annotations

# ---------------------------------------------------------------------------
# Translation table — keys are semantic identifiers, values are
# {"zh": ..., "en": ...}.  All user-visible strings live here.
# ---------------------------------------------------------------------------

_TRANSLATIONS = {
    # ---- Window / frame titles ----
    "window_title":         {"zh": "BitMind 国际跳棋",       "en": "BitMind Draughts"},
    "notation_frame":       {"zh": "棋谱",                    "en": "Notation"},
    "engine_frame":         {"zh": "引擎分析",                "en": "Engine Analysis"},
    "status_frame":         {"zh": "对局状态",                "en": "Game Status"},

    # ---- Status row labels ----
    "status_mode":          {"zh": "模式",                    "en": "Mode"},
    "status_side":          {"zh": "行棋方",                  "en": "Side"},
    "status_moves":         {"zh": "合法动作",                "en": "Legal Moves"},
    "status_result":        {"zh": "局面结果",                "en": "Result"},
    "status_last":          {"zh": "最近动作",                "en": "Last Move"},

    # ---- Buttons ----
    "btn_new_game":         {"zh": "新建对局",                "en": "New Game"},
    "btn_restart":          {"zh": "重开本局",                "en": "Restart"},
    "btn_undo":             {"zh": "悔棋",                    "en": "Undo"},
    "btn_rotate":           {"zh": "旋转棋盘",                "en": "Rotate"},
    "btn_pause":            {"zh": "暂停AI",                  "en": "Pause AI"},
    "btn_resume":           {"zh": "继续AI",                  "en": "Resume AI"},
    "btn_ai_step":          {"zh": "AI走一步",                "en": "AI Step"},

    # ---- Language toggle ----
    "lang_toggle":          {"zh": "English",                 "en": "中文"},

    # ---- Treeview headings ----
    "tree_action":          {"zh": "编号",                    "en": "ID"},
    "tree_prob":            {"zh": "概率",                    "en": "Prob"},
    "tree_qval":            {"zh": "Q值",                     "en": "Q"},
    "tree_visits":          {"zh": "访问",                    "en": "N"},
    "tree_path":            {"zh": "路径",                    "en": "Path"},
    "tree_captures":        {"zh": "吃子",                    "en": "Captures"},

    # ---- Section labels ----
    "section_moves":        {"zh": "合法动作",                "en": "Legal Moves"},
    "section_log":          {"zh": "日志",                    "en": "Log"},

    # ---- Analysis panel ----
    "analysis_value":       {"zh": "估值",                    "en": "Value"},
    "analysis_search":      {"zh": "搜索量",                  "en": "Search"},
    "analysis_best":        {"zh": "首选",                    "en": "Best"},
    "analysis_eval":        {"zh": "{v:+.3f}  →  {side}方胜率 {pct:.0f}%",
                             "en": "{v:+.3f} → {side} win {pct:.0f}%"},
    "analysis_sims":        {"zh": "{sims} 次模拟  (c_puct={c_puct})",
                             "en": "{sims} sims (c_puct={c_puct})"},
    "analysis_depth":       {"zh": "depth={depth}",           "en": "depth={depth}"},
    "analysis_best_fmt":    {"zh": "#{aid} ({pct:.1f}%)",     "en": "#{aid} ({pct:.1f}%)"},
    "analysis_delta":       {"zh": "搜索修正",                "en": "Search Δ"},
    "analysis_delta_fmt":   {"zh": "网络{netv:+.2f}→搜索{searchv:+.2f}",
                             "en": "Net{netv:+.2f}→Search{searchv:+.2f}"},
    "analysis_na":          {"zh": "—",                       "en": "—"},

    # ---- Side labels ----
    "side_black":           {"zh": "黑方",                    "en": "Black"},
    "side_white":           {"zh": "白方",                    "en": "White"},
    "side_black_short":     {"zh": "黑",                      "en": "B"},
    "side_white_short":     {"zh": "白",                      "en": "W"},

    # ---- Result labels ----
    "result_ongoing":       {"zh": "进行中",                  "en": "Ongoing"},
    "result_black_win":     {"zh": "黑方胜",                  "en": "Black Wins"},
    "result_white_win":     {"zh": "白方胜",                  "en": "White Wins"},
    "result_draw":          {"zh": "和棋",                    "en": "Draw"},

    # ---- Player kind labels ----
    "kind_human":           {"zh": "人类",                    "en": "Human"},
    "kind_random":          {"zh": "随机AI",                  "en": "Random AI"},
    "kind_alphabeta":       {"zh": "AlphaBeta AI",            "en": "AlphaBeta AI"},
    "kind_supervised":      {"zh": "监督模型AI",              "en": "Supervised AI"},
    "kind_mcts":            {"zh": "MCTS AI",                 "en": "MCTS AI"},

    # ---- Dialog ----
    "dialog_title":         {"zh": "新建对局",                "en": "New Game"},
    "dialog_prompt":        {"zh": "选择双方决策源",          "en": "Select Players"},
    "dialog_black":         {"zh": "黑方",                    "en": "Black"},
    "dialog_white":         {"zh": "白方",                    "en": "White"},
    "dialog_start":         {"zh": "开始对局",                "en": "Start Game"},
    "dialog_create_fail":   {"zh": "创建对局失败",            "en": "Create Game Failed"},
    "dialog_no_checkpoint": {"zh": "未选择模型",              "en": "No Model Selected"},
    "dialog_need_checkpoint": {"zh": "请选择 checkpoint 文件后再开始对局。",
                               "en": "Please select a checkpoint file."},
    "dialog_checkpoint":    {"zh": "请选择{side} {kind} checkpoint",
                             "en": "Select {side} {kind} checkpoint"},
    "dialog_mcts_config":   {"zh": "请选择{side} MCTS 配置文件（取消则使用默认）",
                             "en": "Select {side} MCTS config (cancel for default)"},
    "dialog_ft_pytorch":    {"zh": "PyTorch checkpoint",      "en": "PyTorch checkpoint"},
    "dialog_ft_all":        {"zh": "所有文件",                "en": "All files"},
    "dialog_ft_yaml":       {"zh": "YAML config",             "en": "YAML config"},
    "dialog_fp16":          {"zh": "启用 FP16 混合精度推理（GPU 加速约 1.5-2×）",
                             "en": "Enable FP16 mixed precision (~1.5-2× GPU speedup)"},
    "dialog_compile":       {"zh": "启用 torch.compile 算子熔合（首次有编译延迟）",
                             "en": "Enable torch.compile (first-run compilation delay)"},

    # ---- Player name templates ----
    "player_name":          {"zh": "{side}{kind}",             "en": "{side} {kind}"},
    "player_name_human":    {"zh": "{side}玩家",              "en": "{side} (Human)"},

    # ---- Mode label template ----
    "mode_label":           {"zh": "黑方{kind_black} vs 白方{kind_white}",
                             "en": "Black ({kind_black}) vs White ({kind_white})"},

    # ---- Log messages ----
    "log_new_game":         {"zh": "新对局开始：{mode}。",
                             "en": "New game: {mode}."},
    "log_ai_paused":        {"zh": "AI 自动走棋已暂停。",
                             "en": "AI auto-play paused."},
    "log_ai_resumed":       {"zh": "AI 自动走棋已继续。",
                             "en": "AI auto-play resumed."},
    "log_game_ended_ai":    {"zh": "对局已经结束，无法继续执行 AI。",
                             "en": "Game already ended."},
    "log_human_no_ai_step": {"zh": "当前行棋方是人类，不能执行 AI 走一步。",
                             "en": "Current player is human, cannot AI-step."},
    "log_game_over_btn":    {"zh": "对局已经结束。请点击\"重新开始\"。",
                             "en": "Game over. Click \"Restart\"."},
    "log_ai_controlled":    {"zh": "当前行棋方由 AI 控制。",
                             "en": "Current turn is AI-controlled."},
    "log_selection_cancel": {"zh": "已取消选择。",
                             "en": "Selection cancelled."},
    "log_ambiguous":        {"zh": "有 {count} 条路线都落在 {sq} 号位，请在右侧动作列表中双击选择完整路线。",
                             "en": "{count} routes land on #{sq}, double-click in move list."},
    "log_not_landing":      {"zh": "该位置不是当前棋子的合法落点。",
                             "en": "Not a valid landing square."},
    "log_ai_ctrl_step":     {"zh": "当前行棋方由 AI 控制，请使用\"AI走一步\"或自动走棋。",
                             "en": "AI-controlled turn. Use \"AI Step\" or auto-play."},
    "log_action_stale":     {"zh": "动作 {aid} 已不再合法。",
                             "en": "Action {aid} no longer legal."},
    "log_exec_fail":        {"zh": "执行失败：{error}",
                             "en": "Execution failed: {error}"},
    "log_move":             {"zh": "第 {turn} 手，{actor} 执行动作 {aid}：{move}",
                             "en": "Move {turn}: {actor} action {aid}: {move}"},
    "log_game_result":      {"zh": "对局结束：{result}。",
                             "en": "Game over: {result}."},
    "log_no_undo":          {"zh": "当前没有可悔的动作。",
                             "en": "No moves to undo."},
    "log_undo_fail":        {"zh": "悔棋失败：{error}",
                             "en": "Undo failed: {error}"},
    "log_undone":           {"zh": "已悔棋：{notation}",
                             "en": "Undid: {notation}"},
    "log_no_moves":         {"zh": "{name} 没有可执行动作。",
                             "en": "{name} has no legal moves."},
    "log_model_loaded":     {"zh": "{name} 已加载模型：{file}",
                             "en": "{name} loaded model: {file}"},
    "log_model_loaded_mcts":{"zh": "{name} 已加载模型：{file}（MCTS 配置：{cfg}）",
                             "en": "{name} loaded model: {file} (MCTS config: {cfg})"},
    "log_board_rotated":    {"zh": "棋盘视角已旋转 180 度。",
                             "en": "Board rotated 180°."},
    "log_no_legal_moves":   {"zh": "该棋子当前没有合法动作。",
                             "en": "No legal moves for this piece."},
    "log_square_selected":  {"zh": "已选择 {sq} 号位，共 {count} 条合法路线。",
                             "en": "Selected #{sq}, {count} legal routes."},
    "log_need_checkpoint":  {"zh": "{side}监督模型AI需要选择 checkpoint 文件",
                             "en": "{side} Supervised AI needs a checkpoint file"},
    "log_need_mcts_ckpt":   {"zh": "{side}MCTS AI需要选择 checkpoint 文件",
                             "en": "{side} MCTS AI needs a checkpoint file"},
    # -- value curve --
    "curve_frame":          {"zh": "估值曲线",                "en": "Value Curve"},
    "curve_empty":          {"zh": "(无 AI 参与，估值曲线不可用)",
                             "en": "(No AI in game — curve unavailable)"},
    "curve_black_pct":      {"zh": "黑方胜率%",               "en": "Black Win%"},
    "curve_black_wins":     {"zh": "黑胜",                    "en": "Black"},
    "curve_white_wins":     {"zh": "白胜",                    "en": "White"},
    "curve_ply":            {"zh": "步",                       "en": "Ply"},
    "log_rewind":           {"zh": "已回溯至第 {ply} 步",       "en": "Rewound to ply {ply}"},
    "log_rewind_paused":    {"zh": "已暂停对弈并回溯至第 {ply} 步",
                             "en": "Paused game and rewound to ply {ply}"},
    # -- hardware monitor --
    "mon_frame":            {"zh": "系统监控",                "en": "System Monitor"},
    "mon_cpu":              {"zh": "CPU 总利用率",             "en": "CPU Usage"},
    "mon_ram":              {"zh": "系统内存",                "en": "Sys RAM"},
    "mon_proc":             {"zh": "进程 CPU",               "en": "Proc CPU"},
    "mon_gpu":              {"zh": "GPU 利用率",              "en": "GPU Usage"},
    "mon_vram":             {"zh": "显存占用",                "en": "VRAM"},
    "mon_temp":             {"zh": "GPU 温度",               "en": "GPU Temp"},
    "mon_no_gpu":           {"zh": "无 GPU",                 "en": "No GPU"},
}


def _tr_static(key: str, lang: str) -> str:
    """Look up a translation key for the given language.

    Falls back to ``zh`` if the language or key is missing.
    """
    entry = _TRANSLATIONS.get(key)
    if entry is None:
        return key  # safe fallback — shows the key name instead of crashing
    return entry.get(lang, entry.get("zh", key))

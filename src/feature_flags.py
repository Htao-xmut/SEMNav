"""特性开关系统 — 消融实验的统一配置面 (论文表 III/IV 变体)

用法:
    cfg 注入:  cfg['feature_flags'] = {'metric_gate': False}
    CLI:       python bm_run.py 17 <cat> 1 --no-metric-gate
               (bm_run.py / batch runner 调 parse_argv 剥离开关参数)
    代码读:    self._ff = feature_flags.resolve(cfg)
               if self._ff['metric_gate']: ... (每个机制点一行判断)

档位组合 (表 III 五档, describe() 生成记名):
    base        = cascade:F, grid_hub:F, metric_gate:F, warmup_preempt:F
    +C          = cascade:T (其余 F)
    +C+G        = + grid_hub:T
    +C+G+S      = + metric_gate:T
    +C+G+S+W    = + warmup_preempt:T (全量默认)
表 IV 替代实现: --direct-vlm / --maxconf-fusion / --single-thresh
对照行:        --hard-depth-filter;  上限: --gt-detect (SEM-Nav-Ideal)

⚠ 接入状态 (阶段2 逐步填肉, 未接入的开关只记日志不生效):
    cascade         ✓ agent 级联路由
    metric_gate     ✓ env 停票仲裁/守卫/PROXIMITY 门
    warmup_preempt  ✓ env warmup 抢占
    single_thresh   ✓ env 判据覆盖 (收紧前单阈值复现, _apply_flag_overrides;
                    eval 侧 success_threshold 由 bm_run/batch_run 配 1.5)
    grid_hub        ✓ 深度推荐注入侧; ✗ 图前沿/可走边深层过滤 (TODO)
    gt_detect       ✓ wrapper ideal_detect 检测注入 (step/_warmup_scan 两路,
                    env 侧 episode 开始时注入 gt_goal_pos; SEM-Nav-Ideal 上限行)
    hard_depth_filter ✓ env _hard_filter_depth_rec (推荐锥外平移候选剔除;
                    表 III "推荐制 vs 硬剔除" 对照行)
    direct_vlm       ✓ env _vlm_grounding_detect + _apply_detector_variants
                    (每步 VLM grounding 替换 YOLO, 扫描同口径; 表 IV 行)
    maxconf_fusion   ✓ env _apply_detector_variants (YOLO 检出时 VLM 置信
                    max 融合路由; 表 IV 行, VLFM 式)
"""

DEFAULTS = {
    'cascade': True,             # C: 置信分档 YOLO–VLM 前置级联
    'grid_hub': True,            # G: 持久栅格中枢 (深度推荐+前沿引导)
    'metric_gate': True,         # S: 米制停票门 + fp 接力
    'warmup_preempt': True,      # W: warmup 反应式抢占
    'hard_depth_filter': False,  # 深度推荐软偏置→硬剔除 (对照行)
    'direct_vlm': False,         # 每步 VLM 判 present (表 IV)
    'maxconf_fusion': False,     # max-conf 融合 (VLFM 式, 表 IV)
    'single_thresh': False,      # 单阈 0.60 过滤 (表 IV)
    'gt_detect': False,          # GT 检测注入 (SEM-Nav-Ideal)
    # --- 阶段0 三修复 (设计稿 阶段0_FSM设计稿.md §7, 默认全开) ---
    'identity_gate_v2': True,    # 修复1: PROXIMITY wrong 三级化 (L1冷却/L2强证据/L3耗尽)
    'vote_fastpath': True,       # 修复2: 停票链保护 (L1 压制CONFIRM + L2 桥接快通道 2票→1票)
    'vote_fastpath_min_conf': 0.30,  # L2 质量门 (OR 语义: conf≥此值 或 area≥下值;
    'vote_fastpath_min_area': 0.01,  #  AND 会误杀 ep2 — 实测 area=0.00/conf=0.36)
    'neutral_rooms': True,       # 修复3: 房间标签中和 (zone_i_j 替代 GT 阈值房间名)
    # --- 阶段0.1 问题驱动修复 (三局实弹诊断: ep4 转圈 / ep1 方位错认) ---
    'hr_180_cooldown': True,     # 修2A: 硬规则 1a/1d 的 180° 掉头冷却 (ep4:
                                 #   VLM 扫视旋转 + 1d 窗口惯性 → 50 步掉头
                                 #   5 次, 1b/1d 交替对抗 = 区域内转圈)
    'hr_180_cooldown_steps': 12, # 掉头一次后冷却步数 (7 步窗滑出 + 新方向尝试)
    'zero_detect_crop': True,    # 修2B: 零检出感知兜底 (ep4 softsoap 全程零
                                 #   YOLO 检出 → 无线索纯扫视; ≥12 步未见目标
                                 #   每 6 步 crop 放大看一次前方近处)
    'first_dir_gate': True,      # 修3A: FIRST_DIRECTION 证据门 (ep1: 六图联合
                                 #   分析跨图串扰 — 120° 照片的可乐瓶被归属到
                                 #   240°; 单图重问一次, 无多图即无归属混淆)
    # --- 问题1 距离判断加强 (ep6 诊断: 框内深度网格会被玻璃洞/遮挡污染) ---
    'support_depth': True,       # 问题1-A: 支撑面深度交叉 — bbox 底边下方
                                 #   台面采样 (实心连续表面), 与框内网格强分歧
                                 #   时保守取远 (两个方向的污染都不产生新 fp)
    'parallax_gate': True,       # 问题1-B: 运动视差测距门 — 连续两帧框面积比
                                 #   + 已知位移 (图节点=自身里程计) 针孔模型反解
                                 #   距离; 独立于深度传感器, 只降不升 (close→far)
    # --- 阶段1 转圈根治 (ep6 后段 40 步游荡 / ep4 前段 16 步徘徊实弹) ---
    'frontier_hop': True,        # 区域看尽→前沿跳跃: 最近 10 步节点全困在
                                 #   <2.5m 小区域且无目标线索 → 探索地图灰格
                                 #   (可走未到访) 最近一个 → 已知空间 Dijkstra
                                 #   路径 (GOTO_WAYPOINT 通道) 强制离开旧区域
    'low_conf_reposition': True, # 低置信误检换机位重看 (ep1 灯下黑: goal 就在
                                 #   0.06m 脚边, 当前机位看不清被 IDENTITY 判
                                 #   no 放走 → 走远 3.46m fp)。IDENTITY no 且
                                 #   低置信 → 先平移换站位再看, 新视角自然重检
    # --- 批次1 四修复 (第三轮三局实弹: ep6 fp@2.42m / ep1 51步绕行) ---
    'dist_xcheck': True,         # P1: 停票 close 距离交叉核 — crop/fastpath
                                 #   close 判定须过独立几何源 (视差 > 支撑面
                                 #   > 前向中央带深度), 任一源 >1.5m → 否决;
                                 #   全部无证据 → 维持 VLM 判定 (防死锁)。
                                 #   ep6 step37: 框内网格 0.61m 实际 2.42m
    'mustdo_gate': True,         # P2 (修6): MUST-DO 行条件化 — 方位被证据门
                                 #   拒 (INCONCLUSIVE) 时不发 "MUST-DO NOW",
                                 #   改推测注记 (ep6: 门拒后 MUST-DO 残留,
                                 #   VLM 拿猜测当令箭往 60° 厨房推测走)
    'hr_servo_bypass': True,     # P3 (修7a): 硬规则 1b/1c 伺服旁路 —
                                 #   APPROACH miss=0 (刚锁定/持续目击) 时
                                 #   force forward 不劫持伺服转向 (ep1 step26:
                                 #   state=APPROACH 中 ESCAPE 1b → 6 步跟丢,
                                 #   绕 26 步)。miss>0 后恢复 (防圈保护仍在)
    'zoom_confirm': True,        # P5 (修7c'): 确认问询 bbox 裁剪放大 —
                                 #   PROXIMITY/IDENTITY 对远距小目标
                                 #   (area<0.01) 用全幅图看不清 (ep1 step25:
                                 #   conf0.98 被判 wrong; ep6 step37: conf0.40
                                 #   unsure → fastpath 误停)。以 bbox 为中心
                                 #   裁剪放大再问
    # --- 批次2 两修复 (用户思路: 区域扫尽语义问询 + 释锁重定向) ---
    'release_rescan': True,      # P4 (修8): 释锁即六向重扫 — 锁定释放
                                 #   (误报纠正/跟丢/两击升级) 后下一步决策
                                 #   前先 360° 重扫重新定向 (预算 ≤2/回合,
                                 #   保已访记忆)。释锁 = 旧方位信念作废,
                                 #   沿旧记忆盲走 = ep1 释锁游荡 26 步主因
    'scan_inquiry': True,        # P6 (修9): 扫描语义问询层 — 六图扫描
                                 #   prompt 追加覆盖检查 (SCAN_SUSPICIOUS:
                                 #   <angle>° / SCAN_CLEAR), 可疑 → 无
                                 #   YOLO/FIRST_DIRECTION 方位时转身查看,
                                 #   CLEAR → 去新区; 判"区域看尽"须 ≥2 个
                                 #   扫描机位 (灯下黑: 单机位不判尽)
    'reject_sight_lock': True,   # #25: 拒停不失忆 — 停票被拒且当前无框
                                 #   时, 用最近目击记录 (≤8 步内, 非所在
                                 #   节点) 的方位建接近锁走回去; 站在目击点
                                 #   上/无新鲜目击 → 置 P4 释锁重扫兜底。
                                 #   实弹: ep6 p16 step27 拒停后锁全丢,
                                 #   1.17m 处盲走 12 步才被周期重扫救回
    'viewpoint_rescan': True,    # #26: 换机位补扫 — 判尽门 (≥2 机位)
                                 #   挡下时主动平移一格换站位 → 下一步
                                 #   360° 重扫登记新机位 (预算 ≤2/回合,
                                 #   独立于 P4)。实弹: p16 cola 判尽门
                                 #   跨格 0 机位被堵 11 次, 门只堵不引,
                                 #   干等周期重扫到回合耗尽
    # --- 扫描机制 v2 (2026-09-13, 用户指令: 困惑即扫 + 无价值区第一时间离区) ---
    'zone_clear_exit': True,     # SCAN_CLEAR 判定的 zone 登记入
                                 #   _zone_clear_cells → 处于已判尽 zone
                                 #   时最高优先强制前沿跳跃离区 (YOLO 实时
                                 #   目击撤销登记)。实弹: mahatma-ab 6 次
                                 #   扫描全 CLEAR 仍原地打转到 49 步
    'zone_clear_sight_guard': True,  # #31: 有目击史的 zone 不登记 CLEAR —
                                 #   六图判词"没看出来"打不过自身 YOLO 目击
                                 #   记录。实弹: coca-ab 出生 6cm, warmup
                                 #   YOLO 61% + 接近期目击全在厨区, step11
                                 #   误 CLEAR → v2 三次离区拖到 1.98m fp
    'semantic_frontier': True,  # #32: 前沿跳跃语义偏好 — 扫描 LIKELY-
                                 #   BEARING (VLM 判断哪个方向通向目标
                                 #   常在区域) ±90° 内优先取前沿; 已清区
                                 #   落点出池。实弹 bmv2 mahatma: "最近
                                 #   灰格"永远隔壁 zone, (-1,-2) 被 CLEAR
                                 #   两次, 51 步困同一功能区终距 10.46m
    'semantic_go': True,        # #35: 扫描定方向即承诺 — 困惑扫描的产出
                                 #   (LIKELY-BEARING) 不再只排序前沿落点,
                                 #   zone 无未查可疑点时立即规划 SEMANTIC-GO
                                 #   旅程走过去 (≥2 查过方位即视为查尽,
                                 #   不追第 3 个"新可疑")。用户指令: "scan
                                 #   以后就是要找出探索的思路, 而不是继续
                                 #   困惑"。实弹 bmv5: 7 次扫描 7 个可疑点
                                 #   7 次转身 0 次方向承诺, 每 zone 磨 ~10
                                 #   步, step27 才出客厅
    'scan_sighting': True,      # #37: 扫描目击登记 + 失而复得 — 六图扫描
                                 #   YOLO 命中按 wrapper 同构格式登记为目击
                                 #   (含 ③c 配对帧), 接近失检释锁后 ≤12 步
                                 #   内走回扫描站位重看一次 (每节点一次)。
                                 #   实弹 bmv6: 3 次 APPROACH-LOCK 全来自
                                 #   扫描命中 (conf 0.51/0.56/0.56), 释锁后
                                 #   证据蒸发 (wrapper 只登记活体帧, 3 步
                                 #   新鲜度也不够), 机器人从"确曾看见"的
                                 #   视角继续盲走
    'semantic_prior': True,     # #38: 语义先验 V2 (用户三选拍板) — ① warmup
                                 #   即承诺: 开局无目击无可疑 → 语义方位直接
                                 #   成行, depth rec 降兜底 (bmv6: 厨房 57°
                                 #   没人听, step1-3 跟几何走沙发); ② 证据门
                                 #   分类: FIRST_DIRECTION 推测声明降级投票
                                 #   不再一杀 (bmv6: 240° 厨房被毙两次); ③
                                 #   投票聚合: 承诺 = 最近 ≤3 票圆均值, 与
                                 #   上次承诺相反 >120° 需两票一致 (bmv6:
                                 #   331°→128°→−15° 旅程互殴)
    'semantic_over_suspicious': True,  # #39 (用户拍板"语义优先, 可疑顺路查"):
                                 #   可疑点与语义先验同级都是推测 (照片里
                                 #   目标不可见), 4 局实弹 P6 追查 0 命中。
                                 #   语义方向新鲜时: 顺路可疑 (±60°) 照旧
                                 #   转身查看, 不顺路留 zone 记忆不追; 无语
                                 #   义方向 → 可疑照旧优先 (原行为)
    'journey_stride': True,     # #41: 旅程/VLM 多步平移 — 深度走廊通畅
                                 #   (exploration_map 走廊采样, 只有 OCCUPIED
                                 #   算阻挡, 零样本合规) 且连续同型图边存在
                                 #   → 提供链式选项 (≤3 跳/≤2.4m), 旅程同向
                                 #   K 边并步弹 K 条。实弹 bmv6: 用户"安全
                                 #   距离完全够的话，直行什么一点一点移?"
                                 #   (SEMANTIC-GO 旅程 0.5-0.9m 边逐步挪)
    'identity_lowconf_confirm': True,  # #42: 低置信 identity=no → 前进二次
                                 #   确认 (≤3 次/节点) 而非立刻拒绝; wrong
                                 #   判决不混入丢失两击 (loss_strike=False)。
                                 #   实弹 bmv7 step27: conf 0.41 真米袋被
                                 #   no → TWO-STRIKES 拉黑 → 扫描重检压制
                                 #   → 4.67m max_steps (用户: "YOLO都看到
                                 #   了，不过去确认下吗? 应该去前进二次确认
                                 #   下")。#43 配套: zoom 窗 2.5→1.0 倍边距
                                 #   (窗口 36×→9× bbox, 目标占窗 ~11%)
    'servo_geometry': True,    # #44: 伺服/对齐选边用落点位移几何 — 旧
                                 #   yaw+0°(forward) 朝向启发在走廊图上可偏
                                 #   ~50° (节点朝向 −37° 而 forward 边位移
                                 #   正西), bmv8 "aligned 1deg" 实偏致真目
                                 #   标出画释锁循环 → TWO-STRIKES 拉黑真目
                                 #   标区。三处接线: 精确伺服/锁定方位对齐/
                                 #   旅程方位等价边; 断链回落旧模型。#45 配
                                 #   套: 扫描目击同位置不同朝向 → 旋转补全
                                 #   到目击朝向再配对 (活体目击保持 ③c 跳过)
    'journey_stepcost': True,  # #46: 旅程 Dijkstra 步数等价代价 — 旧按
                                 #   边距离(米)计价, 而图里旋转边 distance
                                 #   均值 0.014m (128 条 0.0) = 旋转免费,
                                 #   救援重规划拿免费旋转洪泛路径 (bmv9
                                 #   step18: 17 边含 7×rotate_ccw ≈ 原地转
                                 #   330°, step19-33 烧 15 步近零位移)。
                                 #   旋转 1 步/边、平移 0.45 步/边 (#41 并步
                                 #   折算); 复算 v9 病例 17 边→14 边、est
                                 #   ~15 步→10 步, #45 同位转向补全不受影响
    'lock_bearing_fix': True,  # #47: 扫描目击锁方位双修 — ① wrapper 途中
                                 #   null 步 (rescan 取 obs) 纯读化: 原先
                                 #   无条件 _find_closest_node(位置)+A1 对齐
                                 #   = 同位置 12 朝向变体取 dict 首键任意
                                 #   朝向 → current_node 偷换 + sim 相机物理
                                 #   转向 (bmv10 step38b: −28.3°→153.2°,
                                 #   锁错 181.6°); ② 锁公式补画面内框偏移
                                 #   (目标框 cx 反解, 照片 HFOV≈89°) — 相机
                                 #   朝向 ≠ 目标方位 (bmv10 step42: middle_
                                 #   right +41°, 锁 304° 真 344.9°≡−15.1°
                                 #   ≈ GT −13.8°)。两次锁全错 → miss →
                                 #   TWO-STRIKES 拉黑真目标区 → 3.99m 终距
    'bridge_metric_guard': True,  # #49: 桥接配对 close 米制守卫 — 配对
                                 #   VLM (目击帧+当前帧) 能证 present 证不
                                 #   了 close (bm16 教训)。当前决策帧有目标
                                 #   框但 <2% 画面 = 无近证据 → 拒停转接近。
                                 #   bmv11 step28: 框 area 0.003 conf 0.86
                                 #   @2.42m 被配对放行 fp (fp 救济线 <1.0m
                                 #   不救)。配套: _re_scan 后 _last_obs 同
                                 #   步刷新 (#30a 补全) + _stop_evidence
                                 #   静默 None 全部改 INFO 可审计
    'small_box_quarantine': True,  # #50 正修 (d03r fp@5.84m): 小框
                                 #   (area<0.5%) bbox-grid 深度双向隔离 —
                                 #   三件套实弹: bmv12 九点饿死→None,
                                 #   d01r 0.42m 近探针→误 FAR, d03r
                                 #   0.53m 框外地板→误 CLOSE 放行 fp。
                                 #   小框 CLOSE 须独立源佐证 (视差/支撑
                                 #   面/前向带 ≤1.0m), 无 → FAR 拒停转
                                 #   接近 (#49b 同语义补 proximity 缺口);
                                 #   bmv12 成功停 area 0.020 走面积捷径
                                 #   不受影响
    'vlm_angle_parse': False,  # #52a (d01r step43-46): agent 裸抓 reasoning
                                 #   第一个角度折算 rotate_steps — 抓到的常是
                                 #   目标 bearing (step43: 选 30° 微调,
                                 #   reasoning 提 "+177°" → 6 连转 180° 甩头,
                                 #   与 APPROACH 覆盖互甩 4 步零位移)。
                                 #   prompt 无此协议 (forward 的 "say
                                 #   forward Xm" 才是明文协议) → 默认停用,
                                 #   角度 = 所选图边 chain_count; 开 = 旧行为
    'approach_bearing_path': True,  # #53a (d01r2 step25-31): APPROACH 锁
                                 #   方位与当前朝向差 >45° 时, 对齐选边一步
                                 #   只能挪 30° 且每步涨 miss → 预算耗尽释锁
                                 #   → #37 拉回重锁同方位无限循环 (锁 −124°
                                 #   vs 朝向 −3°, 3 步耗尽 ×2 轮)。改为沿
                                 #   锁方位图路径走 (_plan_bearing_path tag=
                                 #   APPROACH-TURN, 带平移的 GOTO 通道,
                                 #   旅程执行期接近让路、世界系锁不丢);
                                 #   活体重瞄 ≤44.7° 天然不触发, 只救陈旧
                                 #   扫描锁; 规划失败回落旧行为。#53b 配套
                                 #   (无 flag): 扫描锁写 _approach_last_area
                                 #   → 远距锁 (area<0.10) miss 预算 3→6
    'zone_revisit_gate': True,  # #51 (bmd01/d03 实弹, 用户指令"走过的区域
                                 #   其他区确认完才可再去"): 区域记忆+回访门。
                                 #   #51a 远距小框 identity-no 不可信 → 前进
                                 #   二次确认 (unarmed 路径补接 #42); #51b
                                 #   zone 否决记账 (≥2 击 → 否决区) + 每步
                                 #   prompt 注入"已否决勿回"警示; #51c 前沿
                                 #   分层: 全新 zone > 扫过未否决 > 否决区
                                 #   (仅全部其他选项耗尽才兜底)。d01 coca:
                                 #   真瓶 4 次被 no 放走 → 目击区不 CLEAR →
                                 #   记忆拽回再看再否 50 步; d03: 前沿池只
                                 #   认站立点, 最近灰格永远在走过的功能区
}

FLAG_ARGS = {
    '--no-cascade': ('cascade', False),
    '--no-grid-hub': ('grid_hub', False),
    '--no-metric-gate': ('metric_gate', False),
    '--no-warmup-preempt': ('warmup_preempt', False),
    '--hard-depth-filter': ('hard_depth_filter', True),
    '--direct-vlm': ('direct_vlm', True),
    '--maxconf-fusion': ('maxconf_fusion', True),
    '--single-thresh': ('single_thresh', True),
    '--gt-detect': ('gt_detect', True),
    '--no-identity-gate-v2': ('identity_gate_v2', False),
    '--no-vote-fastpath': ('vote_fastpath', False),
    '--no-neutral-rooms': ('neutral_rooms', False),
    '--no-hr-180-cooldown': ('hr_180_cooldown', False),
    '--no-zero-detect-crop': ('zero_detect_crop', False),
    '--no-first-dir-gate': ('first_dir_gate', False),
    '--no-support-depth': ('support_depth', False),
    '--no-parallax-gate': ('parallax_gate', False),
    '--no-frontier-hop': ('frontier_hop', False),
    '--no-low-conf-reposition': ('low_conf_reposition', False),
    '--no-dist-xcheck': ('dist_xcheck', False),
    '--no-mustdo-gate': ('mustdo_gate', False),
    '--no-hr-servo-bypass': ('hr_servo_bypass', False),
    '--no-zoom-confirm': ('zoom_confirm', False),
    '--no-release-rescan': ('release_rescan', False),
    '--no-scan-inquiry': ('scan_inquiry', False),
    '--no-reject-sight-lock': ('reject_sight_lock', False),
    '--no-viewpoint-rescan': ('viewpoint_rescan', False),
    '--no-zone-clear-exit': ('zone_clear_exit', False),
    '--no-zone-clear-sight-guard': ('zone_clear_sight_guard', False),
    '--no-semantic-frontier': ('semantic_frontier', False),
    '--no-semantic-go': ('semantic_go', False),
    '--no-scan-sighting': ('scan_sighting', False),
    '--no-semantic-prior': ('semantic_prior', False),
    '--no-semantic-over-suspicious': ('semantic_over_suspicious', False),
    '--no-journey-stride': ('journey_stride', False),
    '--no-identity-lowconf-confirm': ('identity_lowconf_confirm', False),
    '--no-servo-geometry': ('servo_geometry', False),
    '--no-journey-stepcost': ('journey_stepcost', False),
    '--no-lock-bearing-fix': ('lock_bearing_fix', False),
    '--no-bridge-metric-guard': ('bridge_metric_guard', False),
    '--no-small-box-quarantine': ('small_box_quarantine', False),
    '--vlm-angle-parse': ('vlm_angle_parse', True),
    '--no-approach-bearing-path': ('approach_bearing_path', False),
    '--no-zone-revisit-gate': ('zone_revisit_gate', False),
}


def parse_argv(argv):
    """CLI 参数剥离开关: 返回 (flags, 位置参数 rest)"""
    flags, rest = {}, []
    for a in argv:
        if a in FLAG_ARGS:
            k, v = FLAG_ARGS[a]
            flags[k] = v
        else:
            rest.append(a)
    return flags, rest


def resolve(cfg=None, overrides=None):
    """合成生效开关: DEFAULTS ← cfg['feature_flags'] ← overrides"""
    ff = dict(DEFAULTS)
    if cfg:
        src = cfg.get('feature_flags') or {}
        ff.update({k: v for k, v in src.items() if k in DEFAULTS})
    if overrides:
        ff.update({k: v for k, v in overrides.items() if k in DEFAULTS})
    return ff


def describe(flags):
    """档位记名 (日志/CSV): 对齐表 III 行名; 非标准组合 → 'custom'。

    对照开关 (表 III 硬剔除行 / 表 IV 替代实现行 / GT 上限行) 以
    后缀追加在基础档名上 (+HD/+ST/+DVL/+MC/+GT), 保持 CSV 行可辨识。
    """
    c, g = flags['cascade'], flags['grid_hub']
    s, w = flags['metric_gate'], flags['warmup_preempt']
    if w and s and g and c:
        base = '+C+G+S+W'
    elif s and g and c:
        base = '+C+G+S'
    elif g and c and not s and not w:
        base = '+C+G'
    elif c and not g and not s and not w:
        base = '+C'
    elif not c and not g and not s and not w:
        base = 'base'
    else:
        base = 'custom'
    suffix = ''.join(tag for k, tag in (
        ('hard_depth_filter', '+HD'),
        ('single_thresh', '+ST'),
        ('direct_vlm', '+DVL'),
        ('maxconf_fusion', '+MC'),
        ('gt_detect', '+GT'),
    ) if flags.get(k))
    return f'{base}{suffix}'

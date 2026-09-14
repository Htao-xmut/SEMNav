"""#52 执行一致性双修 单测 (2026-09-13, d01r coca 实弹)

用户报障 (d01r step43-46):
  "step43 和 step45 这两步都执行同一个操作？？？浪费！！"
  "step43 其实就在眼前了，从视觉上看，可以停止了啊。为什么不停止？"

日志取证 (logs/verify_d01r_20260913.log 20:13-20:15):
  step43 VLM 选 [2] turn RIGHT 30° 微调, reasoning 提目击 "+177°"
    → agent 裸正则抓 177 → "VLM wants ~177° → 6 edges" → rotate_steps=6
    → APPROACH 锁覆盖改选 [1] (aligned 22°, 本意 30° 微调)
    → wrapper:1403 用 action.rotate_steps=6 覆盖 chain_count=1
    → EXEC: idx=1 -> rotate_ccw x6 = 180° 甩头
  step44 锁在世界 -11° (step40 拒停 re-aim), 甩到 180° 后锁在背后
    → [0] TURN 180 转回 (aligned 18°) → 43/45 ccw×6 ↔ 44/46 cw×6
    互甩 4 步零位移; VLM 三次叫 stop 全被打断 (consecutive 永远 1/2)

病灶:
  #52a avdb_agent._choose_action 对旋转选项裸抓 reasoning 第一个角度
      数字 — 抓到的常是目标 bearing 不是转动意图; prompt 从未教 VLM
      "在 reasoning 里说转角" 协议 (forward 的 "say forward Xm" 是
      明文协议) → 折算属 agent 越权解释
  #52b _override_and_run/_attach_exec_idx 覆盖 idx 时不清
      rotate_steps/forward_steps → 覆盖动作被 VLM raw 折算残留污染
      (#27 复合残留同族: 解析态与执行态不一致)

修: #52a vlm_angle_parse 默认 False (旋转角度由所选图边 chain_count
    表达); --vlm-angle-parse 消融保留旧行为
    #52b _attach_exec_idx 带 obs 时同步 steps 到被选选项 chain_count

运行: python -u test_exec_consistency.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []

ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
AGT_SRC = open('/home/tao_h/VLMnav/src/avdb_agent.py').read()


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ========== 开关注册 (2 项) ==========
def test_flags():
    import feature_flags as F
    check('#52a: DEFAULTS vlm_angle_parse=False (折算默认停用)',
          F.DEFAULTS.get('vlm_angle_parse') is False)
    flags, rest = F.parse_argv(['--vlm-angle-parse', 'x'])
    check('#52a: --vlm-angle-parse 消融注册',
          F.FLAG_ARGS.get('--vlm-angle-parse') == ('vlm_angle_parse', True)
          and flags == {'vlm_angle_parse': True} and rest == ['x'])


# ========== #52a: agent 角度折算门 (3 项) ==========
def test_angle_parse_gate():
    from avdb_agent import AVDBAgent
    a = object.__new__(AVDBAgent)
    opt = {'chain_type': 'rotate_cw', 'chain_count': 1,
           'direction': 'rotate_right'}
    # d01r step43 病例: reasoning 提目击 bearing +177° (非转动意图)
    reasoning = ("The target 'coca_cola_glass_bottle' was last sighted at "
                 "a bearing of +177°, ~1.7m away. The depth scan recommends "
                 "turning right (~30deg).")
    # 默认 (flag off) → 不折算: 30° 微调保持 30° (chain_count=1)
    a._ff = {'vlm_angle_parse': False}
    check('#52a: 默认停用 — bearing 数字不再放大成 6 连转',
          a._parse_vlm_rotate_steps(opt, reasoning) is None)
    # 消融 on → 旧行为保留 (折算 177° → 6)
    a._ff = {'vlm_angle_parse': True}
    r = a._parse_vlm_rotate_steps(opt, reasoning)
    check('#52a: 消融 on — 旧行为保留 (177° → 6 edges)',
          r == 6, f'r={r}')
    # reasoning 无角度 → None (旧行为本就如此)
    a._ff = {'vlm_angle_parse': True}
    check('#52a: 无角度 reasoning → 不折算',
          a._parse_vlm_rotate_steps(opt, 'no angle here') is None)
    # flag 未注入 (老调用路径) → getattr 默认 = 停用
    a2 = object.__new__(AVDBAgent)
    check('#52a: 无 _ff 属性 → 默认停用',
          a2._parse_vlm_rotate_steps(opt, reasoning) is None)


# ========== #52b: 覆盖时 steps 同步 (4 项) ==========
def test_attach_sync():
    from avdb_env import AVDBEnv
    from agent import PolarAction

    e = object.__new__(AVDBEnv)
    OPTS = [
        {'chain_type': 'rotate_cw', 'chain_count': 6, 'direction': 'turn_around'},
        {'chain_type': 'rotate_ccw', 'chain_count': 1, 'direction': 'rotate_left'},
        {'chain_type': 'forward', 'chain_count': 1, 'direction': 'forward'},
        {'composite': [('rotate_cw', 1), ('forward', 1)],
         'direction': 'rotate_right_then_forward',
         'chain_type': None, 'chain_count': 0},
    ]
    obs = {'edge_options': OPTS}

    # d01r step43 病例: action 挂着 VLM 折算残留 rotate_steps=6, 覆盖选
    # [1] (rotate_ccw chain_count=1) → 同步为 1, 不再甩 180°
    act = PolarAction(0, 0)
    act.rotate_steps = 6
    out = AVDBEnv._attach_exec_idx(e, act, 1, obs)
    check('#52b: 覆盖旋转选项 → rotate_steps 同步 chain_count (6→1)',
          out.edge_idx == 1 and out.rotate_steps == 1,
          f'rs={getattr(out, "rotate_steps", None)}')

    # forward 残留同理: forward_steps=5 覆盖选 [2] (chain 1) → 1
    act2 = PolarAction(0, 0)
    act2.forward_steps = 5
    out2 = AVDBEnv._attach_exec_idx(e, act2, 2, obs)
    check('#52b: 覆盖 forward 选项 → forward_steps 同步 (5→1)',
          out2.forward_steps == 1)

    # 复合选项: 两个残留都清 (wrapper 走 composite 分支不用它们)
    act3 = PolarAction(0, 0)
    act3.rotate_steps = 6
    act3.forward_steps = 3
    out3 = AVDBEnv._attach_exec_idx(e, act3, 3, obs)
    check('#52b: 覆盖复合选项 → 残留 steps 全清',
          not hasattr(out3, 'rotate_steps') and not hasattr(out3, 'forward_steps'))

    # 不带 obs (旧调用点) → 行为不变, 只挂 edge_idx (回归保护)
    act4 = PolarAction(0, 0)
    act4.rotate_steps = 4
    out4 = AVDBEnv._attach_exec_idx(e, act4, 1)
    check('#52b: 无 obs 旧路径 → 不动 steps (兼容)',
          out4.edge_idx == 1 and out4.rotate_steps == 4)


# ========== 静态接线 (4 项) ==========
def test_static_wiring():
    check('#52a: agent _choose_action 经门控 helper 折算',
          '_parse_vlm_rotate_steps' in AGT_SRC
          and 'self._parse_vlm_rotate_steps(opt, reasoning)' in AGT_SRC)
    check('#52a: 裸正则不再直连 (旧代码行已被门控包裹)',
          "if opt.get('chain_type') in ('rotate_cw', 'rotate_ccw'):" in AGT_SRC
          and AGT_SRC.count('angle_match = re.search') == 1)
    check('#52b: _override_and_run 传 obs 给 _attach_exec_idx',
          'self._attach_exec_idx(agent_action, idx, obs)' in ENV_SRC)
    check('#52b: env 把 _ff 注入 agent (flag 可达 agent 层)',
          'self.agent._ff = self._ff' in ENV_SRC)


if __name__ == '__main__':
    test_flags()
    test_angle_parse_gate()
    test_attach_sync()
    test_static_wiring()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)

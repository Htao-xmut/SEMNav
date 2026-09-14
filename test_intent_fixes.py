"""#40 三修复单测 (2026-09-13, bmv6 mahatma 实弹三连病灶)

用户实弹指控 (logs/ObjectNav_bmv6_mahatmav6/0_of_1/0_Home_001_1/):
  a) "step28说右转30度，但是step29却是左转的结果" — 1b 强制 forward
     落到无 forward 边的节点, fallback_edge 取 dict 首键 rotate_ccw;
  b) "40多步的时候又在原地晃来晃去" — ① step27/40 warmup 六图扫的
     120° 机制转向被 HR 1d (7 步窗 ≥6 旋转 → 强制 180°) 劫持指错方向
     (1b/1c 有 B⑧/P3 豁免, 1d 没有); ② step44 APPROACH 伺服 miss=3、
     6° 对准中被"困惑重扫"(零检出21步+无新方向+区域已扫) 120° 打断。

修复:
  #40a wrapper HR 1d 补 _in_active_servo()/cur_intent 豁免 (镜像 1b/1c);
  #40b wrapper fallback 选边平移优先 (_fallback_edge_pick);
  #40c env 困惑扫/周期重扫在 APPROACH 伺服进行中 (有锁且 miss≤10) 让路。

运行: python -u test_intent_fixes.py
"""
import sys
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import logging

logging.basicConfig(level=logging.CRITICAL)

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ========== #40b 功能: _fallback_edge_pick (5 项) ==========
def test_fallback_pick():
    from avdb_sim_wrapper import AVDBSimWrapper
    w = object.__new__(AVDBSimWrapper)

    # 1 bmv6 step28 病灶: 首键 rotate_ccw, 有 left → 选平移不选首键旋转
    edges = {'R': {'edge_type': 'rotate_ccw', 'distance': 0.5},
             'L': {'edge_type': 'left', 'distance': 0.65}}
    check('#40b: 首键旋转 + 存在平移 → 平移胜出 (step28 左转病灶)',
          w._fallback_edge_pick(edges) == 'L')

    # 2 偏好序: forward 最优
    edges = {'B': {'edge_type': 'backward', 'distance': 0.65},
             'F': {'edge_type': 'forward', 'distance': 0.65},
             'L': {'edge_type': 'left', 'distance': 0.65}}
    check('#40b: forward > left/right/backward',
          w._fallback_edge_pick(edges) == 'F')

    # 3 无 forward → left; 无 left → right; 再 backward (逐级降)
    check('#40b: 无 forward → left 次之',
          w._fallback_edge_pick({'R': {'edge_type': 'right', 'distance': 0.6},
                                 'L': {'edge_type': 'left', 'distance': 0.6}})
          == 'L')
    check('#40b: 平移只剩 backward → 仍先于旋转',
          w._fallback_edge_pick({'CW': {'edge_type': 'rotate_cw', 'distance': 0.3},
                                 'B': {'edge_type': 'backward', 'distance': 0.65}})
          == 'B')

    # 4 全旋转 → 首键兜底 (原行为)
    edges = {'CW': {'edge_type': 'rotate_cw', 'distance': 0.3},
             'CCW': {'edge_type': 'rotate_ccw', 'distance': 0.3}}
    check('#40b: 只剩旋转 → 首键兜底',
          w._fallback_edge_pick(edges) == 'CW')

    # 5 静态: step() 内 fallback 走 helper (不再 dict 首键盲选)
    SRC = open('/home/tao_h/avdb_habitat_converter/scripts/'
               'avdb_sim_wrapper.py').read()
    check('#40b: step() 接线 _fallback_edge_pick (盲选已移除)',
          'self._fallback_edge_pick(edges)' in SRC
          and "target = list(edges.keys())[0]\n            desired" not in SRC)


# ========== #40a 静态: HR 1d 豁免 (3 项) ==========
def test_hr1d_exempt():
    SRC = open('/home/tao_h/avdb_habitat_converter/scripts/'
               'avdb_sim_wrapper.py').read()

    # 6 豁免存在: 伺服 + 机制意图两分支, 且在强制掉头之前短路
    i_1d = SRC.find('HARD RULE 1d: ≥6 rotates')
    i_servo = SRC.find('not hijacking (#40a)', i_1d)
    i_force = SRC.find("force_rule = 'hr_1d_180'", i_1d)
    check('#40a: 1d 有伺服/意图豁免 (bmv6 step27/40 warmup 扫被劫持)',
          0 < i_servo < i_force
          and 'APPROACH servo active (miss=0) — not hijacking (#40a)' in SRC
          and "intent '{cur_intent}' — not hijacking (#40a)" in SRC)

    # 7 豁免先于 _can_force_180 短路 (不掉头也不误记账冷却)
    i_can = SRC.find('self._can_force_180():', i_1d)
    check('#40a: 豁免短路在 _can_force_180 之前 (不误耗冷却记账)',
          0 < i_servo < i_can)

    # 8 与 1b/1c 同族: 三条硬规则口径一致
    n_exempt = SRC.count('not hijacking')
    check('#40a: 1a/1b/1c/1d 四条中三条以上带意图豁免口径',
          n_exempt >= 5, f'n={n_exempt}')


# ========== #40c 静态: 重扫让路伺服 (3 项) ==========
def test_rescan_defer():
    SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()

    # 9 守卫定义: 有锁且 miss≤10 → 不扫 (miss>10 = 真跟丢才恢复)
    check('#40c: 守卫口径 _approach_bearing 非空 且 miss≤10 → 让路',
          "_approach_alive = not (" in SRC
          and "_approach_bearing', None) is not None" in SRC
          and "_approach_miss', 0) <= 10)" in SRC)

    # 10 困惑扫与周期扫两门都挂守卫
    n_use = SRC.count('and _approach_alive \\')
    check('#40c: 困惑扫 + 周期扫两门同挂守卫 (step44 病灶)',
          n_use >= 2, f'n={n_use}')

    # 11 守卫在两触发块内 (不被别处误用)
    i_conf = SRC.find("sig.append(f'vlm repeat choice x3")
    i_peri = SRC.find('[RE-SCAN] Triggered: periodic re-orientation')
    i_def = SRC.find('_approach_alive = not (')
    check('#40c: 守卫定义位于两触发块之前',
          0 < i_def < i_conf < i_peri)


if __name__ == '__main__':
    test_fallback_pick()
    test_hr1d_exempt()
    test_rescan_defer()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)

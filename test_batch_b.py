"""批次 B 单测 — 最小仲裁语义 (2026-09-13)

B⑦ 意图标签: env 机制/强制探索 → wrapper._action_intent (consume-once)
B⑧ 1b/1c 意图豁免 + 复合动作不计连转
B⑨ 随机兜底安全池: 复合/⛔/震荡禁入/visited≥3 出池 + 分层回退
B⑩ rec-follow 修复 (#28): 真选项 obs['edge_options'] 方位匹配
B⑪ 复合子走 over-visited≥3 软停

#28 事故锚点 (smka 冒烟, logs/smoke_batchA_smka_20260913.log):
  33 次 forced 全部 AttributeError('tuple' object has no attribute 'get')
  — agent.py 旧 rec-follow 拿 a_final 占位元组当选项 dict

运行: python -u test_batch_b.py
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


def opt(et, cnt=1):
    return {'chain_type': et, 'chain_count': cnt, 'composite': None,
            'direction': et}


def comp(rot_et='rotate_cw', fwd=1):
    return {'composite': [(rot_et, 1), ('forward', fwd)],
            'direction': f'rotate_{"left" if "ccw" in rot_et else "right"}_then_forward',
            'chain_type': None, 'chain_count': 0}


def saf(idx, **kw):
    d = {'idx': idx, 'composite': False, 'blacklisted': False,
         'osc_banned': False, 'landing_visits': 0, 'landing_fresh': False}
    d.update(kw)
    return d


# ========== B⑨⑩ 功能: _choose_forced_exploration (8 项) ==========
def test_choose_forced():
    from avdb_agent import AVDBAgent
    ag = object.__new__(AVDBAgent)   # 绕过 __init__ (不需要 VLM/sim)

    # 1 #28 回归: 选项是 dict、无异常、返回安全池内索引
    obs = {'edge_options': [opt('rotate_cw', 6)] + [opt('forward'), opt('left'),
                                                    opt('rotate_ccw'), comp()],
           'option_safety': [saf(0), saf(1), saf(2), saf(3), saf(4, composite=True)]}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑨: #28 回归 — dict 选项不再炸, 返回池内索引',
          idx in (1, 2, 3) and kind in ('rec', 'random'), f'{idx},{kind}')

    # 2 安全池过滤: composite/⛔/震荡禁入/visited≥3 全出池
    obs = {'edge_options': [opt('rotate_cw', 6), comp(), opt('forward'),
                            opt('left'), opt('right'), opt('rotate_ccw')],
           'option_safety': [saf(0),
                             saf(1, composite=True),
                             saf(2, blacklisted=True),
                             saf(3, osc_banned=True),
                             saf(4, landing_visits=3),
                             saf(5)]}
    picks = {AVDBAgent._choose_forced_exploration(ag, obs)[0] for _ in range(30)}
    check('B⑨: 复合/⛔/禁入/visited≥3 全部出池 (只剩 idx5 可选)',
          picks == {5}, str(picks))

    # 3 层1 rec 跟随: depth_rec_deg=90 → right 选项 (bearing +90)
    obs = {'edge_options': [opt('rotate_cw', 6), opt('forward'), opt('right'),
                            opt('left')],
           'option_safety': [saf(0), saf(1), saf(2), saf(3)],
           'depth_rec_deg': 90.0}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑩: rec=+90° → 跟随 right 选项 (kind=rec)',
          idx == 2 and kind == 'rec', f'{idx},{kind}')

    # 4 层1 容差: rec=+170° 无 ≤100° 匹配 → 分层回退 (不跟)
    obs = {'edge_options': [opt('rotate_cw', 6), opt('forward')],
           'option_safety': [saf(0), saf(1)],
           'depth_rec_deg': 170.0}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑩: rec 超容差 → 不跟随走分层回退',
          idx == 1 and kind == 'random', f'{idx},{kind}')

    # 5 层2 新鲜落点平移优先: 唯一 landing_fresh 是平移 → 必中
    obs = {'edge_options': [opt('rotate_cw', 6), opt('left'), opt('right'),
                            opt('rotate_ccw')],
           'option_safety': [saf(0), saf(1, landing_fresh=True), saf(2), saf(3)]}
    picks = {AVDBAgent._choose_forced_exploration(ag, obs)[0] for _ in range(30)}
    check('B⑨: 层2 — 新鲜落点平移优先于其他安全项',
          picks == {1}, str(picks))

    # 6 层4 纯旋转最后: 安全池只剩旋转 → 仍可选 (不是 None)
    obs = {'edge_options': [opt('rotate_cw', 6), opt('rotate_ccw')],
           'option_safety': [saf(0), saf(1)]}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑨: 层4 — 只剩纯旋转时仍给出选择 (最后手段)',
          idx == 1 and kind == 'random', f'{idx},{kind}')

    # 7 全员不安全 → (None, 'random') 调用方回退旧行为
    obs = {'edge_options': [opt('rotate_cw', 6), opt('forward')],
           'option_safety': [saf(0), saf(1, blacklisted=True)]}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑨: 安全池空 → None (调用方回退旧随机)',
          idx is None and kind == 'random', f'{idx},{kind}')

    # 8 元数据缺失 → 保守放行 (非 AVDB wrapper 兼容)
    obs = {'edge_options': [opt('rotate_cw', 6), opt('forward'), opt('left')]}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑨: option_safety 缺失 → 保守放行不炸',
          idx in (1, 2) and kind in ('rec', 'random'), f'{idx},{kind}')

    # 9 文本正则兜底: depth_rec_deg 缺失但 trace 有 Recommended 行
    obs = {'edge_options': [opt('rotate_cw', 6), opt('forward'), opt('right')],
           'option_safety': [saf(0), saf(1), saf(2)],
           'depth_trace': '... Recommended: +85deg safe=0.6m ...'}
    idx, kind = AVDBAgent._choose_forced_exploration(ag, obs)
    check('B⑩: 文本正则兜底解析 Recommended 行',
          idx == 2 and kind == 'rec', f'{idx},{kind}')


# ========== B⑦/B⑧/B⑪ 静态断言 (源码级, 7 项) ==========
def test_static():
    AGENT_SRC = open('/home/tao_h/VLMnav/src/agent.py').read()
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    WRAP_SRC = open('/home/tao_h/avdb_habitat_converter/scripts/'
                    'avdb_sim_wrapper.py').read()

    # B⑦ 意图标签: 单点挂全 + 主路径传播 + wrapper consume-once
    check('B⑦: _override_and_run 单点写意图 (全机制覆盖)',
          'self.simWrapper._action_intent = tag' in ENV_SRC)
    check('B⑦: 主路径传播 forced_rec/forced_random 意图',
          "f'forced_{_fk}'" in ENV_SRC and 'warmup_auto' in ENV_SRC)
    check('B⑦: wrapper consume-once (读走即清, 不跨帧残留)',
          'cur_intent = getattr(self, \'_action_intent\', None)' in WRAP_SRC
          and WRAP_SRC.count('self._action_intent = None') == 1
          and "'intent': cur_intent" in WRAP_SRC)

    # B⑧ 1b/1c 意图豁免 + 复合不计连转
    check('B⑧: 1b/1c 双规则机制意图豁免',
          WRAP_SRC.count("mechanism intent '{cur_intent}' — not hijacking (B⑧)") == 2)
    check('B⑧: 复合动作记 composite (连转/交替/滑窗窗口自然断开)',
          "'composite' if opt_is_composite else desired" in WRAP_SRC
          and 'opt_is_composite = True' in WRAP_SRC)

    # B⑨ wrapper 侧安全元数据
    check('B⑨: wrapper 产 option_safety (复合/⛔/禁入/visited/fresh)',
          "obs['option_safety']" in WRAP_SRC
          and 'landing_fresh' in WRAP_SRC)

    # B⑪ 复合子走软停
    check('B⑪: 复合子走落点 visited≥3 软停',
          'not blindly re-walking (B⑪)' in WRAP_SRC)

    # agent 侧意图生命周期
    check('B⑦: agent last_forced_kind 每步重置 + reset 归零',
          'self.last_forced_kind = None   # B⑦' in AGENT_SRC
          and AGENT_SRC.count('last_forced_kind = None') >= 2)


if __name__ == '__main__':
    test_choose_forced()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)

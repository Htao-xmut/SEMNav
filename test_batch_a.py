"""批次 A 单测 — 契约强制 / VLM 判局策略 / 执行审计 (2026-09-13)

A① 契约: vlm.py 基类 __init_subclass__ + inspect 签名强制 (import 时机)
A② 异常治理: agent rec-follow 诊断埋点 + 计量 (forced_rec_used/…)
A③ 判局策略: vlm_fail_policy (CONTRACT/R1/R2/R3) + _vlm_call 通道
A④ 执行审计: wrapper FINAL-EXEC trace 回写 details.txt
A⑤ #27: 复合动作残留清理 (硬规则改写/早退 清 _pending_composite)

事故锚点 (回放数据来源):
  p26 (logs/retest_p25p26_20260913.log): 三局 18 次
    'VLM analysis failed: call_chat() got an unexpected keyword argument
     plain_text' — warmup+rescan 六图分析全崩
  p16 (logs/retest_p1p6_20260913.log): 同路径 0 次失败

运行: python -u test_batch_a.py
"""
import sys, re, types, inspect, logging
sys.path.insert(0, '/home/tao_h/VLMnav/src')
sys.path.insert(0, '/home/tao_h/avdb_habitat_converter/scripts')
import numpy as np

logging.basicConfig(level=logging.CRITICAL)  # 静音被测异常日志

PASS, FAIL = [], []


def check(name, cond, info=''):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f'  [{info}]' if info and not cond else ''))


# ========== A① 契约: import 时机签名强制 (7 项) ==========
def test_contract():
    # 1 import 成功 = 四子类全部通过 __init_subclass__ 检查 (缺参当场炸)
    import vlm
    check('A①: import vlm 成功 (四子类签名契约全过)', True)

    # 2 四子类签名都含 plain_text (inspect 轮询, 不靠字符串 grep — #26c 假绿教训)
    subs = [vlm.GeminiVLM, vlm.OpenAIVLM, vlm.OllamaVLM, vlm.QwenVLClient]
    for cls in subs:
        params = set(inspect.signature(cls.call_chat).parameters) - {'self'}
        check(f'A①: {cls.__name__}.call_chat 签名含 plain_text',
              'plain_text' in params, str(sorted(params)))

    # 3 坏子类 (缺 plain_text) → 类定义即 TypeError (import 时机 fail-fast)
    try:
        type('BadVLM', (vlm.VLM,), {'call_chat': lambda self, h, i, t: None})
        check('A①: 缺参子类类定义抛 TypeError', False, '未抛异常')
    except TypeError as e:
        check('A①: 缺参子类类定义抛 TypeError', 'plain_text' in str(e))

    # 4 **kwargs 糊弄不算过 (参数名缺失仍违约 — 宁可报错不可静默适配)
    try:
        type('KwargsVLM', (vlm.VLM,),
             {'call_chat': lambda self, h, i, t, **kw: None})
        check('A①: **kwargs 吸收仍判违约', False, '未抛异常')
    except TypeError:
        check('A①: **kwargs 吸收仍判违约', True)

    # 5-7 QwenVLClient 行为测试 (mock 传输层, 不起真 API; object.__new__ 绕过 __init__)
    sent = []

    def _fake_call(model, messages, result_format):
        sent.append(messages)
        msg = types.SimpleNamespace(content=[{'text': 'RAW LINE ok'}])
        resp = types.SimpleNamespace(
            status_code=200, output=types.SimpleNamespace(choices=[
                types.SimpleNamespace(message=msg)]))
        return resp

    def _make_qwen():
        q = object.__new__(vlm.QwenVLClient)
        q.model = 'qwen-vl-plus'
        q.max_image_res = 64
        q.system_instruction = 'Respond ONLY with JSON'
        q.client = types.SimpleNamespace(call=_fake_call)
        q.call_count = 0
        return q

    img = np.zeros((8, 8, 3), dtype=np.uint8)

    q = _make_qwen()
    out = q.call_chat(0, [img], 'scan prompt', plain_text=True)
    check('A①: Qwen plain_text=True → 无 system 消息 (规定行不被压)',
          out == 'RAW LINE ok' and not any(
              m.get('role') == 'system' for m in sent[-1]))

    q = _make_qwen()
    q.call_chat(0, [img], 'json prompt', plain_text=False)
    check('A①: Qwen plain_text=False → system 消息保留 (JSON 决策调用不变)',
          any(m.get('role') == 'system' for m in sent[-1]))

    q = _make_qwen()
    q.system_instruction = None
    out = q.call_chat(0, [img], 'p', plain_text=True)
    check('A①: Qwen 无 system_instruction 两态等价不炸', out == 'RAW LINE ok')


# ========== A③ 判局策略: vlm_fail_policy (8 项) ==========
def test_fail_policy():
    from vlm_fail_policy import VLMFailPolicy

    # 8 分类
    check('A③: classify TypeError → contract',
          VLMFailPolicy.classify(TypeError("unexpected keyword 'plain_text'")) == 'contract')
    check('A③: classify 连接超时 → infra',
          VLMFailPolicy.classify(ConnectionError('Connection timed out')) == 'infra')
    check('A③: classify JSON 解析错 → parse',
          VLMFailPolicy.classify(ValueError('Expecting value')) == 'parse')

    # 9 p26 回放: warmup 首崩 (TypeError) → CONTRACT_FAIL 立即判无效
    #   (实弹序列: logs/retest_p25p26_20260913.log 每局 warmup1+rescan5 全崩)
    pol = VLMFailPolicy()
    pol.record_fail('warmup', 'contract',
                    TypeError("call_chat() got an unexpected keyword argument 'plain_text'"))
    abort, reason = pol.should_abort()
    check('A③: p26 回放 — warmup 契约错 1 次即判无效 (step0 止损)',
          abort and reason.startswith('CONTRACT_FAIL@warmup'), reason)

    # 10 R1: warmup infra 重试耗尽 → 废局 (无契约错时)
    pol = VLMFailPolicy()
    pol.record_fail('warmup', 'infra', ConnectionError('Connection timed out'))
    abort, reason = pol.should_abort()
    check('A③: R1 — warmup 重试耗尽 1 次即废局', abort and reason == 'R1_WARMUP_FAIL', reason)

    # 11 R2: 连续计数语义 — 2 败后成功 → consec 清零 (内部状态直读)
    pol = VLMFailPolicy()
    for _ in range(2):
        pol.record_fail('rescan', 'infra', ConnectionError('Connection timed out'))
    check('A③: R2 — 同签名连 2 尚不熔断', not pol.should_abort()[0])
    pol.record_ok('rescan')
    check('A③: R2 — 成功后连续计数清零',
          pol.consec_count == 0 and pol.consec_sig is None)

    # R2 标注: 全新 policy 连 3 同签名 → 熔断且标 R2 (区分系统性 vs 交错退化)
    pol = VLMFailPolicy()
    for _ in range(3):
        pol.record_fail('rescan', 'infra', ConnectionError('Connection timed out'))
    abort, reason = pol.should_abort()
    check('A③: R2 — 同签名连 3 熔断废局 (优先于 R3 标注)',
          abort and reason.startswith('R2_CONSEC'), reason)

    # 12 R3: 滑窗 10 次内 3 次分散失败 (时好时坏, 非连续) → 废局
    pol = VLMFailPolicy()
    seq = [False, True, False, True, False] + [True] * 5  # 3 败分散, consec≤1
    for ok in seq:
        if ok:
            pol.record_ok('rescan')
        else:
            pol.record_fail('rescan', 'infra',
                            ConnectionError('Connection timed out'))
    abort, reason = pol.should_abort()
    check('A③: R3 — 滑窗 10 内 3 分散失败 → 废局 (退化态)',
          abort and reason.startswith('R3_WINDOW'), reason)

    # 13 p16 回放: 同路径 0 失败 → 全程零触发
    pol = VLMFailPolicy()
    for site in ['warmup'] + ['rescan'] * 5:
        pol.record_ok(site)
    check('A③: p16 回放 — 0 失败零误报', not pol.should_abort()[0])

    # 14 计量输出 (版本化 + 分桶)
    st = pol.stats()
    check('A③: stats 含阈值版本/调用数/无效位',
          all(k in st for k in ('vlm_policy_version', 'vlm_critical_calls', 'vlm_invalid')))


# ========== A③ _vlm_call 通道: stub self 功能测试 (3 项) ==========
def test_vlm_call_channel():
    from types import SimpleNamespace
    from avdb_env import AVDBEnv
    from vlm_fail_policy import VLMFailPolicy

    class StubVLM:
        def __init__(self, behavior):
            self.behavior, self.calls = behavior, 0

        def call_chat(self, h, imgs, prompt, plain_text=False):
            self.calls += 1
            return self.behavior(self.calls)

    def make_env(behavior):
        return SimpleNamespace(
            agent=SimpleNamespace(actionVLM=StubVLM(behavior)),
            _ep_stats={'vlm_critical_fails': 0, 'vlm_invalid': 0,
                       'vlm_invalid_reason': ''},
            _vlm_policy=VLMFailPolicy(CONFIG={'retry_sleep_s': 0.0})
            if False else None)

    # 15 契约错 → 返回 '' + 局无效置位 + 计数 (p26 病灶路径)
    def raise_te(n):
        raise TypeError("call_chat() got an unexpected keyword argument 'plain_text'")
    e = make_env(raise_te)
    out = AVDBEnv._vlm_call(e, 'p', [], 'warmup')
    check('A③: _vlm_call 契约错 → 返回空串 + invalid 置位 + CONTRACT_FAIL',
          out == '' and getattr(e, '_vlm_episode_invalid', False)
          and e._ep_stats['vlm_critical_fails'] == 1
          and 'CONTRACT_FAIL' in e._vlm_invalid_reason,
          f"{out!r} {getattr(e, '_vlm_invalid_reason', '')}")

    # 16 infra 2 次后第 3 次成功 → 重试救回, 零失败零无效
    def flaky(n):
        if n <= 2:
            raise ConnectionError('Connection timed out')
        return 'RECOVERED'
    e = make_env(flaky)
    e._vlm_policy = VLMFailPolicy()
    e._vlm_policy.CONFIG['retry_sleep_s'] = 0.0
    out = AVDBEnv._vlm_call(e, 'p', [], 'rescan')
    check('A③: _vlm_call infra 重试 2 次后成功 → 零失败零无效',
          out == 'RECOVERED' and not getattr(e, '_vlm_episode_invalid', False)
          and e._ep_stats['vlm_critical_fails'] == 0)

    # 17 正常成功 → 原文直通
    e = make_env(lambda n: 'OK')
    e._vlm_policy = VLMFailPolicy()
    check('A③: _vlm_call 正常路径原文直通',
          AVDBEnv._vlm_call(e, 'p', [], 'warmup') == 'OK')


# ========== A②/A④/A⑤ 静态断言 (源码级, 10 项) ==========
def test_static():
    AGENT_SRC = open('/home/tao_h/VLMnav/src/agent.py').read()
    ENV_SRC = open('/home/tao_h/VLMnav/src/avdb_env.py').read()
    BASE_SRC = open('/home/tao_h/VLMnav/src/env.py').read()
    WRAP_SRC = open('/home/tao_h/avdb_habitat_converter/scripts/'
                    'avdb_sim_wrapper.py').read()

    # A② agent: rec-follow 静默 except 消除 + 计数
    #   (批次B 重写后标记演进: 选择逻辑移入 _choose_forced_exploration,
    #    异常文案改 '强制探索选择抛异常', 计数与 log.exception 保留)
    check('A②: rec-follow 异常 log.exception + 计数 (FORCED-REC 埋点)',
          '强制探索选择抛异常' in AGENT_SRC
          and "_forced_rec_errors = getattr" in AGENT_SRC
          and '[FORCED-REC]' in AGENT_SRC)
    rec_block = AGENT_SRC[AGENT_SRC.find('rec_deg = obs.get'):AGENT_SRC.find("logging.warning(f\"Forced exploration")]
    check('A②: rec-follow 块内裸 except:pass 已消除',
          'except Exception:\n                            pass' not in rec_block)
    check('A②: agent.reset 计数归零 (forced_rec_used/errors/random_used)',
          all(s in AGENT_SRC for s in
              ['self._forced_rec_used = 0', 'self._forced_rec_errors = 0',
               'self._forced_random_used = 0']))
    check('A②: EP-STATS 增 forced_rec_used/forced_rec_errors/forced_random_used',
          all(f"'{k}'" in ENV_SRC for k in
              ['forced_rec_used', 'forced_rec_errors', 'forced_random_used']))

    # A③ env: 判局中止 + _vlm_call 通道
    check('A③: _run_episode 判局提前中止 + EP-STATS 无效位落盘',
          '_vlm_episode_invalid' in BASE_SRC and 'vlm_invalid_reason' in ENV_SRC)
    check('A③: 六图调用走 _vlm_call (warmup/rescan 双站点)',
          "self._vlm_call(" in ENV_SRC
          and "'warmup' if step_tag == 'warmup_scan' else 'rescan'" in ENV_SRC)

    # A④ trace 回写
    check('A④: FINAL-EXEC 回写 details.txt (决策/覆盖/硬规则/执行四层)',
          'FINAL-EXEC' in ENV_SRC and '_record_exec_trace' in BASE_SRC)
    check('A④: wrapper last_exec_trace 含 forced_by/intent/sub_walk',
          all(s in WRAP_SRC for s in
              ["'forced_by': force_rule", "'intent': cur_intent",
               "'sub_walk': sub_walk"]))

    # A⑤ #27 残留清理
    check('A⑤: #27 — desired 改写后丢弃 pending legs (分歧检测)',
          'pending legs discarded' in WRAP_SRC and 'parsed_desired' in WRAP_SRC)
    check('A⑤: #27 — 早退路径清 pending (不拖到下一帧)',
          WRAP_SRC.count('#27: 早退不清 pending') == 1)


if __name__ == '__main__':
    test_contract()
    test_fail_policy()
    test_vlm_call_channel()
    test_static()
    print('=' * 40)
    print(f"✅ {len(PASS)} passed, ❌ {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f'  FAIL: {n}')
        sys.exit(1)

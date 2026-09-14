"""VLM 失败判局策略 (批次 A③, 2026-09-13, 两轮专家评审收敛版)

背景 (p26 事故): plain_text 加错 vlm.py 类 → 三局 18 次关键 VLM 分析
全部 TypeError 崩溃, 但系统静默降级继续跑完 50 步 × 3 局, 产出 0/3 的
伪数据 + 白烧 API。教训不是"崩了", 是"崩了还在跑且跑完才发现"。

规则 (专家1 分层阈值 + 专家2 三规则, 合成):
  CONTRACT  契约类错误 (TypeError/AttributeError/NameError) 1 次即局无效,
           不重试 — 代码 bug, 之后的数据全是伪数据
  R1        warmup 关键分析失败 (重试耗尽) ≥1 → 局无效 — 全链路最早探针,
           p26 型死亡在 step0 拦截 (省 17 次调用/局)
  R2        同错误签名连续 ≥3 → 局无效 — 系统性 bug 熔断; 瞬时网络抖动
           不会同签名连错 3 次 (专家2: p16 零失败 → 真实故障率上界 ~15%,
           故 N 不能为 1)
  R3        最近 10 次关键调用失败 ≥3 (30%) → 局无效 — "时好时坏"退化态。
           (专家2 原案软降级续跑是为真机器人物理成本设计的; 仿真里续跑
            产出垃圾数据还烧 API → 改判无效+提前终止, config 留模式开关)

失败分类 → 重试语义:
  contract  不重试, 立即计关键失败 (见 CONTRACT)
  infra     超时/连接/限流类 → 重试 2 次 (指数退避), 重试成功不计失败
  parse     JSON/格式类 → 重试 1 次
  语义无效 (present=0 / 方位作废 / SCAN 无解析) 不算调用失败 — 那是机制
  失败, 走既有降级/重扫路径, 与本模块无关。

阈值全部集中在 CONFIG (禁止散落硬编码); 判局局终随 EP-STATS 落 CSV
(vlm_invalid / vlm_invalid_reason), 无效局不许悄悄丢。
"""

from collections import deque


class VLMFailPolicy:
    """按局持有的失败策略状态机。env 侧 _vlm_call 是唯一喂入点。"""

    # 阈值版本化: 调参后旧局无法解释 → 局终记录版本号 (专家2 要求)
    CONFIG_VERSION = 'v1-20260913'

    CONFIG = {
        'retry_infra': 2,          # 基础设施类重试次数
        'retry_parse': 1,          # 解析类重试次数
        'retry_sleep_s': 1.0,      # infra 重试基础退避 (秒, 线性递增)
        'consec_same_sig_invalid': 3,   # R2: 同签名连续失败熔断
        'window_size': 10,              # R3: 滑动窗口长度
        'window_fails_invalid': 3,      # R3: 窗口内失败数阈值 (10 次 30%)
        'warmup_fail_invalid': 1,       # R1: warmup 重试耗尽即废局
        # 真机部署切软降级续跑时置 False (本仓当前只有仿真, 提前终止)
        'sim_early_terminate': True,
    }

    CONTRACT_TYPES = (TypeError, AttributeError, NameError, ImportError)

    def __init__(self):
        self.total_calls = 0
        self.total_fails = 0
        self.fails_by_site = {}
        self.consec_sig = None       # 当前连续失败签名 (成功即清)
        self.consec_count = 0
        self.window = deque(maxlen=self.CONFIG['window_size'])
        self.abort_reason = None     # 一旦置位不再翻转 (首个触发原因保留)
        self.contract_hit = None     # (site, sig) 首个契约错误

    # ---------- 分类 ----------
    @classmethod
    def classify(cls, exc):
        """异常 → 'contract' / 'infra' / 'parse'。

        contract 按异常类型判定 (代码 bug 的确定性信号);
        其余按消息关键词判 infra (网络/服务侧), 兜底 parse。
        """
        if isinstance(exc, cls.CONTRACT_TYPES):
            return 'contract'
        msg = str(exc).lower()
        infra_kw = ('timeout', 'timed out', 'connection', 'connect',
                    'throttl', 'rate limit', 'ratelimit', '429', '502',
                    '503', '504', 'unavailable', 'reset by peer', 'network')
        if any(k in msg for k in infra_kw):
            return 'infra'
        import json as _json
        if isinstance(exc, (ValueError, _json.JSONDecodeError)):
            return 'parse'
        return 'parse'  # 未知归 parse (重试 1 次后计失败, 保守)

    @staticmethod
    def error_signature(exc):
        """错误签名 = 异常类型 + 消息前 60 字符 (R2 同源判定用)。"""
        return f"{type(exc).__name__}:{str(exc)[:60]}"

    # ---------- 喂入 ----------
    def record_ok(self, site):
        self.total_calls += 1
        self.consec_sig = None
        self.consec_count = 0
        self.window.append(False)

    def record_fail(self, site, kind, exc):
        sig = self.error_signature(exc)
        self.total_calls += 1
        self.total_fails += 1
        self.fails_by_site[site] = self.fails_by_site.get(site, 0) + 1
        if self.consec_sig == sig:
            self.consec_count += 1
        else:
            self.consec_sig, self.consec_count = sig, 1
        self.window.append(True)
        if kind == 'contract' and self.contract_hit is None:
            self.contract_hit = (site, sig)

    # ---------- 判定 ----------
    def should_abort(self):
        """返回 (abort: bool, reason: str|None)。优先级 CONTRACT > R1 > R2 > R3。"""
        if self.abort_reason:
            return True, self.abort_reason
        c = self.CONFIG
        if self.contract_hit:
            reason = f"CONTRACT_FAIL@{self.contract_hit[0]}[{self.contract_hit[1]}]"
        elif self.fails_by_site.get('warmup', 0) >= c['warmup_fail_invalid']:
            reason = 'R1_WARMUP_FAIL'
        elif self.consec_count >= c['consec_same_sig_invalid']:
            reason = f"R2_CONSEC_{self.consec_count}x[{self.consec_sig}]"
        elif sum(self.window) >= c['window_fails_invalid']:
            reason = f"R3_WINDOW_{sum(self.window)}/{len(self.window)}"
        else:
            return False, None
        self.abort_reason = reason
        return True, reason

    # ---------- 计量 ----------
    def stats(self):
        return {
            'vlm_policy_version': self.CONFIG_VERSION,
            'vlm_critical_calls': self.total_calls,
            'vlm_critical_fails': self.total_fails,
            'vlm_fails_by_site': dict(self.fails_by_site),
            'vlm_invalid': 1 if self.abort_reason else 0,
            'vlm_invalid_reason': self.abort_reason or '',
        }

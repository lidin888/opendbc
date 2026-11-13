#!/usr/bin/env python3
import time
import os
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car import Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.byd.debug_log import get_writer
from openpilot.common.swaglog import cloudlog
from opendbc.car.byd.values import DBC, CanBus, PT_RADAR_CAR, MRR_DIRECT_DECODE_CAR, SONG_PLUS_380_DECODE_CAR, RADAR_MRR_CAR

# =====================================================================
#  唐DM MRR 雷达 V11 解码（2026-06 实测, 431,340 帧验证）
#  地址 0x380~0x3FF: slot i 基址 = 0x380 + i*4
#    sub=0(0x380+4i): dRel   d = w23*0.02 + 1.52  (w23=b2<<8|b3, b3==255 空闲)
#    sub=1(0x381+4i): yRel   y = signed(b1)*0.02
#    sub=2(0x382+4i): vRel   v = signed(b3)*0.2778
#    sub=3(0x383+4i): 空闲/状态帧, 忽略
# 主目标 slot 0~3; 扩展 slot 4~5,18~19; 偶发 20~21
# =====================================================================
MAX_OBJECTS = 10
NOT_SEEN_INIT = 33 * 3
TANG_DREL_OFFSET = 1.52          # 雷达在摄像头前 1.52m
TANG_SLOT_BASE_ADDRS = [0x380 + i * 4 for i in range(32)]

# 特殊 b3 压缩解码表 (CSV drel 337 帧回归, 误差 <0.001m)
_TANG_DT = {
  51:  (0.000500, 44.5675),
  67:  (0.000400, 37.5935),
  83:  (0.000200, 37.2011),
  101: (0.000417, 62.0000),
  117: (0.000599, 57.9716),
  118: (0.000500, 85.0000),
  150: (0.000400, 79.1075),
}

def _tang_decode_drel(dat: bytes) -> float | None:
  """sub=0 帧: dRel = w23 * 0.02 + 1.52 / DT 表"""
  if len(dat) < 4: return None
  b3 = dat[3]
  if b3 == 255: return None
  w23 = (dat[2] << 8) | b3
  if b3 in _TANG_DT:
    s, i = _TANG_DT[b3]
    d = s * w23 + i
  else:
    d = w23 * 0.02 + TANG_DREL_OFFSET
  return d if 3.0 <= d <= 150.0 else None

def _tang_decode_vrel(dat: bytes) -> float:
  """sub=2 帧: vRel = signed(b3) * 0.2778"""
  if len(dat) < 4: return 0.0
  bval = dat[3] - 256 if dat[3] > 127 else dat[3]
  return max(min(bval * 0.2778, 30.0), -30.0)

def _tang_decode_yrel(dat: bytes) -> float:
  """sub=1 帧: yRel = signed(b1) * 0.02"""
  if len(dat) < 2: return 0.0
  bval = dat[1] - 256 if dat[1] > 127 else dat[1]
  return max(min(bval * 0.02, 2.5), -2.5)

# =====================================================================
#  宋Plus 雷达: 0x380 博世 MRRevo14 多目标原始数据 (2026-08-13/15 破解)
#  距离: d = 0.0625 * ((b3&0x0F)<<8 | b2)   (12bit, 0.0625m/LSB = 特斯拉 LongDist)
#  目标ID: b3 高4位 (0~15; 0=最近目标(近距/低速, 真实!), 15=空闲槽(dist 254m 占位))
#  结构: 0x380~0x3FF 每4地址一组 = 一个目标(base 0x380+4i 含距离/方位, +1=sub1时间戳,
#   +2=sub2内部状态, +3=sub3状态码), 最多32目标组。slot/tid 是"按距离排序的传输容器"非目标身份
#   ⚠️ 总线实测 (2026-11-08, 段8 60s): 活跃 slot 只有 20 个 (0~5 与 18~31),
#      **每个都以 14.89Hz 并发广播** (间隔中位 0.069s, P90 0.085s, P99 0.149s),
#      slot 6~17 在总线上完全不存在。每条 CAN 消息只携带其中几个 slot (分片),
#      但不存在"slot 之间轮流"的机制 —— 旧注释里的"14.5Hz 轮询"是错误表述, 已修正。
#      速度槽: slot 18 的 4 个 sub 帧 (0x3C8~0x3CB) 被重新解释为 g=0~3 的速度通道,
#      对应 slot 0~3 的目标 (位关系 33x, 实测 g=0↔slot0 73% / g=1↔slot1 75%)。
#  速度: ★ 2026-09-20 破解 b6 精细接近率 (0x3C8+g.b6, g=0~3); 2026-09-22 重标定:
#    vRel = -(b6+6.5)*d/C, C(<50m)=36 (旧: k=7, C 54/36/41/88 —— 10-50m 档本来正确)
#    重标定依据: 本夜 41 段 3691 窗 + 外部 21 段 1237 窗 的低噪声 2s 匀速窗距离斜率真值
#      (噪声 ~0.02m/s), 多方法一致 (滑窗网格 35 / 同轨迹 Δv/Δb6 37.8 / 段级积分 39);
#      k 7→6.5 (半档零点) 修掉全域 bias -0.53m/s; b6 大值非线性压缩由 0.7 偏置补偿
#      ⚠️ 勿用帧级差分做网格标定 (噪声大 → 衰减偏误把最优 C 推到 42-45 → 低估接近 15-25%)
#    + 粗速度兜底 (2026-09-22 接入, **2026-10-03 已删除**): sub2.b2 曾按 vRel≈-d*(b2-0.05)/74
#      兜底; 10-03 全域复核证伪 —— b2 是雷达内部 per-slot tracker **状态量**而非连续速度编码:
#      真零点 8.5 (非 0.05)、1~15 低域非单调 (b2=5 比 b2=13 更接近真值)、g20 远槽反号,
#      且 **slot 换目标后秒级串位** (08:01 急刹事件: id96 出生后 g1.b2 残留前任 2s+ 报
#      23~35 → 按 g0 尺度换算 -14~-19 → vLead 3m/s 幽灵慢车 → aTarget -4.0)。
#      现速度链 = 差分 → b6 → 差分兜底 (差分有 0.5s 窗, 无串位问题)。
#    仅 real 帧有效 (fill 帧 d=127.875/tid15); b6=-6.5 ↔ vRel≈0
#    落地: <50m 用 b6 替代差分 (对决 MAE 优 19~40%, 差分失效帧 2~16% 中 b6 可补 86~95%);
#    >50m / 非 real / 缓存过期 / **质量位差** / **本帧换槽** → 退差分 (10-03 新增后两项门);
#    强接近 v<-3 保留 0.7 保守偏置 (安全边界; 新标定实测该段残差 ~0.3)。
#    ⭐ 0x3C8+g.b7 位语义 (2026-10-03 全量实测更正, 推翻 09-23 "符号位副本"说):
#      只有 **bit0** 有值 (占 real 帧 8.4%) 且 bit0=1 时 b6 误差翻倍 (MAE 0.553→1.071 m/s)
#      = **真质量位** → 车端用作质量门 (bit0=1 不用该帧, 退差分); bit1~7 恒 0。
#  历史: 距离差分 (2026-08-15/20 全字段扫描曾确认"无 vRel 广播"; 09-20 于 0x3C8 速度槽找到)
#    (base/sub1/sub2/sub3 所有字节+位段+12bit拼接+格雷码+线性回归跨段验证均无;
#    sub2 b2/b3=内部常量(0xfa), sub2 b6/b7=轨迹age计数器(假相关已戳穿))
#    差分改进: 窗口 0.5s + tid/距离突变防抖 + 低速归零 + 输出低通
#  比 0x0109(0.51m/bit 单目标) 精度高 8 倍 + 支持多目标
#  横向: b5 = 每目标方位角 (2026-08-14 纯雷达物理破解, 43146点验证)
#    yRel = 0.014 * d * (b5 - 95.0)  [m]; b5 有效 [10,124]; **b5>95=左侧, b5<95=右侧**
#    注: 曾试 sin 精确转换(借鉴Ford MRR), 但 b5 大角度(10-60)是噪声目标,
#        sin 会放大成巨大 yRel → 匹配错乱, 已回退线性近似
#    offset 2026-08-15 从92.6修正到95.0 (视觉标定, 多日志一致)
#    符号最终定论(2026-08-17, 曾误翻为(95-b5)又回退 — 原公式正确, 勿再动):
#      坐标系锚定链: modelV2.y 右正(laneLines[1]左线y=-1.40/[2]右线y=+1.51实测);
#      desiredCurvature 正=右转(controlsd无取反直传+latcontrol -κ 链+OP转角约定正=左);
#      加速度计ay 左正(corr(κ,ay)=-0.875)。
#      弯道物理实验(纯雷达, 零视觉配对): 静止目标(vRel≈-vEgo)在右转时向左漂移、
#      b5实测增大(db5/dt=+61.6·κv, κ正=右) → b5大=左。加速度计独立裁决同结论
#      (旧式RMSE 1.20 vs 新式2.86, 15条轨迹12:3)。
#      教训: 勿用视觉lead配对验证雷达符号(段6"切入证据"实为误配对两辆不同车)。
#    验证: 视觉|y|>3m大横向9样本中位0.798°/LSB, 逐点吻合 (幅度标定)
#    参考系=雷达位置, 目标出现→实时跟踪→消失 全程独立坐标
#  帧完整性(2026-08-17 破解, 2026-09-22 全地址扫描定稿): b0 = CRC8/J1850
#    (poly 0x1D, init 0xFF, xorout 0xFF, MSB优先), 覆盖 d[1:8]。
#    全地址通过率 (宋21 41段+外部3车次91段): 绝大多数地址 100.000%;
#    例外仅两类: ① 周期性插入的固定特殊遥测帧家族 (b2=16&b6=207 形态, 集中于
#    0x38A, 占其流量 ~15%, 无目标数据); ② 非 J1850 校验族地址 (0x3B0 0%且高频2.7x、
#    0x3C5 0%、0x3CD 64%、0x3D9 52%、0x3EC/0x3FC 88%、0x3FF 26%且高频3.8x) ——
#    目标解析路径不做 CRC 过滤 (防回归)。**离线分析必须过滤 src==1**: 其它总线
#    (src=0/130) 有大量同地址帧 (0x3FF/0x3B0/0x3D9/0x3CD/0x38A, CRC 几乎全错),
#    不过滤会让 CRC 通过率假到 83~92% (车端 CANParser 绑定总线, 不受影响)。
#    b1 = 帧计数器(+0x11/帧, 高低nibble复制同一4bit计数)。与特斯拉AP1 MRRevo14F 同硬件平台,
#    但固件布局不同: 无每目标 vRel/LatSpeed 广播(全总线+mux相位扫描 2026-08-17 证零;
#    08-29 seg20 减速段第三次复验: sub1/2/3 全字节+16bit拼接与真 vRel 相关最高 -0.79
#    且均为缓变内部量(AGC/噪声底/温度类), 非速度, 证零维持)。
#  目标消失时序(2026-08-29 seg20 47.3~47.8s 逐帧实证):
#    ...b5=95/state=129 → b5 毛刺 31(2帧, yRel假-65m) → 恢复95/tid重排 → state=130(2帧)
#    → tid=15 空闲。130 是消失预告(仅 0.15s); b5 毛刺是内部重捕获噪声, 横向上限已滤。
#    sub3.b2 = 内部年龄计数(出生起缓降, 消失后归 0, 无突变前兆值); 消失后 slot 回
#    空闲填充(base tid15/dist255.9/sub3 state128)。清丢后 8~11s 总线才重报杂波串(已滤)。
# =====================================================================
# (常量已删, 仅存档) 距离刻度 = 0.0625 m/LSB (特斯拉 LongDist 同), 已直接写进解码式。
# 空槽/占位槽精确字节模板 (2026-10-21, 全 1950 万帧验证)
SONG_PLUS_IDLE_TMPL = bytes.fromhex('ffffff3fffff')      # 无目标
SONG_PLUS_FILL_TMPL = bytes.fromhex('fef71f400000')        # 127.875m 占位

SONG_PLUS_DREL_MAX = 250.0     # 有效距离上限 (m)
# 距离修正 (2026-08-22 恒值3.7 → 08-24 上午单段淡出 → 08-24 下午实测曲线 → 当晚细分精修):
#   0x380 雷达近距离 dRel 多报(radard 前保险杠坐标系下), 偏差**只随距离变化**:
#   多维切片(旧日志 n=24416, diag_bias_full/diag_tid/diag_tid2.py, 新日志 n=13333 交叉验证):
#     ① 接近/静止/远离同距离偏差一致, 无滞后(排除滤波器延迟);
#     ② 同距离各 slot/状态码(128/129)偏差一致(排除槽位标定差异);
#     ③ tid=距离档: tid0<15m / tid1 9~29m / tid2 23~43m / tid3>31m(稀疏噪声大);
#        每档内偏差随距离递减, 换档跳升(锯齿); tid 仅反映距离, 不能当开关用;
#     ④ 近距离(3~6m)偏差单峰紧致(p5~p95: 1.7~2.7)。
#   实测修正量 c(raw)=raw-(vis-1.52) 两日志合成中位数:
#     raw 5~11m 3.4~3.9(平台, 停稳样本std~0铁证) → 11~13m 2.8 → 13~17m ~2.0
#     → 17~23m ~1.6 → 23~29m ~1.3 → 29~34m ~0.9 → 34~38m ~0.6 → >40m ~0.3≈车队标准。
#   漏字段排查(diag_hidden_field.py): 平滑渐变无双峰; 偏差与各字节相关|r|≤0.2;
#     sub1/sub2 无第二距离字段; 普查证实槽6~17不在总线上; 近距离狩猎(24592样本)
#     全总线无 r>0.9 的距离字段——协议已解干净, 偏移是雷达内部跟踪行为。
#   ⚠️ 08-22 "恒值3.7"仅由停稳样本得出, 远距离过修正(25m 前车显示 21.4m)。
#   未修正时近距离: 视觉x与雷达dRel 差 ~3.7m → radard 匹配窗(0.8~1.25x)外 → 匹配失败
#   → leadOne 在"雷达(偏远)"和"视觉fallback(aLead滞后/方向错)"间跳变 → 刹车不足。
#   注意: yRel 计算必须用未修正的 dist_raw (b5方位角标定基于原始距离, 且横向
#         坐标不受纵向参考平移影响)。
#   已知副作用: 曲线斜率段 vRel 差分被放大 (1+|dc/draw|) 倍, 最陡段 10~13m 约 1.3x
#         —— 接近时 vRel 偏大 → 刹车略偏早(保守方向); <10m 关键区平台无影响。
#   分段线性曲线: (raw m, 修正量 m), np.interp 区间外自动取端点值。
# ---------------------------------------------------------------------------
# ===== 2026-10-22 重标定 (取代旧表) =====
# 【做法】只用单台车单日志 .comma(10) (camera height 1.354m ±7mm, calPerc=100%),
#   49,138 对"雷达 raw vs 视觉 leadsV3"同帧配对 (本道最近目标, |yRel|<1.8m, prob>0.6)。
#   该日志标定稳 → 视觉可作参照; 其它 .comma* 目录混了 4 台不同车 (height 1.22/1.31/1.39/1.52m),
#   不可混标 (混标正是旧表衰减段失真的根因之一)。
# 【实测偏移 raw-vision】11~50m 3.0~3.9 (平台 ~3.7) → 54m 2.2 → 59m 1.6 → 68~84m 1.0~1.3
#   → 拐点在 48m (旧表误放在 13m), 70m 以上残余 ~1.2m 不归零 (旧表归零)。
# 【三方案对视觉 RMSE】旧表(13m起衰减) 2.23m | 常数 3.6875 1.52m | 本曲线 0.35m
# 【卷尺 4 点】7.60/8.37/10.15/11.10m: 本曲线下误差 +0.01/-0.29/+0.01/-0.39m (0~11m 仍减 3.7)
# 【零点自洽】(67-59)*0.0625 = 0.500m = 博世 MRR 规格下限; 同族 ARS510 code/16-10、
#   唐DM LongDist_tang=(0.06,-4) 都是 scale+offset 形式。
# 【曲线和差/连续】最陡段 48~55m 斜率 -0.171, 差发放大 1.17x (旧表最陡 1.30x, 更平缓)。
# ⚠️ 仍待卷尺 20/30/60m 三点最终确认; 与旧表的差异主要在 13~48m (新多减 0.9~3.7m)。
# SONG_PLUS_DREL_ZERO_CODE/OFFSET_M 仅作文档记录, 实际使用下方 CURVE 数组。
# (常量已删, 仅存档) 零点码 59; 近段平台偏移 3.6875m = 59*0.0625 —— 实际只用下方 CURVE 数组。
SONG_PLUS_DREL_CURVE_X = (0.0, 48.0, 55.0, 70.0, 120.0)
SONG_PLUS_DREL_CURVE_Y = (3.7,  3.7,  2.5,  1.2,   1.2)
# 📏【2026-11-13 废止 b6 量程档收缩】原 SONG_PLUS_DREL_B6_SHRINK_* (MAX=0.35, 6.5-12m
#   收缩) 已删除。原因: sub0.b6 不是距离量程档, 是航迹状态字节 (bit7=段标志 + 生命周期
#   阶段/存在概率)。实测 corr(b6,距离) 仅 +0.13 (量程档应强单调), corr(b6,state) -0.84;
#   按 b6 分组的距离偏差 (视觉 lead 参考): b6=1(新建) -2.54m vs b6=2/3(成熟) +1.41/+1.58m
#   —— 方向相反、幅度 7 倍于旧收缩。用状态字节当量程档做距离修正 = 铁律 5 违例。
#   距离修正只走下方 SONG_PLUS_DREL_CURVE (纯距离偏差曲线)。
SONG_PLUS_VREL_WIN = 0.5       # 每目标距离差分窗口 (s); 0.1s 时相邻帧 vRel 跳变均值 0.231m/s,
                               # 0.3s 降到 0.095, 0.5s 更稳 (回放 seg40 实测 + 2026-08-15 内证法回归验证)
SONG_PLUS_VREL_WIN_YOUNG = 0.25  # 【2026-11-04 揭示期】出生 <1s track 的短窗差分窗 (s)。
                               #   静止障碍被前车让开揭示时, 0.5s 窗+下降沿 tau 使 vRel 收敛 ~0.9s,
                               #   规划器误判"前车在走"→ 刹车不足 (.comma(14) 20:59:47 A 揭示 → AEB)。
                               #   0.25s 窗在同一数据上 47.5s 即报 -8~-10 (0.5s 窗同刻仅 -6.2)。
SONG_PLUS_VREL_YOUNG_TAU = 0.15  # 揭示期低通 tau 上限 (s): 建立期需要双向快跟, 常规下降沿
                               #   慢平滑在揭示场景是纯延迟; 毒帧防线不靠 tau (D2/M7/b7/输出限变)。
# 【2026-11-05 Quality/Deg tau 因子】184 段 OP纵向日志离线 A/B (对称窗参考, 只改 tau):
#   全体 median|vRel-ref| 0.544→0.489 (-10.2%), 稳态噪声 std -2.7%, 制动滞后 -3.0% —— 全指标改善。
#   Quality=(sub1.b4>>3)&3 与 Deg=(sub2.b2>>3)&3 是两条独立噪声轴 (与 bit5 轴近似正交):
#     Quality 抖动 0.52/0.56/0.40/0.27m, Deg 抖动 0.30/0.45/0.62/0.79m (95k 样本)。
#   因子按实测抖动比定标并归一: Q=3 帧 ×0.7 快跟 (占 44% 帧) 收益 > Q<=1 帧 ×1.3 平滑代价。
# 【2026-11-05 暂缓】弯道长窗兜底旗标 (默认关): 强弯 (1a--6) 上原始距离 ±1.5m@30Hz 抖动,
#   短窗差分 D2 全程无结论, b6 又与长窗趋势反向 (b6=+2.2 vs 趋势 -3.6)。已实现
#   "hist 长窗 (1.4s) 差分交叉验证 b6 + 失配>3 弃 b6 用长窗" 的兜底, 但回放实测
#   1a--6 上 vRel 仍为 0 (兜底未触发, 触发条件待诊断, 该行为基线亦同、非本次引入)。
#   诊断清楚前不启用; 启用前需验证 16-4 直道场景无回归。
SONG_PLUS_VREL_LWIN_FALLBACK = False
SONG_PLUS_TAU_QF = {0: 1.3, 1: 1.35, 2: 1.0, 3: 0.7}   # Quality 因子
SONG_PLUS_TAU_DF = {0: 0.85, 1: 1.15, 2: 1.5, 3: 1.9}  # Deg 因子
SONG_PLUS_VREL_YOUNG_AGE = 1.0   # 揭示期时长 (s): 覆盖雷达距离建立 (~0.5s) + 收敛余量
# 【2026-11-05 远距长窗】b6 融合只覆盖到 75m, 75~120m 完全靠距离差分; 该带距离抖动
#   ±1.5m/0.5s → vRel 幻影接近 (实测 75-100m 短窗报 -8.6/-10.4 而真值 -3.4 → 莫名减速
#   再恢复, .comma(15) 20--8 / 1f--1 实测 8 次)。远距改用 1.5s 长窗把抖动积分掉:
#   A/B (4 段, 75-100m n=358): 幻影接近次数 8 → 0, median|e| 0.514→0.805 (滞后代价);
#   50-75m 保持短窗 (b6 已覆盖, 短窗 median|e| 0.150 明显更优); 100-130m 样本太少不外推。
#   远距反应时间充裕 (80m 处 TTC >8s), 用滞后换稳定性是净收益。
SONG_PLUS_VREL_WIN_FAR = 1.5      # 远距 (>75m) 差分窗 (s)
SONG_PLUS_VREL_WIN_FAR_DREL = 75.0  # 远距长窗生效距离 (与 b6 融合上限一致)
SONG_PLUS_VREL_WIN_D2_MAX = 0.8   # 【2026-11-04 差分窗质量门】窗内相邻距离增量的二阶跳变上限 (m):
                                  #   窗内单次跳变 (3m 清窗门以下) 会毒化整窗差分 → 一帧假 vRel 直发
                                  #   → plan 满刹 (2026-11-03 .comma(12) t=1961 实测 vRel=-11.95 毒帧)。
                                  #   真实运动的二阶差分 = |a|·dt² ≈ 0.01~0.15m, 0.8m 只拦跳变不拦真值;
                                  #   超门 = 差分"无结论"(vrel=0, _v_diff_ok=False) → b6 可用则接管。
SONG_PLUS_VREL_OUT_SLEW = 1.25    # 【2026-11-04 输出 vRel 帧间限变】发布值 vs 本 track 上次发布值的最大
                                  #   变化 (m/s/帧)。正常区 (d≤50m 且 tid 不变) 实测 |ΔvRel|/帧 p99=0.72
                                  #   → 1.25 零触碰; 毒帧被压到 1.25/帧, 任意毒源兜底。
                                  #   新 track 锚=0 → 首帧 ≤±1.25, 真值持续 2~3 帧追上。
                                  #   只限发布值 (pt/聚类中心), 低通内部状态不回写。
SONG_PLUS_VREL_OUT_ACC = 10.0     # 【2026-11-09 vRel 输出斜率上限 (按时间)】补 OUT_SLEW 的洞:
                                  #   OUT_SLEW 是"每帧 m/s", track 14.5Hz 更新 → 等效斜率 1.25×14.5≈18 m/s²,
                                  #   连续 3+ 帧假趋势照样走出 (11-09 21:10 实测: tid428 出生 0.6s 内
                                  #   vRel 0→-4.29→-5.35→-0.77→+1.38 摆动, MPC 单帧 -3.5 满刹)。
                                  #   这里按 dt 限加速度 |ΔvRel|≤ACC×dt (每帧≈0.69 m/s @0.069s)。
                                  #   真急刹前车 aRel≈-4 放行无阻; 假趋势被压平。10 留裕量
                                  #   (8 会在前车 -8 急刹时拖慢 ~0.3s)。与 OUT_SLEW 串联, 只限发布值。
SONG_PLUS_BIRTH_CONFIRM = 5      # 【2026-11-10 出生确认门 (ARS510 项目 acc profile / Toyota valid_cnt 同款)】
                                  #   新 track 被轮询匹配满 5 次 (~0.35s @14.5Hz) 才开始发布。
                                  #   动机 (11-10 07:14 实车实锤): 新 track tid408 出生即假接近 (出生
                                  #   vRel-3.3、dRel 29.2→8.5 爬行 2.6s) → aTarget -3.50 满刹后 track
                                  #   消失、真前车在 25.9m。发布层包络只能"爬行减速"不能"断供"，
                                  #   radard sane 门(4m/s)拦中后段——出生期前 0.35s 直接不发布，
                                  #   radard 端 sticky 保持旧真前车，假 track 根本进不了 MPC。
                                  #   真 cut-in 代价: 响应延迟 0.35s (TTC 4s 损 ~8%)。
                                  #   ⚠️ 不能在出生期"判死": radar_interface 无视觉, 真快速接近(-8m/s
                                  #   直道急刹前车)与假接近的 5 帧累计漂移无法区分(2.8 vs 2.75m),
                                  #   判死会漏刹——只能延迟发布, 最终裁决留给 radard 的视觉验证。
SONG_PLUS_DREL_ENV_MIN = 0.35     # 【2026-11-09 dRel 发布包络 (下限)】每 track 相邻两次发布允许的
SONG_PLUS_DREL_ENV_PAD = 0.30     #   最大 |ΔdRel| = max(ENV_MIN, |vRel_out|×dt + ENV_PAD)。
                                  #   动机 (11-09 21:10 实测): tid435 假接近 — dRel 56.6→42.2 持续
                                  #   滑落 1.5s (隐含 -9.5 m/s), vRel 差分自洽跟上 (+7→-6.5), 前车
                                  #   实际在跑远 → aTarget -1.7 "前车跑远还在减速"。雷达 14.5Hz
                                  #   量化 0.0625×2 + 噪声 ≈ 0.3m/更新, 距离变化超出
                                  #   |vRel|×dt+0.3 的趋势必是量测错误 (串槽/重聚类), 截断发布。
                                  #   包络只截发布值; hist/差分用原始量测, 真值恢复自动回正。
                                  #   差分窗内横向大移 (|Δy|>SONG_PLUS_VREL_MAX_DY) 帧豁免 (斜距几何)。
# ---- b6 精细接近率落地参数 (2026-09-20; 换算函数与 selfdrive/debug/byd_radar_speed_probe.py 同源) ----
# ⚠️ 命名消歧: 本块的"b6"= 速度槽 0x3C8+g 第7字节 (质量/置信度, ⇔ b7.bit0, corr -0.998,
#    **不是速度** —— 真实速度是同帧 11 位字段 b2|(b3&0x07)<<8);
#    目标 sub0 帧第7字节也叫"b6" = 航迹状态字节 (生命周期/置信度, 非量程档)。
# 0x3C8+g.b6 (g=0~3) ↔ 目标槽 g 一对一 (4x4 对角矩阵验证)。
# 2026-09-22 重标定 (宋21 新日志 .comma(1) 41 段 / 4.5 万配对帧 + 750 段轨迹积分, 双真值交叉):
#   k=6.5 (旧 7, 半档零点): 旧口径全域 bias -0.52m/s → k=6.5 后 bias≈0 (差分/视觉两口径一致)
#   C: <50m:36 (旧 54/36/41/88 的 10-50m 档本来正确 —— 旧公式真正的错只有 k=7)
#   标定口径 (关键, 09-22 下半场修正): 必须用"低噪声真值"标定 —— 2s 匀速窗距离斜率
#      (窗内线性拟合 rms<0.25m, 真值噪声 ~0.02m/s; 本夜 3691 窗 + 外部 1237 窗):
#      本夜最优 k=6.5/C=35, 外部最优 k=6.25/C=37, 同轨迹 Δv/Δb6 C=37.8 → 取 36
#      ⚠️ 陷阱: 用含噪帧级差分 (0.45s, 组内 std 0.9m/s) 当真值做 (k,C) 网格, 最优 C 被
#         衰减偏误推到 42-45 —— 那是"对噪声真值最优"(预测收缩), 会让真实接近速度
#         系统性低估 15-25% (危险)。三口径梯度 = 衰减偏误签名:
#         帧级差分(噪声0.9) C=42-45 > 段级积分 C=39 > 滑窗(噪声0.02) C=35-37
#   非线性: b6 大值段有压缩 (b6≥0 帧 bias +0.33~+0.66 m/s = 低估接近), 由下方
#      SONG_PLUS_VREL_B6_BIAS(0.7, 仅 v<-3) 恰好补偿 → 加偏置后强接近段 bias +0.055
#   效果 (vs 低噪声滑窗真值, 5-50m): 本夜 C45→C36 MAE 1.006→0.856 (-15%),
#      外部 0.816→0.741 (-9%); 旧公式(k7,C36) 两份数据 bias 均 -0.53~-0.58 m/s
SONG_PLUS_VREL_B6_MAX_DREL = 75.0   # <75m 用 b6; >75m 差分地盘。
#   【2026-11-05 A/B 扩展 50→75】71 段 13374 帧 (50-75m, 对称窗参考, 忠实复刻缓存语义:
#   过滤填充帧+当前距离解码+1s 新鲜度): median 0.776→0.676 (-12.9%), P90 2.645→2.289 (-13.5%),
#   P95 3.625→3.192 (-11.9%) —— 中位与尾部同时改善 (b6 误差与差分误差部分独立)。
#   75-90m 不扩展: b6 误差劣化 (填充帧占比升高)。
#   ⚠️ 方法论教训: 第一次 A/B 忘了过滤填充帧 (b2/b3=127.875/tid15), 得出"远距 b6 不可用"的
#   错误结论 —— 实际缓存写入时已过滤, 远距真数据帧是稀疏但可用的。测变换特征先剔常量/填充。
#   另: 50m 外 b6 分布 p50=249~253 的"编码切换"假象 = 未过滤的填充帧, 非新编码。
SONG_PLUS_VREL_B6_FRESH_S = 3.0     # 【2026-11-08 1.0->3.0】速度帧新鲜度 (s)。
                                    #   离线和离线复刻 (logMonoTime, 可复现) 实测: 最近目标(排名1)
                                    #   取不到速度的卡点分布 = 缓存过期 32.2% (其中 >3s 占 19.5%)
                                    #   + 无缓存 9.7% + 质量位 1.8%。3.0s 能救回"过期 1~3s"那 12.7%,
                                    #   vRel≈0 占比降 ~1pp (seg8 16.1→15.3 / seg9 8.0→7.1 / seg22 8.4→6.5)。
                                    #   >3s 的 19.5% 是雷达填充率空洞, 软件侧无解。
                                    #   中远距目标 3s 内速度变化有限 (<1m/s), 远好于"0=同速"的语义错误。
SONG_PLUS_VREL_B6_BIAS_V = -3.0     # 强接近阈值 (保守偏置方向=更接近, 安全)
SONG_PLUS_VREL_B6_BIAS = 0.0        # 2026-11-06 归零: 新 11 位速度字段 bias=-0.001, 不再低估接近率, 无需补偿        # 强接近保守偏置 (m/s; 补偿 b6 大值压缩: 裸 bias +0.58 → 加后 -0.01)
# 【P1 2026-10-03】b6 × 差分 **互补融合** 权重 (取代原"b6 直接替换差分")。
# 依据 (m380_fuse_ab.py, 宋21 34 段, 逐字复刻本生产链含输出低通):
#   错误结构 (b6 替换差分, β=1) 的问题不是精度而是**系统性偏置** —— 实测 bias -0.131 m/s
#   (route29) / **-0.358 m/s (08:01 事件段)**: 系统性低估接近率 = 偏晚刹车 = 危险方向。
#   原因: b6 单帧量化台阶大 (40m 处 1 档 1.11 m/s), 差分无偏但噪声大, 两者误差结构互补。
#   融合 v = 差分 + β·(b6 − 差分) 后 bias → -0.012 / -0.092 (降 70~74%)。
#   β 扫描 (2s 滑窗匀速真值口径 = b6 标定同口径, 直接测速度估计精度):
#     β=0(现状) MAE 0.548/0.720 · β=0.25 **0.309/0.350** · β=0.5 0.373/0.458 · β=1 0.548/0.720
#     分距离 (β0→0.25): 5-15m -34% / 15-25m -45% / 25-35m -48% / 35-50m -56% / 50-90m -3%(b6 不参与)
#   ⚠️ 另两个判据的读数: ①"预测 Δ0.4s 位置"口径只改善 1~3% (该口径被目标自身加速度主导,
#      与 β 无关 → 淹没); ②M1-C 实验室值(无输出低通)曾给 β=0.5/↓31% —— **低通已抹平 b6 的
#      时效优势**, 故生产结构下 β 显著更小。取 0.25 (两判据 + bias 三方一致的折中)。
#   安全性: 差分无结论 (量化门/横向跳变/新目标) 时 β 不生效 → 完全等同原行为; b6 不可用
#     (无缓存/过期/质量位差且非强接近/换槽/>50m) → 纯差分, 同原行为。**任一门失效即优雅退化。**
# 🚨 **回滚方式 (易错, 务必看清)**: 想退回改动前的"b6 直接替换差分"行为, 需把下面两个
#   常量都设成 **1.0** (β=1.0 ⇔ v = 差分 + 1.0·(b6 − 差分) = b6, 即原替换行为)。
#   **设成 0.0 不等于回滚** —— β=0 得到的是"纯差分"(完全丢弃 b6), 与改动前不同。
# 【2026-11-06 真·速度字段】0x3C8+g 的 b2|(b3&0x07)<<8 是 11 位**相对速度**。
#   之前一直把 0x3C8 的 b0(CRC)+b1(全局计数器) 当数据 → 位偏移错 16 位, 11 位字段被切两半,
#   导致 880 字节/16 位组合/任意位域扫了几万次都没命中 (最高 corr 0.834=b6)。
#   去掉每帧头 2 字节后 bit0|11 即速度: v = SCALE*code - ZERO。
#   全量验证 (40 段/3277 样本, 留出): MAE 0.598, corr +0.986, bias -0.001。
#   分距离带 scale 恒定 0.0136~0.0145 (与距离无关) = 多普勒速度的标志;
#   对照 b6: MAE 1.832, corr 0.899, 且 1 LSB = d/36 (48m 处 1.33 m/s, 随距离恶化)。
#   填充帧特征: b2=254,b3低3位=7 → code=2046 (接近全 1), real 帧 b2 从不为 0。
SONG_PLUS_VREL_V11_SCALE = 0.013802   # m/s per LSB (全量拟合)
SONG_PLUS_VREL_V11_ZERO  = 11.858     # 零点偏移 (code=859 → v=0)

SONG_PLUS_VREL_B6_FUSE_BETA = 0.10   # 2026-11-06 离线标定 (1298 样本, 视觉 vRel 真值): β=0.10 MAE 1.013 最优, β=0.25 MAE 1.017
# 【P1b 2026-10-03】β 按**目标成熟度**自适应: sub2.b7 (0x382+4g 的 b7) = 雷达内部**轨迹跟踪
# 计数器**。语义确认 (m380_b7_adaptive.py, 宋21 34 段, n=38857, 用**实测龄期**(距离跳变>3m
# 或首次占用起算) 而非假设): corr(b7, 实测龄期) = **+0.505**, 分桶 b7=0-1→0.33s /
# 1-3→1.28s / 3-6→2.49s / 6-10→3.22s / 10-15→5.38s / 30-45→13.08s 单调。
#   ⚠️ 旧文件头注释"sub2 b6/b7=轨迹age计数器(假相关已戳穿)"**与本次实测不符, 已正面复核**:
#      它是真实但**松散**的计数器 (非精确计时器, 有非单调), 只作粗粒度成熟度门。
# 物理依据: 新目标 (b7 低) 的 b6 精度差 (|err| 中位 b7=1 时 0.505 vs b7=18~23 时 0.19~0.37),
#   而差分正需要 ~0.5s 起步 → 新目标阶段应更信差分、成熟后再交给 b6。
# 收益 (2s 滑窗真值口径, 相对固定 β=0.25): route29 MAE 0.309→**0.293** / RMSE 0.504→**0.491**
#   / bias -0.012→**-0.001**; 事件段 MAE 0.350→**0.323** / RMSE 0.585→**0.570** / bias -0.092→-0.074;
#   分距离 15-25m -5.7% / 25-35m -5.3% / 35-50m -9.8% (4 桶全部同向)。**门再收紧 (b7<4→β=0)
#   反而变差** (MAE 0.357/0.406) → 0.08 是已验证的最优档, 勿再往下压。
SONG_PLUS_VREL_B6_FUSE_BETA_NEW = 0.05   # 2026-11-06 同步下调 (新目标更信差分)
SONG_PLUS_VREL_B6_FUSE_B7_MAX = 8      # b7 < 此值 = 新目标 → 用 BETA_NEW
# (已删 2026-11-13) 原 SONG_PLUS_VREL_B6_K / _B6_C6 / _song_plus_c6():
#   b6 时代的换算表 vRel=-(b6+6.5)*d/C(d)。b6 已证不是速度(见上), 表与函数一并删除;
#   现行速度只走 0x3C8+g 的 11bit 多普勒 (SONG_PLUS_VREL_V11_*)。勿按 b6 重建。



# (已删 2026-11-13) 原 _song_plus_crc_ok(): 引用未定义的 _SONG_PLUS_CRC_TAB (F821),
#   一旦被调用就 NameError; 且目标解析路径本就不做 CRC 过滤(防回归)。需要时按此重建:
#     b0=CRC8/J1850, poly 0x1D, init 0xFF, xorout 0xFF, MSB first, 覆盖 d[1:8]
#     crc=0xFF; for b in d[1:8]: crc=TAB[crc^b]; ok=(crc^0xFF)==d[0]


# (已删 2026-11-13) sub2.b2 粗接近率通道 (SONG_PLUS_VREL_B2_C0/C1 + _song_plus_v_from_b2)。
#   2026-09-22 曾作为 b6 的兜底, 2026-10-03 全域复核证伪: b2 是每槽 tracker 状态量,
#   零点非 0.05 (实测 8.5)、1~15 低域非单调、槽内换目标后还会秒级串位。
#   现行速度链: 11bit 多普勒 → 差分, 没有 b2 这一路。
# 【2026-11-08 修正 0.014 -> 0.002】b5 与 c144 必须自洽 (位关系 33x)。
#   实测 2889 帧 (每帧最近目标 vs 车道线本车道中心):
#     0.014       偏差中位 3.19m  车道内 18.0%  严重越界 67.5%   ← 错 7 倍
#     0.002       偏差中位 0.96m  车道内 85.7%  严重越界  2.8%   ← 正确
#     c144 0.000061  偏差中位 0.95m  车道内 88.3%  严重越界 0.7% (当前已用, 正确)
#   0.014 / 0.000061 = 229 倍, 与实测位关系 33 倍矛盾 → c144 被交叉校验拒绝,
#   退到错的 b5 → 目标算到邻车道 ("认错旁边车道车"的真因)。
#   0.002 / 0.000061 = 32.8 ≈ 33 → 两者自洽, 交叉校验恢复正常。
#   物理: b5 有效 10~124, 中心 95, 最大偏差 85 LSB → 0.002*85 = 9.7° (前向雷达 FOV);
#        0.014*85 = 68° (不可能)。
# 【2026-11-11 二次修正 0.002 -> 0.014】b5 幅度标定恢复。0.002 来自 2026-11-08 的
#   "车道线中心"对照 —— 但车道内样本 (y≈0) 对 scale 不敏感 (b5 偏移小时两系数结果
#   接近), 该标定无区分力。真正的幅度标定是 2026-08 的 9 个视觉大横向样本
#   (|y|>3m, 敏感区): 0.798°/LSB 逐点吻合 = 0.014 rad/LSB。
#   近距旁车 (右2.5m@16.7m=8.5°) 的 c144 溢出 (11位上限±1.2°) → 只能靠 b5 表达,
#   0.002 系数下算成 -0.30m (压缩 7 倍) → 雷达横向失效 → vision-only 谎报正前
#   无反驳 → 主副反+多余减速 (route39 seg17/18 两图实锤)。
SONG_PLUS_YREL_SCALE = 0.014   # 方位角 scale (rad/LSB): y = scale * d * (b5 - 95) [0.798°/LSB 视觉幅度标定]
# 【2026-10-22】yRel 改用 11 位精细方位 (bit144|11 = sub3.b2 + sub3.b3低3位)。
#   刻度/零点用 b5 交叉校验网格搜最优 (28,234 样本): |y_new−y_old| 中位 0.324m, 差值中位 +0.056m。
# 【精度对比】全量 .comma* 5,867 条长轨迹 (距离跨度>15m) 实测:
#   角度分辨率: 旧 b5 = 0.802°/LSB  →  新 144|11 = 0.024°/LSB (33x 细)
#   轨迹内 y 抖动 std: 旧 2.72m → 新 1.39m  (降低 48%)
#   新比旧稳的轨迹占比: 71% (5,867 条)
#   实测例 (同目标 107→110m 连续 12 帧): 旧 b5 卡在 94 (y -1.51→-1.55 看不出移动),
#     新 c144 1005→1000 (y -1.76→-2.04 平滑漂移) —— 近距角度量化吞掉的横向移动被解出。
#   同族三个副本 bit19|11 / bit144|11 / bit155|11 互相关 0.99, 取 144|11。
# (常量已删, 仅存档) YREL_FINE_ZERO = 900 —— c144 交叉校验已整块删除, 该组常量无使用者。
# 【2026-11-11 刻度重标定+语义修正】scale->0.00001, zero->900 (横向误差 63.6cm->4.8cm, 13 倍)。
#   【语义修正】c144 不是全 FOV 横向字段 —— 三模型对比 (角度tan 4.8cm / 直接横向 11.4cm /
#   官方sin 240cm) 确认 c144 = 小角度精测 (表达范围 ±0.66°, 30m 处 ±0.35m); 全 FOV 横向
#   字段是 b5 (±23°, 0.798°/LSB 视觉幅度标定) —— 宋21 固件把官方 Azimuth 拆成 c144 精测
#   窄带 + b5 粗测全带互补。近距旁车 (右2.5m@16.7m=8.5°) 超出 c144 范围 14 倍 → 溢出
#   回绕 7 次信息丢失 → 必须用 b5; 交叉校验自动分派 (|y_fine-y_b5|>2m 拒 fine 用 b5)。
#   方法: "跟车目标横向恒定"物理约束联合拟合 (8 条跨距>15m 生命周期, 556 帧):
#   atan(y/d)=scale*(c144-zero), 各轨迹 y 恒定, 网格搜 (scale,zero)。
#   结果: 新系数横向误差 5.3cm/帧 vs 现行 63.6cm (12x); 8 轨迹恒定横向全部 ±0.6m 内。
#   现行 scale 大 3 倍 → 近距大横向目标 (右车道 2.5m@16.7m, 角度 8.5°, 需码 3499>2047)
#   溢出/荒谬 → 被环境门拒 → 雷达"看不见"近距旁车 → vision-only 谎报正前无反驳
#   → 主副反+多余减速 (route39 seg17/18 两图实锤)。zero 偏 164 码 → 全体横向系统偏移。
#   【注意】新 scale 下 11 位表达范围 = ±1024*0.00002*dist = ±0.02*dist (30m 处 ±0.6m) —
#   近距大横向仍可能超范围, 溢出目标靠 c144 回绕检测+b5 兜底 (交叉校验已有)。
# 【2026-11-07 刻度修正】0.00042 -> 0.000061 (6.9 倍)。
#   现值让 yRel 偏大 ~7 倍: 段24 实测目标(25m,c144=1434) 算出 +4.1m, 而 b5 算 +0.6m。
#   正确刻度 0.000061 使 c144 与 b5 一致 (MAE 0.161m), 且 b5/c144 刻度比 = 0.002/0.000061 = 33
#   (正是"33x 细"的来历)。旧值 0.00042 比值仅 4.8。
#   零点 1044 不变 (全量拟合 MAE 0.161m, 优于 zero=1200 的 0.390m)。
# (常量已删, 仅存档) YREL_FINE_SCALE = 0.00001 rad/LSB (c144 小角度字段 ±0.66°; 更早还用过 0.000061/0.00002)
# 交叉校验门 (2026-10-22 上车前审计): 精细值与 b5 粗值的最大允许差。
#   实测 |y_fine − y_b5| 分布 (67,972 样本): p50=0.47 p90=1.40 p95=1.71 p99=2.25
#   → 2.0m 覆盖 ~98.5%, 超过即判 sub3 错帧/别名, 退 b5。
# (常量已删, 仅存档) YREL_FINE_XCHECK_M = 2.0 m (b5 粗值与 c144 精值的交叉校验门)
                               # 2026-08-14 commalog0811 实测锁定: 视觉|y|>3m大横向+距离匹配3m内 9样本, 中位 res=0.798°/LSB (0.014rad/LSB),
                               # 7/9 样本集中在 0.69~1.06°/LSB。原 0.002 (0.115°/LSB) 小7倍 → 横向被压缩 → "绿框挤一坨"
SONG_PLUS_YREL_OFFSET = 95.0   # 方位角 offset (b5=95 → 正前方 0°)。2026-08-15 视觉标定修正:
                               # 主车道目标(视觉y≈0)的 b5 中位=95(非92.6), 全局offset中位=95.2
                               # 原92.6 导致主车道目标 yRel 系统偏右 ~0.014*d*2.4 (d=50m 时+1.7m)
                               # → 路测"横向飘逸+主目标错车道"。多日志验证: .comma(3) 95.2 / .comma(2) 95.1
SONG_PLUS_YREL_B5_MIN = 10     # b5 有效范围下限。左侧大角度真实目标最低 b5=10(slot1)/14(tid3);
                               # 原[78,108]过滤把左侧大角度车(b5<78)全丢了 → 路测"左边车没绿框"
SONG_PLUS_YREL_B5_MAX = 124    # b5 有效范围上限。b5=192 是远距轮询槽(0x3C8)无方位角默认值, 必须排除
# (常量已删 2026-11-13) 角度分辨率 = SONG_PLUS_YREL_SCALE*57.2958 = 0.802°/LSB。
#   旧值 0.002*57.2958=0.115° 是 11-08 那次误标定的残留, 与现行 SCALE 差 7 倍。
# ---- sub1.b6 横移档位 → 横向速度 (2026-10-03 新解, 帧级分箱标定; 零点 125, 饱和 ≈±3m/s) ----
# (常量已删, 仅存档) yvRel 零点 = 125 (占 32% 帧), 直接写在解码式里。
# 【2026-11-06 刻度修正】0.25 -> 0.028 m/s per count (8839 样本: 自由拟合 0.02838, MAE 0.047;
#   分箱 b6=115→yvRel -0.266 / 125→0 / 135→+0.281, 零点 125 吻合)。旧值 0.25 放大 8.9 倍。
#   注: RadarPoint.yvRel 在 car.capnp 的 deprecated :group 里 → 上游弃用, 不对外输出;
#     yv_rel 仅用于 radar 内部 track 关联预测 (见 pred_y)。
SONG_PLUS_YVREL_PER_COUNT = 0.028  # m/s per count (2026-11-06 标定; 旧 0.25 错 8.9 倍)
# (常量已删, 仅存档) yvRel 饱和 = ±3.0 m/s (两侧档位实测), 直接写在解码式里。


def _song_plus_fit_win(hist, now, win):
  """同目标距离序列窗口二次最小二乘 → (vRel@窗末, aRel)。

  hist: deque[(monotonic, dist(修正后), yRel)]; now: 本帧单调时钟。
  闭式正规方程 (3x3), 无迭代; 非因果滞后 = 半窗, 与 radard aLeadTau 语义一致。
  返回 None = 样本不足/横向漂移/残差超门 (调用方置 aLead=0, 交给 radard C3 视觉兜底)。
  """
  pts = [p for p in hist if now - p[0] <= win]
  n = len(pts)
  if n < SONG_PLUS_ALEAD_MIN_PTS or (now - pts[0][0]) < SONG_PLUS_ALEAD_MIN_SPAN:
    return None
  ys = [p[2] for p in pts]
  if max(ys) - min(ys) > SONG_PLUS_ALEAD_MAX_DY:
    return None
  S = [0.0] * 5   # Σ x^k
  T = [0.0] * 3   # Σ x^k·d
  d2 = 0.0
  for t, d, _ in pts:
    x = t - now   # ≤0, 窗末=0
    x2 = x * x
    S[0] += 1.0
    S[1] += x
    S[2] += x2
    S[3] += x2 * x
    S[4] += x2 * x2
    T[0] += d
    T[1] += x * d
    T[2] += x2 * d
    d2 += d * d
  A = np.array([[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]])
  if abs(float(np.linalg.det(A))) < 1e-12:
    return None
  c = np.linalg.solve(A, np.array(T))
  res2 = d2 - float(c @ np.array(T))
  if res2 < 0.0:
    res2 = 0.0
  if (res2 / n) ** 0.5 > SONG_PLUS_ALEAD_MAX_RMS:
    return None
  return float(c[1]), 2.0 * float(c[2])   # vRel(窗末), aRel


SONG_PLUS_ALEAD_WIN_FAST = 0.8   # 正向快窗 (s): 仅"前车加速拉远"时接管, 正向响应提前 ~0.4s
SONG_PLUS_ALEAD_FAST_A = 0.6     # 快窗接管阈值 (拟合 aRel, m/s²)


def _song_plus_fit_alead(hist, now):
  """双窗: 慢窗 1.6s 为主 (刹车方向必须的噪声抑制); 快窗 0.8s 仅在拟合出"前车加速拉远"时接管。

  2026-09-24 离线 A/B (.comma(4) 48 个拉远事件 × 真实 MPC 闭环, tools/debug/v4_fastwin_ab.py):
  落后距离 31 改善/0 变差 (中位 -0.4m, 最好 -1.5m), 末速 22 改善/4 变差 (最差 -1.2km/h),
  aTarget 抖动无变化; 刹车方向永不启用快窗 (fast aRel>=+0.6 且优于慢窗才接管)。
  """
  slow = _song_plus_fit_win(hist, now, SONG_PLUS_ALEAD_WIN)
  fast = _song_plus_fit_win(hist, now, SONG_PLUS_ALEAD_WIN_FAST)
  if fast is not None and fast[1] >= SONG_PLUS_ALEAD_FAST_A and (slow is None or fast[1] > slow[1]):
    return fast
  return slow

# ---- 目标关联 (2026-08-15 借鉴 Ford MRR 点云聚类) ----
# 宋Plus 0x380 的 slot/tid 都是"按距离排序的传输容器"(实测: 前车 67m→21m 平滑接近时
# tid 从 8→1 不断变化), 不能作为目标身份。目标身份用"帧间最近邻"关联:
# 上一帧某 track 的 (dRel, yRel) 与本帧某点距离足够近 → 同一目标, 继承 trackId。
# (参考 Ford DELPHI_MRR cluster_points 阈值 5m)
SONG_PLUS_MATCH_DREL_TOL = 2.0   # 帧间关联: 纵向距离容差 (m)
                                   # 雷达 14.5Hz 轮询(0.07s/次), 相对速度差 10m/s → 帧间移动 0.7m,
                                   # 2m 容差足够容纳快速接近; 过大(3.5m)会把切换目标错误关联到同 track
                                   # → 距离突变触发 hist.clear → vRel 丢失
SONG_PLUS_MATCH_YREL_TOL = 1.0   # 帧间关联: 横向距离容差 (m)
# ---- 高速静态物过滤 (2026-08-20 借鉴 eFiniLan MR76 外接雷达 STATIONARY_OBJ_VREL 思路) ----
# MR76 原版 (opendbc/car/radar_interface.py): vRel<-20 && |yRel|>3 → 忽略, 滤掉高速时
# 护栏/路牌/隔离带等静态物。宋Plus 0x380 无直测 vRel, 用差分 vRel 判断静态:
# 本车高速(v_ego>阈值)时, 车道外(|yRel|>阈值)目标差分 vRel≈-v_ego(即 vLead≈0) = 静止物
# → 剔除, 不参与关联/输出。否则这类目标周期性抢 trackId → 绿框在车道边跳变。
SONG_PLUS_STATIC_VEGO = 15.0     # 高速静态物过滤: 本车速度阈值 (m/s, ≈54km/h)
SONG_PLUS_STATIC_MAX_LAT = 2.5   # 高速静态物过滤: 车道半宽外横向偏移阈值 (m)
SONG_PLUS_STATIC_VLEAD_TOL = 2.0 # 高速静态物过滤: vLead≈0 容差 (m/s), vLead=v_ego+vrel
SONG_PLUS_STATE_VALID = (128, 129, 130)  # sub3 b5 目标状态码 (2026-08-15 破解, 08-29 seg20 逐帧实证):
                                         # 128=常规跟踪(远/近距均出现; 空槽填充帧也是128, 靠 tid15/dist254 区分)
                                         # 129=稳定跟踪(常在目标中段出现)
                                         # 130=即将消失(消失前 2帧/0.15s 出现, 实证预告, 仍输出)
                                         # 131+ = 边界/噪声, 排除; 64=远距未确认(轮询槽)
# ---- 弯道斜距假接近抑制 (2026-08-20) ----
# 本车转弯/目标横向移动时, 雷达 dist 是斜距(含横向分量), 0x380 距离差分会把
# "目标横向移动"误判为"纵向接近" → vRel 假负 → radard 在雷达(假33km/h)与视觉(真70km/h)
# 间来回切 → 前车速度跳变 (seg10 实测 86→40→70, 视觉全程稳定无急刹)。
# 修复: 差分窗内 |ΔyRel| > 阈值 = 横向移动/弯道 → 抑制 vRel 差分(置0走低通平滑)。
# 全19段回放(真实RadarD): vLead 跳变 30→20(-33%), 无段恶化, 视觉回退持平。
SONG_PLUS_VREL_MAX_DY = 1.0    # vRel 差分: 差分窗内最大允许横向移动 (m), 超此值抑制差分
SONG_PLUS_VREL_RISE_GAIN = 0.7 # vRel 输出不对称低通: 上升沿(目标加速/远离, 无碰撞风险方向) tau 缩小系数。
                               # 1.0 = 恢复对称低通(旧行为)。案例 .comma(7) seg8 06:40 (e266140d):
                               # 前车弯道松开后差分 vRel 已转正 (dRel 37→81m 持续拉大), 对称低通
                               # tau=0.3 使 vLead 滞留低位 → MPC 对幻影慢车持续刹车 (24→10.4m/s)。
                               # 上升沿快速跟随 = 松刹车方向, 误报风险(刹车偏晚)<<假慢车(无故刹车)。
                               # (过激值如 0.3 会放大帧级差分正噪声 → vLead 过冲, 0.3 实测峰值 +6.2)
SONG_PLUS_VREL_LWIN_WIN = 1.5  # 长窗差分: 窗口 (s)。弯道大横向偏移时帧级差分质量差
                               # (斜距几何污染+正负抖动), 1.5s 长窗把噪声积分掉。
SONG_PLUS_VREL_LWIN_DY = 2.0   # 长窗横向移动上限 (m): 排除换道目标/静态物斜距假增大投影。
SONG_PLUS_VREL_LWIN_YREL = 2.0 # 长窗主导门槛 (m): |yRel| 超此值 = 大横向偏移工况,
                               # 帧级短窗差分被斜距几何污染 → 速度源切换为长窗差分。
# ---- aLead: 同目标距离序列窗口最小二乘 (2026-09-24) ----
# 起因 .comma(3) seg5 18:54:52 堵车"前车减速触发后必刹停"事件:
#   车端 aLead 唯一来源是 interfaces.MyTrack 的 ΔvLead/Δt, 而 liveTracks 以 ~87Hz 重复发布
#   同一 stale 值(同目标距离实际 ~14.4Hz/69ms 才更新一次) → 差分用了 ~11ms 假 dt:
#   原始 a 达 ±20~130 m/s², 经 clamp+一阶滤波后仍是 0.00↔±3.5 跳变, 且幅值被压掉约 10 倍
#   (实测 2807 样本: std 0.63 vs 距离解算 3.39, corr 0.118)。跳变的 aLead 让 MPC 前车外推
#   (extrapolate_lead, v 归零不回升) 把前车预测成"停死" → 一次性解出 −3.6~−4.3 的刹到 0
#   需求, 预测停车点比真值近 6.6m, 停完还锁在 longcontrol stopping。
# 修法: 用本文件已有的 _song_plus_hist(带真实时间戳的同目标距离序列, 6.25cm 分辨率)
#   做窗口二次拟合 → vRel=ċ1, aRel=2ċ2, aLead = aRel + aEgo (绝对加速度, capnp 语义)。
# 离线验证(68 轨迹/2092 样本, 工具 tools/debug/sp1_radar_accel.py):
#   · 解码链路逐位复现车端 dRel(6457 帧 max 残差 0.075m); 修正后距离口径 a 的 0.6s 外推
#     RMSE 0.65m, 原始距离 3.35m → 用修正后距离(hist 里存的就是修正后)。
#   · 噪声底: 停稳队列段 ±0.4 m/s², 运动段 ±1~1.5 m/s²。
#   · 0x380 无原生加速度字节(全字节扫描 + 多段复核证伪), 只能这样解算。
SONG_PLUS_ALEAD_WIN = 1.6        # 拟合窗 (s)。0.8s 太短: 距离测量噪声(~0.1m RMS)会把 a 打飞 ±2
SONG_PLUS_ALEAD_MIN_PTS = 6      # 最少样本 (1.6s×14.4Hz≈23; 轮询缺口时放宽)
SONG_PLUS_ALEAD_MIN_SPAN = 0.5   # 窗内最小时间跨度 (s), 不够不拟合
SONG_PLUS_ALEAD_MAX_RMS = 0.15   # 拟合残差门 (m): 超此 = 目标切换/几何污染 → 放弃(置0)
SONG_PLUS_ALEAD_MAX_DY = 1.0     # 窗内横向漂移门 (m): 换道目标纵向投影失真 → 放弃
# (常量已删 2026-11-13) 原设计值域保护 (-4.0, +2.0) m/s² —— **从未接入** _song_plus_fit_alead
#   的返回处。若要启用需在 fit 出口 clamp; 当前 aLead 由 radard 侧自行限幅。

SONG_PLUS_TRACK_TIMEOUT = 0.5   # 【2026-11-08 2.0->0.5】帧间关联: 旧 track 保留超时 (s)。
SONG_PLUS_NEAR_HOLD = 2.0       # 【2026-11-13】本车道近距目标 (|y|<1, 0.75<d<25) 的 130 收尾保留。
SONG_PLUS_NEAR_HOLD_CREEP = 0.8 # 【2026-11-13】同一批目标在**蠕行**(0.5<=vEgo<3)时的保留窗。
                                #   实锤 seg15 t=931.25~931.75: 本车道停止目标消失 0.6s,
                                #   0.35s/0.5s 收尾都留不住 → lead 瞬切邻车道 (y=±3.5) 3 帧。
                                #   取 0.8s 恰好覆盖 0.6s 空洞, 又不至于像 2s 那样把旧值留太久。
                                #   0.5s 全局超时对它们不够: 停止/排队目标被槽位重排时 state 短暂
                                #   打 130, 雷达还会连续几帧把 b5 打成 30~31 (重捕获噪声, |yrel|>10
                                #   被丢) → 点从流里消失 0.6s+。实锤 .comma25 seg15 23:27 #27:
                                #   930.8 后 slot0 的车 b5 94→30, 点被当噪声丢, 0.5s 后删点。
                                #   代价方向不对称: 多留 1.5s 可能多一次轻刹, 少留就是跟假 lead 撞上去。
                                #   参考 ARS510 tracks.py: slot 静默 max_gap_s=0.3s 即判新目标。
                                #   实测宋Plus 同一 native_id 更新间隔 中位 0.070s / P99 0.149s,
                                #   >0.3s 仅 0.15% → 收紧到 0.5s 几乎不误伤正常目标。
                                #   原 2.0s 期间未匹配目标的 vRel 按 (1-dt) 衰减到 0, 被 radard 当
                                #   "同速前车"选中 → 幽灵前车/莫名减速窗口缩短 4 倍。
                                # 帧间关联: 旧 track 保留超时 (s)。实测活跃 slot 各以 14.89Hz
                                # 并发广播 (间隔中位 0.069s / P99 0.149s), 0.5s = 7 个刷新周期, 足够。
                                # 超过该时长未匹配 = 目标真正消失, 从聚类中心移除 →
                                # 触发 stale 清理删除绿框。曾缺失此过期: 幽灵 track 永驻
                                # _song_plus_tracks, pts 累积到 43, 绿框不消失且污染配对
                                # 实测 1.0s vs 2.0s 的 y_sane 一致(85%), 2.0s 主目标
                                # trackId 连续保持更长(152帧 vs 138帧) → 绿框跟随更稳
SONG_PLUS_NOT_SEEN_INIT = 20    # stale 清理: 目标从聚类中心移除后再过 ~0.4s 删绿框。
                                # 唐DM NOT_SEEN_INIT=99+隔帧递减需 ~4.6s, 宋Plus 14.5Hz 轮询
                                # 下目标消失信号足够快, 4.6s 会导致 pts 大量残留累积(实测32),
                                # 残留旧绿框位置过期 → 污染视觉配对(y_sane 17% vs 解码层80%)
# 【2026-11-05 状态码 130 提前收尾, 对标丰田 radar_interface】
#   丰田: NEW_TRACK / LONG_DIST>=255 时立即删点; 我们因 14.5Hz 轮询分片必须保留旧中心,
#   但 130 = "即将消失" (消失前 2 帧/0.15s 实证预告) —— 已宣告消失的目标没有继续保留 2s 的
#   必要, 保留期间 pts 会带衰减 vRel 持续发布 (vLead=vEgo 的幽灵前车 → 误跟车/无谓减速)。
#   只在 130 帧后缩短保留; 目标重新出现 (128/129) 会清掉标记回正常 2s。
SONG_PLUS_EXPIRING_HOLD = 0.35  # 状态码 130 目标: 失去轮询后的保留时长 (s)。覆盖 ~5 个轮询周期。

# ---------- 宋Plus vRel: 两点差分+低通 (2026-08-17 CV-KF 实验后回退) ----------
# 该雷达不广播每目标 vRel (2026-08-17 全总线+mux扫描证零), 只能由距离观测估计。
# 曾试 CV卡尔曼 (diag_c20): 动态滞后更优(0.28 vs 0.475s), 但跟停核心场景硬伤——
# 距离 0.0625m 量化 + 14.5Hz 分片轮播 → 量化噪声非高斯, KF 对 R 敏感:
# 静止跟停 vRel 偏差 +1.13m/s / std 0.59 (差分 +0.10 / 0.16, 差3.7倍)。
# 结论: 两点差分+低通 在宋Plus 0x380 雷达特性下更平滑准确, 维持差分法。

def _tang_build_slot_map(records) -> dict:
  """按 slot 分组各帧数据: slot_idx -> {sub: data}"""
  slot_map = {}
  for r in records:
    a, d = r[0], r[1]
    if len(d) < 4:
      continue
    if not (0x380 <= a < 0x400):
      continue
    slot_idx = (a - 0x380) // 4
    sub = (a - 0x380) % 4
    slot_map.setdefault(slot_idx, {})[sub] = d
  return slot_map

def _tang_remap_can(can_strings):
  """把 slot 基址帧(0x380+4i, sub=0)重映射为 ObjectData_i(383+i) 地址, 供 CANParser can_valid 判断"""
  if not can_strings or not isinstance(can_strings[0], tuple) or len(can_strings[0]) < 2:
    return [], 0
  records = can_strings[0][1]
  new_list = []
  count = 0
  for rec in records:
    if count >= MAX_OBJECTS:
      break
    if rec[0] not in TANG_SLOT_BASE_ADDRS:
      continue
    if len(rec[1]) < 4:
      continue
    new_list.append((count + 383, rec[1], rec[2]))
    count += 1
  return [(can_strings[0][0], new_list)], count

# -------------------------
#  简易输出一阶低通滤波器（用于 vRel 输出平滑，不反哺 KF）
# -------------------------
class LowPassFilter:
    def __init__(self, x0: float = 0.0):
        self.x = float(x0)

    def update(self, value: float, dt: float, tau: float):
        if dt <= 0:
            return self.x
        alpha = dt / (tau + dt)
        self.x = (1.0 - alpha) * self.x + alpha * float(value)
        return self.x

# --------------------------------------------------------------
#  简单的一维卡尔曼滤波（distance + relative speed）
#  支持可选的伪速度观测 v_meas，用马氏距离门控
# --------------------------------------------------------------
class SimpleKalmanFilter:
    """
    状态向量   x = [d, v_rel]^T
    观测模型   支持:
      - 仅距离观测 z = d + noise
      - 距离+速度观测 z = [d, v_rel]^T + noise
    过程噪声   加速度噪声 σ_a
    观测噪声   R (distance variance), Rv (velocity variance) 可选
    """

    def __init__(
        self,
        initial_d: float,
        initial_time: float,
        sigma_a: float = 0.15,
        R: float = 0.5,
        distance_scale: float = 1.0,
        distance_offset: float = 0.0,
    ):
        # calibration params (frozen; avoid aggressive online adaptation)
        self.distance_scale = distance_scale
        self.distance_offset = distance_offset

        # 状态
        self.d = float(initial_d)
        self.v_rel = 0.0
        self.last_v_rel = 0.0  # 用于加速度限幅

        # 初始协方差：让速度有较大不确定性
        self.P = np.array([[2.0, 0.0],
                           [0.0, 5.0]])

        # 噪声
        self.sigma_a = sigma_a
        self.R = float(R)  # distance variance

        # 计时
        self.last_time = float(initial_time)

        # 统计 / 健康检查
        self.invalid_count = 0
        self.measurement_history = deque(maxlen=5)

        # 最小/最大协方差下界（数值稳定）
        self._min_P = 1e-6

    # -------------------------
    # 辅助：预测步骤（匀速模型）
    # -------------------------
    def predict(self, dt: float) -> None:
        if dt <= 0:
            return

        # 状态预测
        self.d += self.v_rel * dt

        # 过程噪声 Q（标准离散化：右下为 dt**2）
        q = self.sigma_a ** 2
        Q = np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                      [dt ** 3 / 2.0, dt ** 2       ]]) * q

        F = np.array([[1.0, dt],
                      [0.0, 1.0]])
        self.P = F @ self.P @ F.T + Q
        # 数值下界
        self.P = np.maximum(self.P, self._min_P)

    # -------------------------
    # 更新步骤：支持可选速度观测 v_meas 与门控
    # -------------------------
    def update(self, d_meas: float, v_meas: Optional[float] = None, Rv: float = 1.0) -> Tuple[bool, float]:
        """
        返回 (was_updated, md2) 表示是否做了更新以及用于门控的马氏距离平方
        如果 v_meas is None -> 做 1D 更新（仅距离）
        否则做 2D 更新（距离+速度）并使用马氏门控（2 DOF）
        """
        # 基本合法性
        if not (0.5 <= d_meas <= 200.0):
            return False, float('inf')

        # 经验标定（静态）
        z_d = d_meas * self.distance_scale + self.distance_offset

        # 预测后的状态与 P 应在外部 predict() 已更新

        x_pred = np.array([self.d, self.v_rel])
        # 1D distance only
        if v_meas is None:
            H = np.array([[1.0, 0.0]])
            S = float(np.squeeze(np.asarray(H @ self.P @ H.T + self.R)))  # scalar
            y = np.array([z_d - (H @ x_pred)[0]])
            # 马氏距离 1D: y^2 / S
            md2 = float((float(y[0]) ** 2) / S)
            # 1 DOF chi2 thresholds: 25.0 ~ 99%
            if md2 > 25.0:
                self.last_md2 = md2
                return False, md2
            # 卡尔曼增益
            K = (self.P @ H.T) / S  # (2,1)
            state = x_pred + (K.flatten() * y[0])
            self.d, self.v_rel = float(state[0]), float(state[1])
            # Joseph form for P
            I = np.eye(2)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K * self.R * K.T
            self.P = np.maximum(self.P, self._min_P)
            self.invalid_count = 0
            self.last_md2 = md2
            return True, md2
        else:
            # 2D update
            z = np.array([z_d, float(v_meas)])
            H = np.eye(2)
            R_mat = np.diag([self.R, float(Rv)])
            S = H @ self.P @ H.T + R_mat
            y = z - x_pred
            # 马氏距离 y^T S^-1 y
            try:
                invS = np.linalg.inv(S)
                md2 = float(y.T @ invS @ y)
            except np.linalg.LinAlgError:
                md2 = float('inf')
            # 2 DOF chi2 threshold ~ 25.0 for 99%
            if md2 > 25.0:
                self.last_md2 = md2
                return False, md2
            K = self.P @ H.T @ np.linalg.inv(S)
            state = x_pred + K @ y
            self.d, self.v_rel = float(state[0]), float(state[1])
            # Joseph form
            I = np.eye(2)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_mat @ K.T
            self.P = np.maximum(self.P, self._min_P)
            self.invalid_count = 0
            self.last_md2 = md2
            return True, md2

    # -------------------------
    # 根据车辆速度简化调整噪声（保守，幅度较小）
    # -------------------------
    def adjust_noise_params(self, v_ego: float) -> None:
        # 基础映射（较保守）
        if v_ego > 25.0:
            base_sigma_a = 0.12
            base_R = 1.2
        elif v_ego > 15.0:
            base_sigma_a = 0.18
            base_R = 1.5
        elif v_ego > 5.0:
            base_sigma_a = 0.16
            base_R = 1
        else:
            base_sigma_a = 0.12
            base_R = 1.2

        # 简单赋值，不做激进自适应以免引入不稳定
        self.sigma_a = base_sigma_a
        self.R = base_R

    # -------------------------
    # 限制相对速度突变（平滑器内使用）
    # -------------------------
    def limit_acceleration(self, dt: float, max_accel: float = 2.0) -> None:
        if dt <= 0:
            return
        accel = (self.v_rel - self.last_v_rel) / dt
        if abs(accel) > max_accel:
            alpha = 0.7
            self.v_rel = self.last_v_rel * alpha + self.v_rel * (1 - alpha)
        self.last_v_rel = self.v_rel

# --------------------------------------------------------------
#  RadarInterface（集成伪速度观测、马氏门控、输出低通、per-target历史）
# --------------------------------------------------------------
class RadarInterface(RadarInterfaceBase):
    def __init__(self, CP, CP_SP):
        super().__init__(CP, CP_SP)

        # PT_RADAR_CAR (汉DM/EV、唐DM、宋Plus 21/22/23) 才有 MPC 雷达，其他车型不解析
        self.has_radar = CP.carFingerprint in PT_RADAR_CAR
        if not self.has_radar:
            self.rcp = None
            return

        # MRR 直解车型 (唐DM/汉DM/汉EV/宋Plus 21/22): 雷达实测地址 0x380~0x3FF
        # (32 slot x 4 sub, CAN Bus=1), 手动字节解码, 唐DM/汉用 ARS4xx, 宋Plus 实测是博世 MRR evo 14 (2026-08-13)
        #   slot i 四帧: base(0x380+4i)=dRel / +1=yRel / +2=vRel / +3=空闲。
        #   ObjectData_0~9(383~392) 为重映射地址, 仅供 CANParser can_valid 判断。
        # 非直解车型(宋Plus 23款)用单报文 RADAR_MRR(0x374), 原车发 bus2(MPC) (2026-08-18 H宋23日志实测: src=2 直收),
        # panda fwd_hook 转发到 bus0(ESC)。radar_interface 走 bus=MPC(2) 读 src=2。
        self.is_mrr_direct = CP.carFingerprint in MRR_DIRECT_DECODE_CAR
        # 0x380 直解路径自带点级 aLead (同目标距离窗口拟合, 见 SONG_PLUS_ALEAD_* / _update_song_plus),
        # 置 True → 基类 update_carrot 不再用 MyTrack 的 ΔvLead/Δt 覆盖它。
        self.provides_alead = CP.carFingerprint in SONG_PLUS_380_DECODE_CAR
        self.tang_track_msgs = [f"ObjectData_{i}" for i in range(10)]
        if self.is_mrr_direct:
            messages = [(m, float('nan')) for m in self.tang_track_msgs]
        else:
            messages = [
                ("RADAR_MRR", 60),  # 雷达消息，60Hz频率
            ]
        # 宋Plus 23款: 0x374 雷达无 IsValid 位(byte5 恒 0xFC), 有效判定 = TargetID==2 且 LongDist 合理
        self.radar_no_valid = CP.carFingerprint in RADAR_MRR_CAR
        try:
            cloudlog.info(f"BYD radar CANParser init: dbc={DBC[CP.carFingerprint][Bus.pt]} messages={messages} "
                          f"bus={CanBus.MRR if self.is_mrr_direct else CanBus.MPC}")
            self.rcp = CANParser(DBC[CP.carFingerprint][Bus.pt], messages, CanBus.MRR if self.is_mrr_direct else CanBus.MPC)
            cloudlog.info(f"BYD radar CANParser ready: {type(self.rcp)}")
        except Exception as e:
            cloudlog.warning(f"BYD radar CANParser init failed: {e}")
            self.rcp = None
        # 触发帧: 非直解车型用原车雷达时钟报文 0x374; MRR 直解按 slot 基址直接解码, 无触发帧依赖。
        self.trigger_msg = 0x374

        # ---------- 数据结构 ----------
        self.car_fingerprint = CP.carFingerprint
        self.pts: Dict[int, structs.RadarData.RadarPoint] = {}
        self.kalman_filters: Dict[int, SimpleKalmanFilter] = {}

        # 宋Plus 0x380 博世 MRRevo14: 距离历史 (用于两点差分 vRel; 含 yRel 供弯道抑制)
        self._song_plus_hist: Dict[int, deque] = {}  # track_id -> deque[(monotonic, dRel, yRel)]
        self._song_plus_last_t: Dict[int, float] = {}  # track_id -> 上次低通更新时间
        # b6 精细接近率缓存 (2026-09-20): slot g -> (b6, monotonic, 质量位) 最近一次 real 速度帧
        #   质量位 = 0x3C8+g.b7 bit0 (10-03: bit0=1 时 b6 误差翻倍 → 使用端退差分)
        self._song_plus_b6_cache: Dict[int, tuple] = {}
        # 【10-03】粗接近率 b2 缓存已删除 (b2 证伪: per-slot tracker 状态量, 换目标串位)
        # 目标关联: 帧间最近邻 (借鉴 Ford MRR cluster_points)。
        # 宋Plus 的 slot/tid 是排序容器非目标身份, 用上一帧 track 中心匹配本帧点。
        # 2026-08-15 速度预测: 存 (dRel, yRel, vRel, t), 匹配时用 vRel 预测当前位置。
        # 2026-08-20 FIX1: 加 slot 字段 (同槽优先关联防目标分裂)。
        self._song_plus_tracks: Dict[int, tuple] = {}  # track_id -> (dRel, yRel, vRel, t, slot) 上次状态
        self._song_plus_next_id = 1  # 自增 trackId (从1开始, 避免 carrot.cc isLeadSCC 误判 trackId<1)
        self._song_plus_native_id = {}  # track_id -> 雷达原生航迹 ID (sub3.b7&0x3F), 用于确定性换目标检测

        # 唐DM: V11 直解数据结构 (trackId = slot 编号)
        self._pts_cache: Dict[int, structs.RadarData.RadarPoint] = {}
        self._pts_not_seen: Dict[int, int] = {k: 0 for k in range(255)}

        # 宋23 (0x374 MUX 多目标): 目标跨消息累积缓存 (轮询广播, 每条消息仅1~4个ChildID)
        # child_id -> (last_t, dRel, yRel, vRel)
        self._mrr23_tracks: Dict[int, tuple] = {}
        self._mrr23_track_timeout = 0.5   # 目标超时 (s): 轮询周期~0.1s(100Hz), 超时未更新则移除

        # per-target helpers
        self.dist_hist: Dict[int, deque] = {}       # (t, d) deque for pseudo velocity
        self._dist_bufs: Dict[int, deque] = {}      # per-target median buffer for preprocess
        self.last_valid_long: Dict[int, float] = {}
        self.vrel_filters: Dict[int, LowPassFilter] = {}
        self._song_plus_vrel_out: Dict[int, float] = {}   # 输出 vRel 帧间限变锚点 (2026-11-04)
        self._song_plus_birth_cnt: Dict[int, int] = {}    # 出生确认计数 (2026-11-10, SONG_PLUS_BIRTH_CONFIRM)
        self._song_plus_vrel_out_t: Dict[int, float] = {}  # vRel 斜率上限的时间锚 (2026-11-09)
        self._song_plus_dout: Dict[int, tuple] = {}        # dRel 发布包络锚 (d, y, t) (2026-11-09)
        self._song_plus_birth: Dict[int, float] = {}      # track 创建时刻 (2026-11-04 揭示期判龄)
        self._song_plus_expiring: Dict[int, float] = {}   # track_id -> 最后见到状态码 130 的时刻

        # ---------- 统计 ----------
        self.updated_messages = set()

        # ---------- 日志 ----------
        # 文件日志非必需：/data/debug 目录可能无写权限（comma 用户），若 mkdir 抛异常会把 card 进程搞挂。
        # 故改为可写才启用、失败则静默禁用（唐DM 直解分支不依赖此日志）。
        # 写盘走异步 writer（后台线程批量刷盘），杜绝 100Hz 控制循环被磁盘抖动拖住。
        self.log_path = "/data/debug/byd_radar_complete_data.log"
        self._log_enabled = True
        self._log_writer = None
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a"):
                pass
            self._log_writer = get_writer(self.log_path)
        except Exception as e:
            self._log_enabled = False
            cloudlog.warning(f"[RadarInterface] Radar log disabled (no write permission): {e}")

        # ---------- 车速 (card 每帧注入) ----------
        # 默认 0: 首帧不因缺属性抛 AttributeError (2026-11-01 审核)
        self.v_ego = 0.0

        # ---------- 参数 ----------
        self.max_distance = 200.0
        self.min_distance = 0.5
        self.max_lat_offset = 4.0
        self.max_speed_change = 30.0
        self.max_dist_jump = 10.0
        self.filter_invalid_frames = 6     # 保守一点
        self.filter_delete_timeout = 1.5   # 秒

    # -----------------------------------------------------------------
    #  前处理：per-target 中值 + EMA
    # -----------------------------------------------------------------
    def preprocess_radar_data(self, target_id: int, raw_dist: float, msg_mrr: dict) -> Optional[float]:
        # per-target median buffer
        buf = self._dist_bufs.setdefault(target_id, deque(maxlen=5))
        buf.append(raw_dist)
        sorted_buf = sorted(list(buf))
        median = sorted_buf[len(sorted_buf) // 2]

        # per-target EMA state stored in attribute
        ema_attr = f"_dist_ema_{target_id}"
        if not hasattr(self, ema_attr):
            setattr(self, ema_attr, float(median))
            return float(median)
        else:
            prev = getattr(self, ema_attr)
            change = abs(median - prev)
            alpha = 0.4 if change < 0.5 else 0.2
            new = alpha * median + (1 - alpha) * prev
            setattr(self, ema_attr, float(new))
            return float(new)

    # -----------------------------------------------------------------
    #  生成伪速度（最小二乘线性拟合）
    # -----------------------------------------------------------------
    def compute_pseudo_velocity(self, target_id: int, current_ts: float, d_meas: float) -> Optional[float]:
        h = self.dist_hist.setdefault(target_id, deque(maxlen=6))
        h.append((current_ts, float(d_meas)))
        if len(h) < 3:
            return None
        ts = np.array([x[0] for x in h])
        ds = np.array([x[1] for x in h])
        t0 = ts[0]
        A = np.vstack([ts - t0, np.ones_like(ts)]).T
        try:
            slope, _ = np.linalg.lstsq(A, ds, rcond=None)[0]
            # slope = d(distance)/dt, which equals lead_speed - ego_speed (i.e. relative speed)
            return float(slope)
        except Exception:
            return None

    # -----------------------------------------------------------------
    #  主循环（每帧 CAN 数据）
    # -----------------------------------------------------------------
    def update(self, can_strings):
        if self.rcp is None:
            return super().update(None)

        # 【2026-11-10 时基统一】优先日志时间 (回放可复现), 实车 fallback 墙钟
        try:
            current_ts = float(can_strings[0][0]) / 1e9
        except (TypeError, ValueError, IndexError):
            current_ts = time.monotonic()

        values = self.rcp.update(can_strings)
        if self.is_mrr_direct:
            # MRR 直解: 0x380~0x3FF slot 帧 (V11 实测解码), 不依赖触发帧。
            return self._update_mrr_direct(can_strings)

        # 宋Plus 23款 (指纹=23): 0x374 RADAR_MRR 单目标直解 (不走 KF, 1m 量化距离短窗口差分噪声大)
        if self.radar_no_valid:
            self.frame += 1
            self._pts_cache.clear()
            self._decode_radar_mrr_from_can(can_strings)
            # 清理过期目标 (get 兜底: 新 trackId 可能未在 _pts_not_seen 中初始化)
            if self.frame % 2 == 0:
                stale = [k for k in self.pts if k not in self._pts_cache]
                for k in stale:
                    self._pts_not_seen[k] = self._pts_not_seen.get(k, 0) - 1
                    if self._pts_not_seen[k] <= 0:
                        del self.pts[k]
                        self._song_plus_hist.pop(k, None)
                        self._pts_not_seen.pop(k, None)
            self.pts.update(self._pts_cache)
            if not self.rcp.can_valid:
                return None
            ret = structs.RadarData()
            ret.errors.canError = False
            ret.points = list(self.pts.values())
            return ret

        # ============================================================
        # 【2026-10-22 审计 L1】以下到本函数末尾 (trigger_msg/KF/preprocess/
        #   compute_pseudo_velocity 路径) 对现有全部车型指纹 **不可达**:
        #   values.py: PT_RADAR_CAR = {汉DM,汉EV,唐DM,宋21,宋22,宋23}
        #              MRR_DIRECT_DECODE_CAR = {唐DM,汉DM,汉EV,宋21,宋22}
        #              RADAR_MRR_CAR = {宋23}   —— 三集合精确覆盖 PT_RADAR_CAR
        #   → is_mrr_direct 或 radar_no_valid 必为 True, 上面两条早返回已拦截。
        #   保留代码以防将来接非直解车型; 勿在此添加新逻辑 (跑了也没用)。
        #   附加: 此段的 KF 清理条件 age>1.5s 恒假 (kf.last_time 每帧刷新)。
        #   ⚠️【2026-11-13 定稿】**这是死代码, 不要当活实现维护或调参**:
        #     · 单测/回放都不会进入这里, 任何改动都无法通过数据验证;
        #     · 将来真接非直推车型, 应按新车型报文重写, 而不是复用这里的 KF/触发帧逻辑;
        #     · 本文件其余部分 (0x380 直解) 与这段没有任何共享状态, 删改互不影响。
        # ============================================================
        if self.trigger_msg not in values:
            # no radar in this frame
            return None

        msg_mrr = self.rcp.vl['RADAR_MRR']
        target_id = int(msg_mrr['TargetID'])  # 1:left, 2:front, 3:right

        # 仅处理前向目标
        if target_id != 2:
            return None

        # 创建/获取 RadarPoint
        if target_id not in self.pts:
            self.pts[target_id] = structs.RadarData.RadarPoint()
        rp = self.pts[target_id]
        rp.trackId = target_id

        # 原始测量
        raw_long = float(msg_mrr['LongDist'])
        raw_lat = float(msg_mrr['LatDist'])
        # 宋Plus 23款 (RADAR_MRR_CAR): 0x374 报文无 IsValid 位, 有效性靠距离范围判断;
        # 其他车型(如有)仍用 IsValid 位。
        is_valid = raw_long > 0 or (bool(msg_mrr['IsValid']) and not self.radar_no_valid)

        # 基本合法性检查
        if not (self.min_distance <= raw_long <= self.max_distance):
            is_valid = False
        if abs(raw_lat) > self.max_lat_offset:
            is_valid = False
        if not self.radar_no_valid and not bool(msg_mrr['IsValid']):
            is_valid = False

        # 连续性检查（per-target last_valid_long）
        if target_id in self.last_valid_long:
            if is_valid:
                jump = abs(raw_long - self.last_valid_long[target_id])
                if jump > self.max_dist_jump:
                    is_valid = False
                else:
                    self.last_valid_long[target_id] = raw_long
        else:
            if is_valid:
                self.last_valid_long[target_id] = raw_long

        # 预处理
        if is_valid:
            processed_long = self.preprocess_radar_data(target_id, raw_long, msg_mrr)
            if processed_long is None:
                is_valid = False
            else:
                raw_long = processed_long

        # Default dt_for_log
        dt_for_log = 0.0

        # 卡尔曼滤波器管理
        if is_valid:
            if target_id not in self.kalman_filters:
                kf = SimpleKalmanFilter(
                    initial_d=raw_long,
                    initial_time=current_ts,
                    sigma_a=0.15,
                    R=0.5,
                    distance_scale=1.0,
                    distance_offset=0.0,
                )
                self.kalman_filters[target_id] = kf
                # create output lowpass
                self.vrel_filters[target_id] = LowPassFilter(0.0)
                # init dist_hist
                self.dist_hist.setdefault(target_id, deque(maxlen=6))
            else:
                kf = self.kalman_filters[target_id]

            # adjust noise conservatively
            kf.adjust_noise_params(self.v_ego)

            # dt compute & clip
            dt = float(current_ts - kf.last_time)
            if dt <= 0 or dt > 0.5:
                dt = max(0.01, min(dt, 0.2))
            dt_for_log = dt  # save for logging before we overwrite last_time

            # predict
            kf.predict(dt)

            # compute pseudo velocity from recent preprocessed distances
            v_meas = self.compute_pseudo_velocity(target_id, current_ts, raw_long)

            # choose Rv (velocity observation variance) conservatively
            Rv = 1.0  # (m/s)^2 baseline; can be tuned
            if v_meas is not None:
                hist_len = len(self.dist_hist.get(target_id, []))
                if hist_len >= 5:
                    Rv = 0.8 ** 2
                elif hist_len >= 3:
                    Rv = 1.0 ** 2
                else:
                    Rv = 1.5 ** 2

            # update with gating
            updated, md2 = kf.update(raw_long, v_meas, Rv) if v_meas is not None else kf.update(raw_long, None)

            # 在update调用后添加重置机制
            if not updated:
                kf.reject_count = getattr(kf, 'reject_count', 0) + 1
                if kf.reject_count > 5:  # 连续5帧被拒绝就重置
                    kf.d = raw_long
                    kf.v_rel = 0.0
                    kf.P = np.array([[2.0, 0.0], [0.0, 5.0]])
                    kf.reject_count = 0
                    cloudlog.debug(f"[RadarInterface] Reset KF for target {target_id} due to consecutive rejections")
            else:
                kf.reject_count = 0

            # acceleration / velocity limiting
            max_accel = 3.0 if self.v_ego > 15 else 2.0
            kf.limit_acceleration(dt, max_accel=max_accel)

            # set last_time to now (we did a predict and maybe update)
            kf.last_time = current_ts

            # output smoothing (per-target low-pass)
            v_filter = self.vrel_filters.setdefault(target_id, LowPassFilter(kf.v_rel))
            # 【2026-11-06 tau 下调】11 位速度字段 (MAE 0.598, 噪声 0.085 m/s 且不随距离恶化)
            #   到位后, 强滤波不再必要。离线标定 (8 轨迹, 噪声×滞后): tau=0.20 最优
            #   (0.10→7.74 / 0.15→6.06 / 0.20→5.01 / 0.30→7.30, 0.30 后滞后翻倍至 139ms)。
            #   回放 (段9+13): 硬刹 89→73 (-18%), jerk 1.015→0.936, 微振荡 5.4 不变。
            #   滞后 70ms vs 原 18.6m/s 时 139ms → 规划器提前响应, 反而减少事后猛刹。
            tau = 0.20
            v_smoothed = v_filter.update(kf.v_rel, dt, tau)

            # fill RadarPoint - ensure consistency: use same (smoothed) vRel for both vRel and vLead
            rp.dRel = float(kf.d)
            rp.vRel = float(v_smoothed)
            rp.yRel = raw_lat
            pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # yvRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # measured: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)

            # reset invalid counter on successful read
            if updated:
                kf.invalid_count = 0

        else:
            # 无效测量 -> 如果已有滤波器则仅做 predict (不把无效测量更新进来)，并输出预测值
            if target_id in self.kalman_filters:
                kf = self.kalman_filters[target_id]
                dt = float(current_ts - kf.last_time)
                dt = max(0.01, min(dt if dt > 0 else 0.01, 0.2))
                dt_for_log = dt
                kf.predict(dt)
                kf.limit_acceleration(dt, max_accel=3.0 if self.v_ego > 15 else 2.0)
                kf.last_time = current_ts
                kf.invalid_count += 1

                # output smoothed prediction
                v_filter = self.vrel_filters.setdefault(target_id, LowPassFilter(kf.v_rel))
                if self.v_ego < 10.0:
                    tau = 0.6
                elif self.v_ego < 20.0:
                    tau = 0.35
                else:
                    tau = 0.25
                v_smoothed = v_filter.update(kf.v_rel, dt, tau)

                rp.dRel = float(kf.d)
                rp.vRel = float(v_smoothed)
                rp.yRel = raw_lat
                pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                pass  # yvRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                pass  # measured: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            else:
                # 没有历史滤波器，保留默认或先不填
                pass  # measured: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                try:
                    rp.dRel = float(raw_long) if raw_long > 0 else 199.0
                except Exception:
                    rp.dRel = 199.0
                rp.vRel = 0.0
                pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                rp.yRel = raw_lat

        # 速度突变保护（对输出值再次防护）
        if hasattr(rp, 'vRel') and rp.vRel is not None:
            if abs(rp.vRel) > self.max_speed_change:
                if target_id in self.kalman_filters:
                    rp.vRel = float(self.kalman_filters[target_id].last_v_rel)
                    # ensure vLead consistent too
                    pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                else:
                    rp.vRel = 0.0
                    pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)

        # 日志：记录更多诊断信息（使用 dt_for_log 保存的值）
        # 异步写盘：put_nowait 微秒级返回，磁盘抖动只丢日志、不拖慢 card。
        if self._log_enabled:
            try:
                kf = self.kalman_filters.get(target_id, None)
                kf_dist = kf.d if kf is not None else 199.0
                kf_vrel = kf.v_rel if kf is not None else 0.0
                dist_diff = abs(raw_long - rp.dRel) if rp.dRel < 199 else 0.0
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                v_ego_kmh = self.v_ego * 3.6
                kf_vrel_kmh = kf_vrel * 3.6
                final_vrel_kmh = rp.vRel * 3.6 if rp.vRel is not None else 0.0
                md2_info = getattr(kf, 'last_md2', 0.0) if kf is not None else 0.0
                self._log_writer.write(
                    f"{timestamp}, dt={dt_for_log:.3f}, "
                    f"raw: dist={raw_long:.2f}, lat={raw_lat:.2f}, valid={is_valid}, "
                    f"kf: dist={kf_dist:.2f}, kf_vrel_raw={kf_vrel_kmh:.1f}km/h, "
                    f"final: dist={rp.dRel:.2f}, vrel={final_vrel_kmh:.1f}km/h, "
                    f"vrel={(rp.vRel)*3.6:.1f}km/h, diff={dist_diff:.2f}, "
                    f"md2={md2_info:.2f}, v_ego={v_ego_kmh:.1f}km/h\n"
                )
            except Exception as e:
                self._log_enabled = False  # 写失败后禁用，避免每帧重复报错
                cloudlog.warning(f"[RadarInterface] log error (disabled): {e}")

        # 清理：删除长时间未更新的滤波器（按 invalid_count 和时间共同判断）
        to_del = []
        for tid, kf in list(self.kalman_filters.items()):
            age = current_ts - kf.last_time
            if kf.invalid_count >= self.filter_invalid_frames and age > self.filter_delete_timeout:
                to_del.append(tid)
        for tid in to_del:
            self.kalman_filters.pop(tid, None)
            self.pts.pop(tid, None)
            self.dist_hist.pop(tid, None)
            self._dist_bufs.pop(tid, None)
            self.last_valid_long.pop(tid, None)
            self.vrel_filters.pop(tid, None)

        radar_data = structs.RadarData()
        # 融合层契约: card 用 errors.to_dict() 判 valid, radard 用 radarErrors 上报;
        # 此路径对现有指纹不可达(见上方审计注释), 补上以保持四路径一致。
        radar_data.errors.canError = not self.rcp.can_valid
        radar_data.points = list(self.pts.values())
        return radar_data

    # -----------------------------------------------------------------
    #  MRR 直解 (V11): 0x380~0x3FF, 32 slot x 4 sub (唐DM/汉DM/汉EV/宋Plus 21/22/23 全系通用)
    # -----------------------------------------------------------------
    def _update_mrr_direct(self, can_strings):
        self.frame += 1
        # CANParser（重映射 slot base -> ObjectData_i, 仅用于 can_valid 判断）
        parsable, size = _tang_remap_can(can_strings)
        if size > 0:
            self.rcp.update(parsable)

        if not can_strings or not isinstance(can_strings[0], tuple) or len(can_strings[0]) < 2:
            return None

        # 宋Plus 21/22/23: 走 0x380 博世 MRRevo14 多目标解码 (跳过唐DM V11 布局, 布局不同)
        if self.car_fingerprint in SONG_PLUS_380_DECODE_CAR:
            return self._update_song_plus(can_strings)

        records = can_strings[0][1]
        slot_map = _tang_build_slot_map(records)

        if not slot_map:
            return None  # 无 0x380~0x3FF 雷达帧, 不产出 (同 _update_song_plus 契约)
        self._pts_cache.clear()
        for sidx in sorted(slot_map.keys()):
            sm = slot_map[sidx]

            # dRel (sub=0)
            base_dat = sm.get(0)
            if base_dat is None:
                continue
            d = _tang_decode_drel(base_dat)
            if d is None or d < 3.0:
                continue

            # vRel (sub=2)
            vrel_val = _tang_decode_vrel(sm.get(2, bytes(4)))
            # yRel (sub=1)
            yrel_val = _tang_decode_yrel(sm.get(1, bytes(4)))

            # trackId = slot 编号 (固定, 跨帧不变)
            track_id = sidx
            if track_id not in self._pts_cache:
                self._pts_cache[track_id] = structs.RadarData.RadarPoint()
                self._pts_cache[track_id].trackId = track_id

            self._pts_not_seen[track_id] = SONG_PLUS_NOT_SEEN_INIT
            pt = self._pts_cache[track_id]
            pt.dRel = d
            pt.yRel = yrel_val
            pt.vRel = vrel_val
            pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # yvRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # measured: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)

        # 清理过期目标
        if self.frame % 2 == 0:
            stale = [k for k in self.pts if k not in self._pts_cache]
            for k in stale:
                self._pts_not_seen[k] -= 1
                if self._pts_not_seen[k] <= 0:
                    del self.pts[k]

        self.pts.update(self._pts_cache)

        # 契约 (官方 test_car_interfaces): CAN 无效时不产出 RadarData——
        # radard 无新帧 → tracks 自然老化; 若照发空/陈旧点会被当有效雷达目标。
        if not self.rcp.can_valid:
            return None
        ret = structs.RadarData()
        ret.errors.canError = False
        ret.points = list(self.pts.values())
        return ret

    # -----------------------------------------------------------------
    #  宋Plus: 0x380 博世 MRRevo14 多目标解码 (2026-08-13/15/16 破解落地)
    #  距离: 0x0625m/LSB, d = 0.0625 * ((b3&0x0F)<<8 | b2), b3高4位=距离排序号(非目标ID)
    #  横向: b5 = 每目标方位角 (2026-08-14 定论), yRel = 0.014*d*(b5-95.0), 有效 b5∈[10,124]
    #        (offset 2026-08-15 从92.6修正到95.0, 视觉标定多日志一致)
    #  结构: 0x380~0x3FF 每4地址一组=一个目标(base 0x380+4i 含距离/方位,
    #    +1=sub1时间戳, +2=sub2内部状态, +3=sub3状态码); 最多32目标组并行广播。
    #    slot/tid 是按距离排序的"传输容器"非目标身份 (前车接近时 tid 从 8→1 变化)
    #  目标关联: 帧间最近邻 (借鉴 Ford MRR cluster_points), 按 (dRel,yRel) 匹配上一帧
    #    track 中心 → 绿框跟随目标; 匹配不上=新目标分配新 trackId
    #  有效标志: sub3 b5 状态码 (128=新/过渡, 129=稳定跟踪, 130=即将消失, 131+排除)
    #  速度: 距离差分 (0x380 无双向 vRel 广播; 差分法唯一双向来源)
    #  比 0x0109(0.51m/bit 单目标) 精度高 8 倍 + 多目标
    # -----------------------------------------------------------------
    def _update_song_plus(self, can_strings):
        records = can_strings[0][1]
        # 契约: 本帧没有任何 0x380~0x3FF 雷达帧 → 不产出 RadarData (官方测试要求;
        # 生产语义: 无雷达帧 = 无新目标, radard 侧 tracks 自然老化)。
        if not any(0x380 <= _r[0] <= 0x3FF and len(_r[1]) >= 8 for _r in records):
            return None
        # 【2026-11-10 时基统一】内部时间全部用日志时间 (logMonoTime):
        #   回放/离线全速跑时墙钟间隔≈0 → 差分/长窗/关联预测全部失效 (回放与实车行为
        #   不一致的根因)。logMonoTime 与 monotonic 在实车差固定偏移 → 窗口差值等价,
        #   实车行为不变; 回放则可复现。
        try:
          _log_t = float(can_strings[0][0]) / 1e9
        except (TypeError, ValueError, IndexError):
          _log_t = time.monotonic()
        now = _log_t
        # 【2026-11-09】发布钳制的时基必须用日志时间而非墙钟: 回放/离线全速跑时
        #   墙钟间隔≈0, 包络 dt 被 clamp 到下限 → 真实高速接近被人为卡住。
        #   实车日志时间≈墙钟, 行为不变。
        try:
            _log_t = float(can_strings[0][0]) / 1e9
        except (TypeError, ValueError, IndexError):
            _log_t = now
        # (2026-11-13 删除) 此处原有 `v_ego = self.v_ego ...`, 赋值后从未使用。
        self._pts_cache.clear()
        # 阶段1: 按地址收集帧 (base + sub3)
        frame_map = {}  # addr -> dat
        for r in records:
            a, d = r[0], r[1]
            if 0x380 <= a <= 0x3FF and len(d) >= 8:
                frame_map[a] = d
                # 速度槽 0x3C8+g (g=0~3): b6 = 目标槽 g 精细接近率 (2026-09-20 破解)。
                # real 帧判据 (2026-09-20 夜审计, 12 段 3.3 万帧实测):
                #   fill  = d≈127.875 & tid15  → 排除 (⚠️ 其 b6 并不恒 -6: 87% 是 0, 13% 是 -6,
                #          绝不能按 b6 值识别, 否则 b6=0 会换算成 -7d/C 的假接近率污染缓存);
                #   b2=0 帧 → 排除; tid==15 → 排除 (模糊残帧组 854 帧 b6 中位 -6, 残帧画像);
                #   real = 其余 (实测 tid 全部 0~14, 与目标 tid 语义一致)。
                # b7 位语义 (2026-10-03 全量实测**推翻** 09-23 的"符号位副本"说, 两车队
                #   + 本次 route29/route6a 共 2 万+ real 帧):
                #   只有 bit0 有值 (占 real 帧 8.4%), 且 bit0=1 时 b6 误差翻倍
                #   (MAE 0.553 → 1.071 m/s) = **真质量位** → 缓存带标记, 使用端退差分。
                #   bit1~7 恒 0 (09-23 所称"b7==1 ⟺ b6>=0, 215951 帧仅 1 例外"在本数据上
                #   不成立: 那批统计把整个 b7 字节当单一布尔, 实为 bit0 的误读)。
                #   ⚠️ 若日后有人想"撤销质量门", 先看今天的数据: 强接近帧 (|真接近率|>3)
                #   bit0=1 占 30.7% —— 门只丢 8.4% 的帧, 而这些帧的 b6 误差翻倍。
                #   值域保护 |b6|<=50 保留 (垃圾帧 0.002%); 0.7 强接近保守偏置不变。
                if 0x3C8 <= a <= 0x3CB:
                    _b6r = d[6]
                    _b6s = _b6r - 256 if _b6r >= 128 else _b6r
                    _sd = 0.0625 * (((d[3] & 0x0F) << 8) | d[2])
                    _is_fill = abs(_sd - 127.875) < 0.01 and (d[3] >> 4) == 15
                    # 【2026-11-13 删除 _b6r<249 门】速度槽 b6 **不是速度** (旧注释"粗接近率"
                    #   与"填充簇 249~253 残留"均错): 4 段 7976 条 REAL 记录实测 b6≥247 占
                    #   90.0%、≤1 仅 3.2%; FILL 记录 b6=0 占 95.8% —— 方向与旧注释正好相反。
                    #   且 2026-11-06 起缓存已存 11 位解码速度 (_v11, 不读 b6), 这道门是拿
                    #   质量字段 (b6⇔b7.bit0, corr -0.998) 当速度筛 → 只保留 34.5% 真实速度
                    #   记录, 强接近 (v<-3) 811 条丢 618 条 = 76% (危险方向=欠刹)。
                    #   铁律 5: 字节语义纠正后, 基于它的过滤器必须重验 → 删除此门。
                    #   _is_fill / d[2]!=0 / tid!=15 / |b6s|<=50 保留 (填充哨兵与值域保护)。
                    if (not _is_fill) and d[2] != 0 and (d[3] >> 4) != 15 and -50 <= _b6s <= 50:
                        # b7 bit0 = 质量位 (10-03 实测: bit0=1 时 b6 误差翻倍 0.553→1.071 m/s,
                        # 占 real 帧 8.4%; bit1~7 恒 0) → 缓存里带质量标记, 使用端退差分。
                        # 【2026-11-05 补测】b7 是整帧测量质量位: 与距离抖动相关 +0.40 (四分带
                        #   0.44/0.41/0.39/0.33 一致), 不止 b6。但它**不进 tau** —— 离线 A/B
                        #   (249 段/184 有效, 逐变体只改 tau): qd 基线 median 0.4885, 加 b7×1.2/1.4/1.6
                        #   后 0.4940/0.4984/0.5020, P90/稳态/制动 bias 同向劣化 —— 滞后代价 > 降噪收益,
                        #   与 sub3.b4.bit7 同结论。b7 的正确用法就是现在这个: 只用来退 b6 差分。
                        # 【2026-11-06】存解码后的 11 位速度 (取代原始 b6, 见常量区说明)
                        _v11 = SONG_PLUS_VREL_V11_SCALE * (int(d[2]) | ((int(d[3]) & 0x07) << 8)) - SONG_PLUS_VREL_V11_ZERO
                        self._song_plus_b6_cache[a - 0x3C8] = (_v11, now, bool(d[7] & 1))
                # sub2 帧 (0x382+4g) 的 b2 粗接近率: **10-03 已证伪, 缓存不再写入**
                # (b2 = 雷达 per-slot tracker 状态量, 换目标后串位; 见 SONG_PLUS_VREL_B2_* 注释区)。

        # 阶段2: 解码所有有效点 (不再按 slot 当目标身份)
        # 宋Plus 0x380 的 slot/tid 是"按距离排序的传输容器"(前车 67m→21m 平滑接近时
        # tid 从 8→1 不断变化), 不能作目标身份。有效标志用 sub3 b5 状态码
        # (128=新/过渡, 129=稳定跟踪, 130=即将消失; 131+ 排除)。
        points = []  # (dist, yrel, state, slot, dist_raw)
        for slot in range(32):
            base_addr = 0x380 + slot * 4
            if base_addr not in frame_map:
                continue
            # 轮询槽排除: slot18/19 (0x3C8/0x3CC) 每帧换 tid 轮询报告多个远距目标
            if slot in (18, 19):
                continue
            d = frame_map[base_addr]
            tid = d[3] >> 4
            # 空槽判据用精确字节模板 (2026-10-21 修订)。
            # ⚠️ 旧代码用 tid==15 判空是错的: b3 高 4 位是"距离档索引"(0≈10m … 15≈118m),
            # 空闲模板 ff ff ff 3f ff ff 的 b3=0xFF 恰好解出 tid=15, 与最远档重合 →
            # 118m 附近的真实目标(1.5% 有效帧)被误杀。实测空闲/占位两种模板各有精确字节。
            if d[2:8] in (SONG_PLUS_IDLE_TMPL, SONG_PLUS_FILL_TMPL):
                continue
            # 注意: tid=0 是"最近目标"(近距跟车/低速, dist 5-10m) 真实目标!
            # 2026-08-15 修正: 原代码把 tid=0 当无效过滤, 导致近距(<10m)目标全丢
            # (低速时前车 6.6→5.8m 持续跟踪 tid=0, b5 有效 100%, b7=7 占57%)
            v = ((d[3] & 0x0F) << 8) | d[2]
            dist_raw = 0.0625 * v
            if not (2.0 <= dist_raw <= SONG_PLUS_DREL_MAX) or abs(dist_raw - 127.9) < 0.1:
                continue  # 127.9m=0x7FF 空闲标记
            b5 = d[5]
            if not (SONG_PLUS_YREL_B5_MIN <= b5 <= SONG_PLUS_YREL_B5_MAX):
                continue  # b5 无效: 远距默认值(192)或噪声
            # 状态码 (sub3 b5); 缺 sub3 帧时保守默认 129 (有效, 不丢目标)
            state = 129
            sub3 = frame_map.get(base_addr + 3)
            if sub3 is not None:
                state = sub3[5]
            if state not in SONG_PLUS_STATE_VALID:
                continue  # 131+ = 边界/噪声目标, 排除
            # 【2026-10-22 新】yRel 改用 11 位精细方位字段 (bit144|11, 进位链切分所得)。
            #   旧: yrel = 0.014 * dist_raw * (b5 - 95)   ← 8 位粗方位, 0.802°/LSB
            #   新: yrel = 0.00042 * dist_raw * (code144 - 1044)  ← 11 位精细方位, 0.024°/LSB (33x 细)
            # 刻度/零点用 b5 交叉校验定死 (28,234 样本网格搜最优):
            #   |y_new − y_old| 中位 0.324m, 差值中位 +0.056m。
            # 收益: 长轨迹(跨度>15m)内 y 抖动 std 2.72 → 1.39 (降 48%), 71% 轨迹更稳。
            # 同族三个副本 bit19|11 / bit144|11 / bit155|11 互相关 0.99, 取 144|11。
            # 【三层兜底 (2026-10-22 上车前审计修复)】精细值必须通过校验才用, 否则退 b5 粗值:
            #   ① sub3 帧缺失 (CAN 丢帧时该槽 sub0 到了但 sub3 没到) —— 旧代码对此
            #      明确设计"默认 state=129 不丢目标", 若 _c144 取默认 0 会让
            #      yrel = 0.00042*dist*(0-1044) = -0.438*dist, 22.8m 以上目标被
            #      |yrel|>10 全部误杀 → 严重回归。
            #   ② c144 越界 (实测有效 288~2047; 0/2047 是填充哨兵)。
            #   ③ 与 b5 粗值交叉校验: |y_fine - y_b5| > 2.0m (实测 p99=2.25m) 判为
            #      sub3 错帧/别名, 退 b5 (b5 有车道三峰结构独立验证, 是可信基线)。
            # 【2026-11-11 c144 退出】横向唯一来源 = b5 (0.798°/LSB 视觉幅度标定)。
            #   原因: c144 的编码模型三次标定互相矛盾 (0.000061/0.00002/0.00001,
            #   全部基于"跟车 y 恒定"假设 —— 弯道下不成立, 标定互相污染);
            #   官方同族 Azimuth 是 10bit@0.002 全FOV±58°, c144 无论如何解释都在
            #   近距旁车 (8.5°) 处有歧义。b5 的 0.798°/LSb 是唯一有独立真值
            #   (9 视觉大横向样本) 背书的横向来源 —— 数据源唯一、每个数字有出处。
            #   c144 字段保留解码但不再参与 yRel (下方代码注释保留, 待编码模型
            #   有独立真值标定后再评估启用)。
            yrel = SONG_PLUS_YREL_SCALE * dist_raw * (b5 - SONG_PLUS_YREL_OFFSET)   # 唯一横向来源
            # (已删 2026-11-13) 原 c144 交叉校验块 —— 2026-11-11 已用 `if False:` 停用, 现整块删除。
            #   c144 的编码模型三次标定互相矛盾 (0.000061 / 0.00002 / 0.00001, 全部基于
            #   "跟车 y 恒定" 假设, 弯道不成立), 近距旁车 (8.5°) 超出其 ±0.66° 表达范围会回绕。
            #   若要重启用, 必须先有独立真值标定, 不能沿用任何历史 scale。
            # 横向物理上限 (2026-08-29, seg20 47.4s/58.5s 实测破解):
            # ① 目标消失前 0.2~0.3s 雷达内部重捕获噪声使 b5 单帧 95→31 跳变,
            #    yRel = 0.014*73*(31-95) ≈ -65m 假点会新建幽灵 track (id340) 污染关联 2s+;
            # ② 目标清丢后总线上出现成串杂波 (yRel -20~-65, dist 20~55m, 护栏/隔离带),
            #    差分 vRel 未收敛 (≈0) 时逃过静态过滤直接输出。
            # 真实交通目标横向 ≤±7m (三车道), radard 横向需求上限 cut-in 4m;
            # |yRel|>10m = 毛刺/杂波, 直接丢弃 (左侧近距大角度真目标 b5=14/8m ≈ -9.1m 仍保留)。
            if abs(yrel) > 10.0:
                continue
            # 距离修正: 按实测偏差曲线分段线性 (见 SONG_PLUS_DREL_CURVE 注释)
            # 注意: np.interp 返回 numpy.float64, 直接赋给 capnp 字段(dRel)会抛
            # KjException(Tried to set field ... unsupported type numpy.float64) → card 崩溃,
            # 必须 float() 转回 Python 标量。2026-08-25 实测 crash, 根因确认。
            # 【2026-11-13 删除 b6 量程档收缩】sub0.b6 不是距离量程档, 是航迹状态字节
            #   (bit7=段标志 + 已确认段=生命周期阶段 0-24 / 未确认段=距离相关 237-255)。
            #   实测 corr(b6,距离) 仅 +0.13 (量程档应强单调), corr(b6,state) -0.84。
            #   按 b6 分组的距离偏差 (视觉 lead 参考): b6=1(新建) -2.54m vs b6=2/3(成熟)
            #   +1.41/+1.58m —— 方向相反、幅度 7 倍于旧收缩 (6-15m 恒 -0.35m)。
            #   用状态字节当量程档做距离修正 = 铁律 5 违例 → 删除。距离只走
            #   SONG_PLUS_DREL_CURVE (纯距离偏差曲线, 与 b6 无关)。
            corr = np.interp(dist_raw, SONG_PLUS_DREL_CURVE_X, SONG_PLUS_DREL_CURVE_Y)
            dist = float(dist_raw - corr)
            if dist < 0.5:
                continue  # 修正后 <0.5m = 贴脸/噪声
            # ---- sub1.b6 = 每目标横移档位 (2026-10-03 新解) → yvRel ----
            # 零点 125 (中性), 110~119 = 向右移 (左正坐标下为负), 130~139 = 向左移;
            # 帧级稳定 (相邻帧 |Δ| 中位 0), 方向与真实横移一致率 78.7%(横移)/91.7%(本道),
            # 两侧饱和 ≈±3 m/s。125 占 32% 帧 (目标不横移)。255 语义未定 → 按 0 处理。
            # 标度 0.25 m/s/count 由帧级分箱标定 (110~119→-2.82, 130~139→+2.31 中位), 饱和限 ±3。
            s1 = frame_map.get(base_addr + 1)
            yv_rel = 0.0
            if s1 is not None:
                _s1b6 = int(s1[6])
                if _s1b6 != 255:
                    yv_rel = max(min((_s1b6 - 125) * SONG_PLUS_YVREL_PER_COUNT, 3.0), -3.0)
            # ---- sub3.b7 低 6 位 = 雷达原生航迹 ID (2026-10-21 新解, 全量 701 段验证) ----
            # 同目标 ID 不变率 99.47% (860,838 样本), 换目标时跳变 (检出 5,385 / 漏检 84 = 1.54%)。
            # 用途: 确定性识别"槽内换目标", 取代原先靠距离/速度差分的启发式 _slot_changed。
            # 提取式 (与位提取自检一致): native_id = s3[7] & 0x3F == (pl >> 184) & 0x3F
            _native_id = -1
            _q4 = -1          # -1 = sub3 帧缺失/无效 (质量场不可用), ≥0 才是有效字节值
            s3f = frame_map.get(base_addr + 3)
            if s3f is not None and len(s3f) >= 8:
                _native_id = int(s3f[7]) & 0x3F
                _q4 = int(s3f[4])
            points.append((dist, yrel, state, slot, dist_raw, yv_rel, _native_id, _q4))

        # 阶段3: 帧间最近邻关联 (借鉴 Ford MRR cluster_points)
        # 每个活跃 slot 以 14.89Hz 刷新, 但每条 CAN 消息 (100Hz) 只带其中几个 slot,
        # 故同一目标在连续的 CAN 消息里是"隔几帧出现一次"(间隔中位 0.069s)。
        # 关联基准用"该 track 最近一次位置"(跨帧保留, 供下次刷新匹配), 而非"仅上一帧"。
        # 匹配: score = |Δd|/DREL_TOL + |Δy|/YREL_TOL, score < 2.0 视为同一目标
        # 匹配上 → 继承 trackId (绿框跟随); 匹配不上 → 新目标, 分配新 trackId。
        # 2026-08-15 借鉴官方 radard Track 平滑: track 中心存 (dRel, yRel, vRel, t),
        # 匹配时用 vRel 预测当前位置 (pred = last + vRel*dt), 目标接近/远离时匹配更准。
        # 2026-08-20 FIX1-v2: track 状态扩展存来源 slot。宋Plus 0x380 的 slot 是稳定传输槽
        # (tid 会随距离排序重排 14→13→...→7, 但同一物理目标持续占同一 slot), 关联时
        # 同 slot 且距离近优先匹配 → 减少"前车快速接近时 tid 重排导致目标分裂成两个 track"
        # 引发的绿框/前车速度跳变。回放实测(全19段): 零误合并(track内跳变0, 与BASE同),
        # s1track 283→262(-7.4%), 切换 496→479(-3.4%); seg4 分裂改善显著(25→19, 切换32→28)。
        prev_tracks = self._song_plus_tracks  # {track_id: (dRel, yRel, vRel, t, slot, yvRel)} 上次状态 (2026-11-06 加 yvRel)
        new_tracks = {}  # track_id -> (dRel, yRel, vRel, t, slot, yvRel) 本帧更新后
        assigned = set()
        slot_used = set()  # FIX1: 本帧已被匹配的 slot (同 slot 优先只作用于首个候选)
        _static_dropped = set()  # 【M4】本帧被高速静态物过滤剔除的 track, 保留循环需跳过
        _replaced = set()  # 【2026-11-05】原生航迹 ID 变化 = 槽内换目标, 旧 track 本帧末丢弃

        for dist, yrel, state, slot, dist_raw, yv_rel, native_id, q4 in points:
            best_track = None
            best_score = float('inf')
            for tid, _tr in prev_tracks.items():
                if tid in assigned:
                    continue
                pd, py, pv, pt, pslot = _tr[:5]
                pyv = _tr[5] if len(_tr) > 5 else 0.0   # 横向速度 (2026-11-06 接入; 兼容旧 5 元组)
                # 速度预测: 目标按上次 vRel 移动 dt 后的预期位置
                dt = now - pt
                if dt > 0 and dt < 0.5:
                    pred_d = pd + (pv * dt if abs(pv) > 0.05 else 0.0)
                    pred_y = py + pyv * dt   # 【2026-11-06】横向速度预测 (原"保持", 横移目标跟丢原因)
                else:
                    pred_d = pd
                    pred_y = py
                score = (abs(dist - pred_d) / SONG_PLUS_MATCH_DREL_TOL
                         + abs(yrel - pred_y) / SONG_PLUS_MATCH_YREL_TOL)
                # FIX1-v2 (2026-08-20): 同 slot 候选 score 打 0.3 折 → 优先继承同槽 track。
                # slot 是雷达传输容器, tid 重排时物理目标不变, 同 slot 是最强连续性线索。
                # 保守条件: 仅当距离预测差 <3m 才折扣。无条件折扣(v1)会把"slot 换目标但
                # 距离相近"误继承成同一 track (回放实测 track 内跳变 85 次), 加距离门后
                # 零误合并 (跳变 0) 且 seg4 分裂改善保留 (s1track 25→19, 切换 32→28)。
                if pslot == slot and slot not in slot_used and abs(dist - pred_d) < 3.0:
                    score *= 0.3
                if score < best_score:
                    best_score = score
                    best_track = tid
            # 【2026-11-05 原生航迹 ID = 目标身份, 对标丰田 NEW_TRACK】
            #   丰田 radar_interface: NEW_TRACK 置位 → 分配新 trackId (不继承旧历史)。
            #   我们解出的 sub3.b7&0x3F 就是雷达自己的航迹 ID (全量 701 段: 同目标不变率
            #   99.47%)。若槽内换了目标, 仅"本帧禁用 b6"不够 —— hist/vRel 低通/限变锚点
            #   仍带着前任目标的距离序列 (距离相近时不触发 >3m 清空) → vRel 被污染 0.2~0.5s。
            #   这里按丰田语义: ID 变化 = 新目标, 分配新 trackId 并丢弃旧 track (不做跨目标继承)。
            _matched = best_track is not None and best_score < 2.0
            _prev_nid_of_best = self._song_plus_native_id.get(best_track) if _matched else None
            # 【2026-11-05 弯道修正】原生 ID 在弯道上会抖动 (1a--6 实测: 同一目标 d=53.9
            #   恒定, nid 97→98→94 反复换, 0.5s 内 13 次) —— 雷达横向重聚类时重新编号,
            #   不是换目标。仅"ID 变化"就清历史会让差分持续无结论 → vRel=0 → 弯道漏刹。
            #   修正: ID 变化 且 距离也跳变 (>1.5m) 才判换目标重置; 距离连续 = 同一目标
            #   被重新编号, 保留差分历史 (仅更新存储的 nid)。
            # 【2026-11-13 几何双跳变】实锤 (.comma25 seg15 23:27 #473): 排队停止目标被
            #   槽位重排, 同一辆车以新 nid 在 slot0 复活, 距离阶跃 +1.75m 但横向连续 (b5
            #   同值)。旧判据"距离跳>1.5m 即换目标"把它当新车 → 清差分历史 + 重出生门
            #   → vRel=-0.5 假值 (真值 -2.2) → 规划器继续加速冲向停着的真车。
            #   换目标 = 距离**和**横向都跳 (几何整体换了一台车); 只跳距离 = 同车测量
            #   阶跃/重新编号, 承接历史。横向量化噪声 ±1m (70m 处 ±0.97m) 不触 1.5m 门。
            _hist_bt = self._song_plus_hist.get(best_track) if _matched else None
            _nid_d_jump = True
            if _hist_bt:
                _d_jump = abs(dist - _hist_bt[-1][1])
                _y_jump = abs(yrel - _hist_bt[-1][2])
                # 护栏 (2026-11-13): 原判据要求"距离和横向都跳"才换车, 反向风险是
                #   同车道前后两车在槽内替换、横向 <1.5m 时, 会把前车的历史速度带给后车。
                #   加一条上限: 横向虽连续, 但距离跳 >6m (远超重编号的 +1.75m 阶跃,
                #   也不是量化噪声 0.06m) → 仍判换车。
                _nid_d_jump = ((_d_jump > 1.5 and _y_jump > 1.5) or _d_jump > 6.0)
            _native_replaced = (_matched and native_id >= 0
                                and _prev_nid_of_best is not None and _prev_nid_of_best != native_id
                                and _nid_d_jump)
            if _matched and not _native_replaced:
                track_id = best_track
                assigned.add(best_track)
            else:
                track_id = self._song_plus_next_id
                self._song_plus_next_id += 1
                assigned.add(track_id)
                self._song_plus_birth[track_id] = now
                if _native_replaced:
                    assigned.add(best_track)   # 防止同帧另一候选再匹配旧 track
                    # _replaced 的完整清理在下方"保留未轮询旧 track"循环里统一做
                    # (含 self.pts / hist / last_t / vrel_filters / vrel_out / not_seen /
                    #  native_id / birth / expiring 共 9 项)。此处只登记, 不重复清。
                    _replaced.add(best_track)
            if slot not in slot_used:
                slot_used.add(slot)

            # 状态码 130 = 即将消失 (见 SONG_PLUS_STATE_VALID 注释): 记录该 track 的最后
            # 感知时刻, 本轮匹配结束后若不再出现, 只保留 SONG_PLUS_EXPIRING_HOLD 秒
            # (丰田 NEW_TRACK/255 立即删点的近似); 重新出现 128/129 即清除标记。
            if state == 130:
                self._song_plus_expiring[track_id] = now
            else:
                self._song_plus_expiring.pop(track_id, None)

            # 阶段4: vRel 距离差分 + 输出低通 (track_id 现在关联稳定, 差分有效)
            # 雷达 14.5Hz 轮询分片传输: 同目标每 ~0.07s 匹配一次, 但批间隔仅 ~0.01s。
            # 故 hist 保留窗口必须 > 差分窗 (否则 0.5s 窗刚够就被 popleft 挤出,
            # 实测 hist 最大跨度 0.47s 永远达不到差分窗 → vRel 恒 0)。
            # 实际保留 2.0s (见下方 popleft): 覆盖 1.5s 长窗差分 + 1.6s aLead 拟合窗。
            hist = self._song_plus_hist.setdefault(track_id, deque(maxlen=32))
            # 距离突变保护: 相邻匹配距离跳变 >3m = 目标切换/噪声, 不能算差分
            # (与聚类阈值一致; 过大会把切换目标混入差分 → 假速度)
            if hist and abs(dist - hist[-1][1]) > 3.0:
                hist.clear()
            hist.append((now, dist, yrel))
            while len(hist) >= 2 and now - hist[0][0] > SONG_PLUS_VREL_LWIN_WIN + 0.5:  # 保留 2.0s (长窗差分用)
                hist.popleft()

            vrel = 0.0
            # 【P1 2026-10-03】差分有效性标志: 量化门未触发 / 横向跳变抑制 / 新目标无历史时
            # vrel=0.0 是"**无结论**"而非"真实 0", 不能参与 b6 互补融合 (否则把目标拉向
            # 0 = 低估接近率 = 危险方向)。仅真正算出差分值时置 True。见 SONG_PLUS_VREL_B6_FUSE_BETA。
            _v_diff_ok = False
            # 低速跟停起步: 前车起步时本车静止(v_ego=0), 差分vRel必须能测到前车远离
            # 2026-08-16 去除 abs(v_ego)>=1.0 门控 (原门控导致跟停起步 vRel 恒 0)
            # 距离阈值 3 LSB (0.1875m): 0.5s 窗对应 0.375 m/s 最小可测速度,
            # 前车缓慢起步(0.2~1 m/s)也能测到; 静止时 Δd=0 不触发, 纯量化抖动(1-2 LSB)不触发
            # 2026-08-26 保留窗扩到 2s 后, 短窗点改取"≥0.45s 前的最新样本"(窗≈0.5s),
            # 不再用 hist[0] (否则窗会被拉长到 2s)。
            # 【2026-11-04 揭示期】出生 <SONG_PLUS_VREL_YOUNG_AGE 的 track 走快收敛:
            #   短窗差分 0.25s 窗 + 低通 tau 封顶 (见常量注释; 静止障碍被前车让开揭示后
            #   vRel 收敛 0.9s → ~0.5s)。防毒不靠窗/tau: D2 质量门/M7/b7/输出限变全保留。
            _young = now - self._song_plus_birth.get(track_id, -1e9) < SONG_PLUS_VREL_YOUNG_AGE
            short_pt = None
            if _young:
                _win_s = SONG_PLUS_VREL_WIN_YOUNG
            elif dist >= SONG_PLUS_VREL_WIN_FAR_DREL:
                _win_s = SONG_PLUS_VREL_WIN_FAR     # 远距长窗 (见常量注释)
            else:
                _win_s = SONG_PLUS_VREL_WIN
            for pt_t, pt_d, pt_y in reversed(hist):
                if now - pt_t >= _win_s - 0.05:
                    short_pt = (pt_t, pt_d, pt_y)
                    break
            if short_pt is not None:
                t_old, d_old, y_old = short_pt
                dt = now - t_old
                # 弯道斜距假接近抑制 (2026-08-20): 差分窗内横向移动大 → dist 变化含横向
                # 分量, 不是真纵向接近 → 抑制 vRel(置0走低通, 平滑收敛到 0)。
                # 全19段回放: 跳变 30→20(-33%), 无段恶化。
                if abs(yrel - y_old) > SONG_PLUS_VREL_MAX_DY:
                    vrel = 0.0
                else:
                    # 【2026-11-04 差分窗质量门】窗内相邻距离增量的二阶跳变超 0.8m = 单次跳变
                    #   污染 (3m 清窗门以下的跳变逃过 hist.clear), 整窗差分不可信 → 判"无结论"
                    #   (vrel=0, _v_diff_ok 保持 False) → b6 可用则接管, 否则保持 0 走低通。
                    _win_d = [p_d for p_t, p_d, _p_y in hist if p_t >= t_old]
                    _d_jump_ok = True
                    if len(_win_d) >= 3:
                        _dd_list = [_win_d[i + 1] - _win_d[i] for i in range(len(_win_d) - 1)]
                        _d2_max = max(abs(_dd_list[i + 1] - _dd_list[i]) for i in range(len(_dd_list) - 1))
                        # 远距长窗样本更多 → max 更容易触门, 阈值按窗长放宽 (1.5s→×1.6)
                        _d2_limit = SONG_PLUS_VREL_WIN_D2_MAX * (1.6 if _win_s > SONG_PLUS_VREL_WIN else 1.0)
                        if _d2_max > _d2_limit:
                            _d_jump_ok = False
                    if not _d_jump_ok:
                        # 无结论: 不用短窗差分, 不 continue —— 长窗(下方)与 b6 融合仍有机会接管
                        vrel = 0.0
                    else:
                        raw_vrel = (dist - d_old) / dt
                        if abs(dist - d_old) >= 0.1875:
                            vrel = max(min(raw_vrel, 30.0), -30.0)
                            _v_diff_ok = True
            # 2026-08-26 长窗差分主导 (e266140d seg8 06:40 幻影刹车): 大横向偏移(|yRel|>2m,
            # 弯道中本车道前方的几何投影)时, 0.5s 帧级短窗差分被斜距几何污染 → vRel 正负抖动
            # → vLead 振荡 (案例 12~17m/s 来回摆) → MPC 追"幻影减速"刹过头 (24→10.4m/s, 
            # 实际前车 ~12.5m/s 恒定且在远离)。1.5s 长窗差分把量化噪声/几何抖动积分掉,
            # 作为该工况的速度源 (双侧取值, 经低通平滑无切换阶跃)。
            # |yRel|≤2m (目标在本车道正前方) 仍用短窗差分 (响应快, 质量好)。
            # 门: 窗内横向移动 <2m (排除换道目标/静态物斜距投影); |Δd|≥0.1875m (量化噪声门)。
            if abs(yrel) > SONG_PLUS_VREL_LWIN_YREL and len(hist) >= 2:
                for pt_t, pt_d, pt_y in reversed(hist):
                    if now - pt_t >= SONG_PLUS_VREL_LWIN_WIN:
                        if abs(yrel - pt_y) < SONG_PLUS_VREL_LWIN_DY:
                            dd = dist - pt_d
                            if abs(dd) >= 0.1875:
                                vrel = max(min(dd / (now - pt_t), 30.0), -30.0)
                                _v_diff_ok = True
                        break
            # ---- b6 精细接近率替代 (2026-09-20 破解落地; 常量/公式注释见文件头) ----
            # <50m 且速度槽有 ≤1s 的新鲜 real 帧 → 用 b6 换算 (对决 MAE 优 19~40% 且不失效);
            # >50m / 非 real / 缓存过期 / **质量位差** / **本帧刚换槽** → 保留上方差分结果。
            # b6 换算用未修正 raw 距离 (标定口径; ⚠️ 速度帧自带的 d 字段不是目标距离, 仅作 fill 哨兵)。
            # 强接近 (v<-3) 保留 0.7 保守偏置 (更接近方向=安全; 新标定 k=6.5 后该段残差 ~0.3)。
            # 限幅 ±30 与差分路径一致 (real 帧 b6 偶见 -56..+28 极端值, 占 0.28%)。
            #
            # 【10-03 修订 1】b2 粗接近率兜底**已移除** (证伪: 见 SONG_PLUS_VREL_B2_* 注释区)。
            #   b2 = 雷达内部 per-slot tracker 状态量, 非连续速度编码: 真零点 8.5 (非 0.05)、
            #   1~15 低域非单调、slot 换目标后秒级串位 (08:01 急刹事件: g1.b2=23~35 ↔ 真值
            #   v/d 仅 -0.19, 换算 -14~-19 → 幽灵慢车 → aTarget -4.0)。b6 不可用时一律退差分。
            # 【10-03 修订 2】质量位门: 0x3C8+g.b7 bit0 = 1 时该帧 b6 误差翻倍
            #   (实测 MAE 0.553 → 1.071 m/s, 占 real 帧 8.4%) → 常规退差分。
            #   ⚠️ **强接近豁免** (同轮审核定): 该位在急刹帧高度集中 (实测 |真接近率|>3 的帧
            #   31% 带位, 而 ~0 帧仅 0.4%), 而急刹正是差分最不可靠 (0.5s 窗+6.25cm 量化,
            #   低估接近=刹车偏晚=危险方向) 的时刻 → 门只在"非强接近"时启用。
            #   豁免集残余风险 (同轮数据核实, 勿再重复调查): 坏帧+自称强接近的集合里
            #   4.6% 帧 |误差|>3 m/s (P99=6.8, max 7.5)——差分佐证无区分度 (垃圾帧 88% 差分
            #   同样显强接近 vs 真帧 99%)、换槽门也不覆盖 (非换槽帧)。**维持豁免的依据**:
            #   ①坏帧误差方向=高估接近=刹车偏早 (安全侧); ②孤立坏帧被输出低通 (单帧冲量
            #   ~1.3 m/s) + radard P1 vLead 跳变钳制 (2 m/s/帧) 双重吸收; ③不豁免的代价
            #   (急刹 30% 帧退差分→低估接近→偏晚刹) 在危险方向。
            # 【10-03 修订 3】换槽失效: b6 缓存是**槽级**, 槽内目标被距离排序换走后,
            #   缓存仍是前任的值 → 本帧换槽 (prev_tracks 的 slot != 当前 slot) 时禁用 b6 一次。
            _slot_changed = False
            _prev = prev_tracks.get(track_id)
            # 【2026-11-08 修正】删掉"slot 变化 = 换目标"判据。
            #   实测 slot 是【距离排名】: 637 个多目标帧 100% 满足"slot 越小距离越近"
            #   (slot0 中位 35.8m / slot1 63.4m / slot2 85.0m / slot3 108.3m)。
            #   目标接近时排名上升 → slot 编号必变, 但目标身份没变 (同一 native_id 跨
            #   slot 移动: nid56 用过 s0/s1/s2)。旧判据使 3.2% 的帧 (440/13804) 在
            #   "目标跨越排名边界的瞬间"被禁用速度字段 —— 恰是最需要精确 vRel 的时刻。
            #   换目标判定只保留 native_id (sub3.b7, 不变率 99.47%) 与新 track 两条。
            # 【2026-10-22 审计修复 M7】新建 track 首帧必须禁用 b6 一次。
            #   两条换目标判据都依赖 prev_tracks / _song_plus_native_id 的旧值,
            #   而关联失败导致"分配全新 track_id"这一最常见换目标形态下, 两个 get 都
            #   返回 None → _slot_changed=False → 首帧放行 b6。b6 缓存是**槽级**,
            #   新鲜度窗 1s, 此时缓存仍是槽内前任目标的值 → vRel 可错 1 档(≈1.1m/s@40m)
            #   以上, 经低通污染 0.2-0.5s。
            #   新 track = 没有历史 = 无从判断是否换目标 → 保守按"已换目标"处理。
            if _prev is None:
                _slot_changed = True
            # 【2026-10-21】用雷达原生航迹 ID (sub3.b7 低 6 位) 做确定性换目标检测。
            # 旧判据 (槽号变化 + 距离/速度差分) 是启发式, 4.26% 的槽内换目标漏检会毁掉 vRel。
            # 全量 701 段验证: ID 不变率 99.47%, 换目标检出 5,385 / 漏检 84 (1.54%)。
            if native_id >= 0:
                _prev_nid = self._song_plus_native_id.get(track_id)
                if _prev_nid is not None and _prev_nid != native_id:
                    _slot_changed = True
                self._song_plus_native_id[track_id] = native_id
            # 【2026-11-05 弯道长窗兜底】短窗差分无结论 (强弯距离噪声 ±1.5m@30Hz, D2 门
            #   频触发, 1a--6 实测全程 NA) 时, b6 单独兜底在强弯上不可信 (实测 b6=+2.2 远离
            #   而长窗趋势 -3.6 接近, 方向相反)。预计算 hist 长窗差分 (span≈1.4s, hist 保留
            #   2s), 供 b6 交叉验证与质量位跳过时的兜底; 逐帧噪声在长窗被平均。
            _v_lwin = None
            _b6_used = False
            if SONG_PLUS_VREL_LWIN_FALLBACK and (not _v_diff_ok) and len(hist) >= 4:
                _t_old, _d_old = hist[0][0], hist[0][1]
                for _he in hist:
                    if now - _he[0] <= 1.4:
                        _t_old, _d_old = _he[0], _he[1]
                        break
                _dt_l = now - _t_old
                if _dt_l >= 1.0:
                    _v_lwin = (dist - _d_old) / _dt_l
            if dist_raw < SONG_PLUS_VREL_B6_MAX_DREL and not _slot_changed:
                _c6 = self._song_plus_b6_cache.get(slot)
                if _c6 is not None and now - _c6[1] <= SONG_PLUS_VREL_B6_FRESH_S:
                    # 【2026-11-06】缓存存的是 11 位解码后的 m/s (见常量区), 直接用
                    v_b6 = float(_c6[0])
                    # 【2026-11-13 读侧反转】旧式 `not _c6[2] or v_b6 < -3` 把 b7.bit0=1
                    #   的帧判为可疑 → 而写侧 `_b6r<249` 只收 bit=1 的帧 → 缓存里 90% 的值
                    #   只能靠"v_b6<-3 强接近"这一条出路 (删门 A/B 只动 0.1 m/s 的根因)。
                    #   实测 (11 位速度): bit=1 侧 |err| 0.65 < bit=0 侧 0.84, bit=1 更准 →
                    #   读侧应信任 bit=1。改为 `if _c6[2] or ...`: bit=1 帧恒用, bit=0 帧
                    #   仅强接近时用。写侧 _b6r<249 门一并删除 (见上方缓存写入)。
                    if _c6[2] or v_b6 < SONG_PLUS_VREL_B6_BIAS_V:
                        if v_b6 < SONG_PLUS_VREL_B6_BIAS_V:
                            v_b6 -= SONG_PLUS_VREL_B6_BIAS
                        v_b6 = max(min(v_b6, 30.0), -30.0)
                        # 【P1 2026-10-03】互补融合 (取代"b6 直接替换差分")。
                        # 差分无结论时保持原行为(替换), 融合权重见 SONG_PLUS_VREL_B6_FUSE_BETA 注释。
                        if _v_diff_ok:
                            # 【P1b】β 按目标成熟度自适应 (sub2.b7 轨迹计数器, 见常量注释)
                            _b7f = -1
                            _s2f = frame_map.get(0x380 + 4 * slot + 2)
                            if _s2f is not None and len(_s2f) >= 8:
                                _b7f = _s2f[7]
                            _beta = (SONG_PLUS_VREL_B6_FUSE_BETA_NEW
                                     if 0 <= _b7f < SONG_PLUS_VREL_B6_FUSE_B7_MAX
                                     else SONG_PLUS_VREL_B6_FUSE_BETA)
                            # 【2026-11-05 失配保护】b6 与差分严重失配 (填充残留/换目标串位)
                            #   → 丢弃 b6 保差分。合法失配 (两路各自误差之和) 实测 <2.5 m/s,
                            #   毒化样例 +5.2 m/s, 4.0 阈值留足余量。
                            if abs(v_b6 - vrel) <= 2.0:   # 2026-11-06 收窄: 新字段 MAE 0.598 (原 b6 1.832)
                                vrel = max(min(vrel + _beta * (v_b6 - vrel), 30.0), -30.0)
                        else:
                            # 【2026-11-05 弯道长窗兜底】与 b6 交叉验证 (见上方 _v_lwin 注释):
                            #   失配 >3 → 弃 b6 用长窗差分; 一致 → 用 b6 (更新鲜)。
                            if _v_lwin is None or abs(v_b6 - _v_lwin) <= 2.0:   # 2026-11-06 收窄, 同上
                                vrel = v_b6
                            else:
                                vrel = max(min(_v_lwin, 30.0), -30.0)
                            _b6_used = True
                # b6 不可用 (无缓存/过期/坏帧且非强接近/本帧换槽) 或 >50m → 保留上方差分结果。
                # 【10-03】原此处 b2 粗接近率兜底已删除 (证伪: tracker 状态量, 换目标串位,
                #   08:01 事件 b2=23~35 ↔ 真值 v/d -0.19 → 假接近 -14~-19 → 幽灵慢车急刹)。
            # 【2026-11-05 弯道长窗兜底】b6 未采用 (质量位跳过/无缓存/过期) 且短窗无结论
            #   → 用长窗差分兜底 (弯道上唯一方向正确的来源)。
            if SONG_PLUS_VREL_LWIN_FALLBACK and (not _v_diff_ok) and (not _b6_used) and _v_lwin is not None:
                vrel = max(min(_v_lwin, 30.0), -30.0)
            # 输出低通 (每 track 独立, 平滑差分噪声; 低速/无效帧也更新 → 速度平滑收敛)
            # 2026-08-26 不对称低通: 上升沿 (vRel 升 = 目标加速/远离, 无碰撞风险方向) tau 压缩
            # 快速跟随; 下降沿 (目标减速/接近, 危险方向) 维持原 tau 平滑不变。
            v_filter = self.vrel_filters.setdefault(track_id, LowPassFilter(vrel))
            if self.v_ego < 10.0:
                tau = 0.5
            elif self.v_ego < 20.0:
                tau = 0.3
            else:
                tau = 0.2
            # 【2026-10-21】质量场 (sub3.b4) 调 vRel 低通时间常数。
            # 全量 701 段 842,008 样本实测下一帧 |Δd|:
            #   bit5=1 (降级) 0.312m  |  bit5=0 且 低4位≥11 (好) 0.125m  —— 2.5x
            #   低4位 q=14/15 → 0.062m (最稳), q=0-3 → 0.188-0.250m。
            # 差帧用更长 tau 平滑, 好帧用更短 tau 快速跟随。
            # 【2026-11-05 字段反查补充】sub3.b4 不是单纯质量字节:
            #   bit7 与状态码强绑定 (state128 时 93.5% 置 1, state129/130 仅 16.7%);
            #   低4位是距离噪声等级 (state129 内 g=11 抖动 0.116m, g=0-3 0.38-0.45m,
            #   g=15 0.35m 非单调, ">=11 好" 需谨慎); 低7位**不是检测置信度**。
            #   0x380 全字段反查(71 段)确认无 SCORE 等价字段, 勿把 q4/状态码用作新目标
            #   发布门控 (实测门控精度 4%/2.4%)。
            # 【2026-11-05 深夜补测】bit7×bit5 联合抖动表 (77k 样本):
            #   b7=0/b5=0 0.202m | b7=0/b5=1 0.355m | b7=1/b5=0 0.309m | b7=1/b5=1 0.458m
            #   → bit5 ×1.63 (现有 1.6 正确), bit7 ×1.37 为独立轴, 但**不加**:
            #   离线 A/B (71 段, 同输入同参考) 实测加 bit7×1.4 使 median|vRel-ref|
            #   0.603→0.632 (+5%), bit7=1 子集 0.732→0.785 —— 滞后代价 > 降噪收益。
            #   现有 tau 已在噪声/滞后权衡的最优附近, 不要再动。
            # 【2026-10-22 M12 审计修正】原文误写 "bit4=1 & bit5=0" —— 代码只用
            #   bit5(降级) 与低4位(等级), 不看 bit4(有效性)。bit4 与抖动的关系
            #   被 bit5 主导 (bit5=1 时无论 bit4 都是 0.25~0.31m), 故不参与调权。
            if q4 >= 0:
                if (q4 >> 5) & 1:
                    tau *= 1.6        # 降级帧, 多平滑
                elif (q4 & 0x0F) >= 11:
                    tau *= 0.7        # 高质量帧, 快速跟随
            # 【2026-11-05 Quality/Deg 独立噪声轴】184 段 OP纵向 A/B: median -10%, 滞后 -3%。
            #   与 bit5 轴近似正交 (联合表见 SONG_PLUS_TAU_QF 注释); young 封顶仍在其后兜底。
            _s1f_q = frame_map.get(0x380 + 4 * slot + 1)
            if _s1f_q is not None and len(_s1f_q) >= 8:
                tau *= SONG_PLUS_TAU_QF.get((_s1f_q[4] >> 3) & 3, 1.0)
            _s2f_q = frame_map.get(0x380 + 4 * slot + 2)
            if _s2f_q is not None and len(_s2f_q) >= 8:
                tau *= SONG_PLUS_TAU_DF.get((_s2f_q[2] >> 3) & 3, 1.0)
            if _young:
                tau = min(tau, SONG_PLUS_VREL_YOUNG_TAU)   # 揭示期双向快跟 (见常量注释)
            if vrel > v_filter.x:
                tau *= SONG_PLUS_VREL_RISE_GAIN
            vrel = v_filter.update(vrel, max(min(now - self._song_plus_last_t.get(track_id, now), 0.2), 0.01), tau)
            self._song_plus_last_t[track_id] = now
            # 【2026-11-04 输出 vRel 帧间限变】毒帧兜底 (常量注释见 SONG_PLUS_VREL_OUT_SLEW):
            #   锚点=本 track 上次发布值, 新 track 锚=0 (揭示期收敛由 0.25s 短窗+tau 封顶负责)。
            #   只限发布值 (聚类中心/pt 同此值), 低通内部状态不回写 → 真值恢复时滤波连续。
            _v_out_prev = self._song_plus_vrel_out.get(track_id, 0.0)
            vrel = float(max(min(vrel, _v_out_prev + SONG_PLUS_VREL_OUT_SLEW),
                             _v_out_prev - SONG_PLUS_VREL_OUT_SLEW))
            # 【2026-11-09 vRel 输出斜率上限】OUT_SLEW 是每帧 m/s, 对"连续多帧同向假趋势"
            #   (差分/b6 毒化数帧) 等效斜率 18 m/s², 拦不住 → 按时间再限一道加速度包络。
            #   时间锚=上次发布时刻 (含保留循环衰减写入, 见 _song_plus_vrel_out 同步处)。
            _v_out_t = self._song_plus_vrel_out_t.get(track_id, _log_t)
            _dt_v = min(max(_log_t - _v_out_t, 0.05), 0.3)
            vrel = float(max(min(vrel, _v_out_prev + SONG_PLUS_VREL_OUT_ACC * _dt_v),
                             _v_out_prev - SONG_PLUS_VREL_OUT_ACC * _dt_v))
            self._song_plus_vrel_out[track_id] = vrel
            self._song_plus_vrel_out_t[track_id] = _log_t
            # 【2026-11-09 dRel 发布包络】距离变化超出 |vRel|×dt+量化余量 = 量测错误
            #   (串槽/重聚类/假接近), 截断发布。hist/差分已用原始量测 (1574 行先 append),
            #   故这里只改发布链 (聚类中心 new_tracks 与 pt.dRel 均取本变量), 真值恢复
            #   自动回正。⚠️ 不做横向豁免: 弯道真实斜距 dRel 变化仅 ~0.01m/帧
            #   (v·y/R), 而 |Δy|>1m 几乎全是 b5 毛刺帧 —— 豁免恰好放行最毒的帧。
            #   新 track 无锚 → 首帧不限。dt 用日志时间 (见 _log_t 注释)。
            _d_out = self._song_plus_dout.get(track_id)
            if _d_out is not None:
                _dt_d = min(max(_log_t - _d_out[2], 0.01), 0.3)
                _d_env = max(SONG_PLUS_DREL_ENV_MIN, abs(vrel) * _dt_d + SONG_PLUS_DREL_ENV_PAD)
                if abs(dist - _d_out[0]) > _d_env:
                    dist = _d_out[0] + (_d_env if dist > _d_out[0] else -_d_env)
            self._song_plus_dout[track_id] = (dist, yrel, _log_t)
            # 高速静态物过滤 (借鉴 MR76 STATIONARY_OBJ_VREL, 2026-08-20):
            # 本车高速时, 车道外(|yrel|>2.5m)且差分 vRel 判为静止(vLead≈0) 的目标 = 护栏/路牌/隔离带。
            # 剔除: 不写聚类中心(不再参与关联)/不写 pts(不输出绿框), 防车道边静态物周期性抢 trackId。
            # 注意: vrel_filters 低通状态保留, 目标下帧重新匹配时速度平滑不中断。
            if self.v_ego > SONG_PLUS_STATIC_VEGO and abs(yrel) > SONG_PLUS_STATIC_MAX_LAT:
                if abs(self.v_ego + vrel) < SONG_PLUS_STATIC_VLEAD_TOL:  # vLead≈0 = 静止
                    # 【2026-10-22 逐行审计修复 M4】原代码只 continue, 但下方"保留循环"
                    #   会把不在 new_tracks 的 tid 重新加回来 (它用 prev_tracks 的旧值),
                    #   导致注释宣称的"不写聚类中心 / 不写 pts" 两个都没做到 —— 静态物
                    #   仍留在聚类中心抢 trackId, 仍输出绿框。
                    #   显式记入 _static_dropped, 让保留循环跳过它。
                    _static_dropped.add(track_id)
                    continue
            # 补全聚类中心 (含速度用于下一帧预测匹配; slot 用于 FIX1 同槽优先关联)
            new_tracks[track_id] = (dist, yrel, vrel, now, slot, yv_rel, dist)   # 第7项=锚点(原始测量)

            # 【2026-11-10 出生确认门】新 track 前 SONG_PLUS_BIRTH_CONFIRM 次轮询只积累
            #   不发布 (聚类中心照写保证关联连续, pts 不写 → radard 看不到 → sticky 保持
            #   旧真前车)。动机与"不能判死"的原因见 SONG_PLUS_BIRTH_CONFIRM 注释。
            self._song_plus_birth_cnt[track_id] = self._song_plus_birth_cnt.get(track_id, 0) + 1
            if self._song_plus_birth_cnt[track_id] < SONG_PLUS_BIRTH_CONFIRM:
                self._pts_cache.pop(track_id, None)
                continue

            if track_id not in self._pts_cache:
                self._pts_cache[track_id] = structs.RadarData.RadarPoint()
                self._pts_cache[track_id].trackId = track_id
            self._pts_not_seen[track_id] = SONG_PLUS_NOT_SEEN_INIT
            pt = self._pts_cache[track_id]
            pt.dRel = dist
            pt.yRel = yrel
            pt.vRel = vrel
            pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            # aRel = 雷达原生不发加速度 (2026-10-22 博世 DBC 对照 + 12 种方法确认:
            #   本车 0x380 配置只发 LongDist/LongSpeed/LatDist/MovingState/ProbExist,
            #   不发 LongAccel/dZ/Length/Sigma —— 特斯拉同款雷达发, BYD 配置裁掉了)。
            #   博世 DBC: LongAccel: 40|10 (0.03125,-16) 存在于协议, 但我们 20 个位偏移
            #   × 3 种参照 (a_ego/a_ground/静止目标) 全部 |r|<0.03, 确认不在此 CAN。
            # 改用 _song_plus_fit_alead 的窗口二次拟合 (见下方) 输出的 aRel。
            # 【2026-10-22 精度量化】蒙特卡洛 500 次/点, 1.6s 窗 24 点 70ms 采样:
            #   距离噪声 0cm → 偏差 0.00 std 0.00 (精确)
            #             5cm → 偏差 ≤0.23 std 0.11~0.35  (可用)
            #            10cm → 偏差 0.05~0.45 std 0.39~0.72 (边缘)
            #            15cm+ → std 0.94+ (噪声大)
            #   实测距离抖动 0.125m(质量场好) / 0.312m(差) → 实际 aRel std ≈ 0.5 / 1.2 m/s²
            #   已知偏差: 二次拟合+噪声有凸性正偏差, 噪声越大越偏正 (10cm→+0.4, 20cm→+1.0)。
            # 与 ARS510 对照: 其原生 aRel 是 "first-order filter 0.9-1.5s" 的平滑值,
            #   DC 增益仅 0.035-0.08 (本身就是有滞后的滤波量) → 我们 1.6s 窗拟合时间尺度同阶。
            # radard 侧: Kalman Q 对 aRel 精度不敏感, 主要用 vRel/aLead;
            #   pt.aRel 非 NaN 才算有效 (capnp 语义), 从"恒 NaN"变"有值"是本次的实际收益。
            pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            # yvRel = 真实横向速度 (10-03 新解: sub1.b6 横移档位, 零点 125, 0.25m/s/count,
            # 饱和 ±3)。此前恒写 1.0 作"本车有横向能力"哨兵; 改真实值后:
            #   · Track.yRel_future = yRel + yvRel*RadarLatFactor 成为真实横向预测
            #     (RadarLatFactor 现为 0 → 该链休眠, 语义变化无行为影响)
            #   · 哨兵语义变为"数值 0 = 本帧不横移"; radard.get_RadarState 的
            #     "有横向能力"判据 (yRel!=0 or yvLead!=0) 对本车退化无害 (正前方目标的
            #     视觉 yRel 也≈0), 已在 radard 侧注明。
            #   · sub1 帧缺失/255 → 0.0 (不写 nan: nan 会沿 yRel_future→in_lane_prob_future
            #     比较传播, 使该目标的 cut-in 判定静默失效)。
            pass  # yvRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # measured: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            # ---- aLead: 同目标距离序列窗口二次拟合 (2026-09-24, 见 SONG_PLUS_ALEAD_* 注释) ----
            # 这是全仓 aLeadK 在本车的实际来源: pt.aLead → radard Track.aLead/aLeadK
            # → long_mpc.process_lead 的前车外推。provides_alead=True 保证基类不覆盖它。
            # 取不到(新目标/断链/横向漂移/残差超门) = 0.0: 语义与旧版一致, radard C3(视觉加速度)
            # 在 |aLeadK|≈0 且 |jLead|≈0 时自动接管兜底。
            _alead_fit = _song_plus_fit_alead(hist, now)
            if _alead_fit is None:
                pass  # aLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            else:
                pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
                pass  # aLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)

        # 保留本帧未轮询到的旧 track (目标只是隔几帧出现, 不是消失)
        # 过期判定: now - 上次匹配时间 > SONG_PLUS_TRACK_TIMEOUT → 目标真正消失,
        # 不保留 → 走 stale 清理删除绿框 (否则幽灵 track 永驻, 绿框不消失 + 污染配对)
        # 2026-08-17 修复: 保留期间 vRel 向 0 衰减 (目标未轮询=无新距离观测, 旧 vRel 不可信)。
        #  段26 实测: id896 在 dRel 卡 24.75m 后 vRel 仍残留 -5.36 → L2 带假速度重刹到 1848。
        #  保留时长 ≤2.0s, 每帧衰减到 0, 目标重新轮询时由差分法恢复真实 vRel。
        for tid, pos in prev_tracks.items():
            if tid in _static_dropped:
                # 【M4】本帧被判为高速静态物 (护栏/路牌/隔离带) → 不保留,
                #   同时从 pts 移除, 防止绿框留在车道边。
                self.pts.pop(tid, None)
                continue
            if tid in _replaced:
                # 【2026-11-05】原生航迹 ID 变化 = 槽内换目标 (丰田 NEW_TRACK 语义):
                #   旧 track 不能保留 (否则旧位置/旧 vRel 以衰减值继续发布 2s → 幽灵前车),
                #   立即从 pts 与全部速度状态中移除。
                self.pts.pop(tid, None)
                self._song_plus_hist.pop(tid, None)
                self._song_plus_last_t.pop(tid, None)
                self.vrel_filters.pop(tid, None)
                self._song_plus_vrel_out.pop(tid, None)
                self._song_plus_vrel_out_t.pop(tid, None)   # 2026-11-09 斜率时间锚同步清理
                self._song_plus_dout.pop(tid, None)          # 2026-11-09 dRel 包络锚同步清理
                self._pts_not_seen.pop(tid, None)
                self._song_plus_native_id.pop(tid, None)
                self._song_plus_birth.pop(tid, None)
                self._song_plus_birth_cnt.pop(tid, None)   # 2026-11-10 出生确认计数同步
                self._song_plus_expiring.pop(tid, None)
                continue
            if tid not in new_tracks:
                # 130 目标已宣告消失 → 只留 SONG_PLUS_EXPIRING_HOLD 后立即删点
                # (丰田 NEW_TRACK/255 语义); 正常目标仍用 2s 覆盖 14.5Hz 轮询缺口。
                _hold = SONG_PLUS_EXPIRING_HOLD if tid in self._song_plus_expiring else SONG_PLUS_TRACK_TIMEOUT
                # 【2026-11-13】近车道近距目标不走 130 短收尾: 排队/停止目标被槽位重排时
                #   state 会短暂打 130, 雷达还连续几帧把 b5 打成 30~31 (重捕获噪声 →
                #   |yrel|>10 被丢) → 点从流里消失 0.6s+, 0.35s 收尾把真车删掉 (实锤
                #   .comma25 seg15 23:27 #473/#27)。本车道近距目标的代价方向是不对称的:
                #   多留只是可能多一次轻刹, 少留就是跟着假 lead 撞上去 → 放宽为 SONG_PLUS_NEAR_HOLD。
                #   真消失时 stale 清理 (_pts_not_seen) 仍会删, 不会永驻。
                # 护栏 (2026-11-13): 只在**车基本停住**时才放宽, 否则 2s 保留会让旧距离+旧速度
                #   一起发出去 (行驶中"看不见前车在动"比"丢一次点"危险得多)。
                #   v_ego 由 card 每帧写入 (card.py: RI.v_ego = CS.vEgo)。
                # 分级保留 (2026-11-13): 停车 2.0s / 蠕行(<3m/s) 0.8s / 其余走 0.5s 或 130 的 0.35s。
                #   只用 vEgo<0.5 会漏掉 23:27 的滑行段 (v 1~3, 停止目标丢 0.6s → lead 瞬切);
                #   一律放宽到 2s 又会让行驶中的旧距离+旧速度留太久。分级是两者的折中。
                if tid in self._song_plus_expiring and abs(pos[1]) < 1.0 and 0.75 < pos[0] < 25.0:
                    _v_ego_now = getattr(self, 'v_ego', 0.0)
                    if _v_ego_now < 0.5:
                        _hold = SONG_PLUS_NEAR_HOLD
                    elif _v_ego_now < 3.0:
                        _hold = SONG_PLUS_NEAR_HOLD_CREEP
                # 【2026-11-13 补洞】原来这里还要求 `tid in _song_plus_expiring`: 超窗但不在
                #   130 名单里的 track 会落进"既不更新也不删除"的空洞 —— self.pts 里
                #   那个点以最后值原样发布, 直到另一套 stale 计数 (~0.4s) 才删。
                #   实锤 .comma25 seg15 23:27 #35: 雷达在修正后 3.9m 处进入近场盲区
                #   (32 槽全空), 我们却把 d=3.50 / v=-1.80 冻结着发了 0.8s, 规划器
                #   看不到间距闭合 → 继续滑行到 1.86m 才由人工接管。
                #   超窗即删: 目标丢了就让 lead 消失, 交给视觉, 不要发旧值。
                if now - pos[3] > _hold:
                    # 130 已宣告消失且超过保留窗: 不等 stale 递减, 立即从 pts 移除,
                    # 防止幽灵前车 (vLead=vEgo) 继续参与发布/视觉配对。
                    self.pts.pop(tid, None)
                    self._song_plus_hist.pop(tid, None)
                    self._song_plus_last_t.pop(tid, None)
                    self.vrel_filters.pop(tid, None)
                    self._song_plus_vrel_out.pop(tid, None)
                    self._song_plus_vrel_out_t.pop(tid, None)   # 2026-11-09 斜率时间锚同步清理
                    self._song_plus_dout.pop(tid, None)          # 2026-11-09 dRel 包络锚同步清理
                    self._pts_not_seen.pop(tid, None)
                    self._song_plus_native_id.pop(tid, None)
                    self._song_plus_birth.pop(tid, None)
                    self._song_plus_birth_cnt.pop(tid, None)   # 2026-11-10 出生确认计数同步
                    self._song_plus_expiring.pop(tid, None)
                    continue
                if now - pos[3] <= _hold:
                    dt_unseen = now - pos[3]
                    # 【2026-11-13 取消 vRel 衰减】原来 v = pos[2]*(1-dt) 会在 1s 内归零, 而下方
                    #   距离 _pred_d 是按 pos[2] 外推的 —— 两者自相矛盾, 且方向危险:
                    #   实锤 .comma25 seg15 t=34.5: 真实接近 -4 m/s, 发布 v=-0.2 (d 却还在按 -4 走),
                    #   规划器看到"同速" → 不制动 (934.01 那段的直接原因)。
                    #   保留期短 (TIMEOUT 0.5s / NEAR_HOLD 2s) 已由"超窗即删"兜住, 不再靠归零防幽灵。
                    vrel_decay = pos[2]
                    # 【2026-11-13 保留期距离外推】旧式保留把距离冻结在最后观测值 → 本车
                    #   持续接近时"车没动" → 低估接近率 (实锤 .comma25 seg15 23:27 #27:
                    #   933.5-934.0 保留期 d 冻结 13.62, 真值已到 8.11, vRel 衰减到 0)。
                    #   物理: 位置随速度演化 → 保留期按最后 vRel 外推 (pred = d + vRel*dt)。
                    #   目标静止时 vRel≈-vEgo, 外推正好跟上本车接近; 目标驶离时同理。
                    # 【2026-11-13 关联中心改用外推值】锚点 = 最后一次真实测量的距离 (第7项),
                    #   每帧都从锚点重算 _pred_d (不会复利); 而 new_tracks[0] 写**外推值**,
                    #   让下一帧的帧间关联对上目标现在的真实位置。
                    #   动机: 原来关联中心用冻结的 pos[0], 保留 0.6s/接近 5m/s 时落后 3m,
                    #   超过 MATCH_DREL_TOL=2m → 目标回来关联不上、分新 trackId, 正好抵消
                    #   NEAR_HOLD 保住停止目标的目的 (.comma25 seg15 23:27 #473/#27)。
                    _anchor_d = pos[6] if len(pos) > 6 else pos[0]
                    # 外推下限 0.5m: NEAR_HOLD 最长 2s, 若期间 vRel=-4 会把距离推到负数
                    #   (8m 起算 → -0)。负距离进 radard 的 Track 会污染 leadOne, 这里截断。
                    # 【2026-11-13 外推限幅(方向不对称)】外推会把 vRel 的误差放大到距离上,
                    #   两个方向的后果不对称:
                    #     · 假接近 → 距离报小 → 多刹 → 安全方向, 不设限;
                    #     · 假远离 → 距离报大 → 间距看似拉开 → **欠刹**, 正是 23:27 的成因方向。
                    #   所以向外(距离变大)方向限幅 +2m, 向内不设限。
                    #   试过"向外完全冻结", 会重新制造"距离不动而 v 非零"的自相矛盾
                    #   (自洽 p50 0.33→0.43、冻结矛盾 1.3%→2.8%、跳变 1.02→2.40m), 故用限幅。
                    _d_raw_ext = _anchor_d + pos[2] * dt_unseen
                    if pos[2] > 0.0:
                        _pred_d = min(_d_raw_ext, _anchor_d + 2.0)
                    else:
                        _pred_d = _d_raw_ext
                    _pred_d = max(_pred_d, 0.5)
                    # 时间戳必须保留原始上次匹配时间 pos[3]，不能写 now:
                    # 写 now 会每帧刷新存活时间戳, now-pos[3] 恒~0 → 永不过期
                    # → 幽灵 track 永驻, 绿框不消失 + vRel 衰减失效 (2026-08-18 bug 修复)
                    new_tracks[tid] = (_pred_d, pos[1], pos[2], pos[3], pos[4],
                                       pos[5] if len(pos) > 5 else 0.0, _anchor_d)
                    # 同步衰减 pts 输出的 vRel: 旧 track 在 _pts_cache 外(仅匹配帧写入),
                    # 保留期间 stale 清理不删它 → pts 持续带旧 vRel 输出 (段26 id896 假速度 -5.4 骗重刹)。
                    # 这里直接改 self.pts[tid] 让衰减后的 vRel 生效。
                    if tid in self.pts:
                        self.pts[tid].dRel = _pred_d   # 2026-11-13 保留期距离外推 (见上方 _pred_d)
                        self.pts[tid].vRel = vrel_decay
                        self._song_plus_vrel_out[tid] = vrel_decay  # 限变锚点跟随衰减, 防重逢首帧从旧值放行
                        self._song_plus_vrel_out_t[tid] = _log_t    # 2026-11-09 斜率时间锚同步 (日志时基)
                        # _song_plus_dout 锚保留旧 d (dRel 不衰减, 保留值原样发布, 包络口径一致)
                        # vLead/aLead: 现代 RadarPoint 只有 dRel/yRel/vRel/trackId,
                        # 这两个字段不存在 (2026-11-01 审核, 写入即 AttributeError 崩 card)。
                        # vLead 可由 vRel+vEgo 在消费端推出; aLead 已在 radard 侧由差分计算。
                        # aRel 不在此置 NaN (2026-10-22 审计曾加, 后回退):
                        #   aLead 是**渐进衰减** (1s 归零), 若 aRel 立即置 NaN 会造成
                        #   "aLead 说有值、aRel 说无值"的语义不一致, 而且让 aRel 有效率
                        #   从 51% 骤降到 9.7% (每目标轮询间隔内都被标记为无效)。
                        #   aRel 的失效语义由 _alead_fit is None 时写 NaN 表达
                        #   (拟合失败 = 真的无可靠加速度), 与 aLead 的衰减语义分离是合理的:
                        #   aLead=渐进置信度, aRel=有效/无效标志。
        # 数量上限: 防止异常时 track 无限累积 (MAX_OBJECTS=10)
        if len(new_tracks) > MAX_OBJECTS:
            # 保留最近匹配到的 (new_tracks 迭代顺序: 先本帧匹配/新建, 后保留旧)
            kept = list(new_tracks.items())[:MAX_OBJECTS]
            # BUG修复 2026-08-18: 原 dropped = set(new_tracks) - set(items[:10]) 是 key集合 减
            # (key,value)元组集合, 类型不同恒不相等 → dropped=全部track, 每次截断把本车道主目标
            # 的 hist/vrel_filters 全删 → vRel 频繁归0 (AEB事件段 track4 hist 每~0.3s 被清)。
            # 应为 key 集合相减: 只删被挤出 MAX_OBJECTS 的 track。
            all_keys = set(new_tracks)
            dropped = all_keys - {k for k, _ in kept}
            new_tracks = dict(kept)
            # 被截断丢弃的 track 同步从 pts/聚类中心剔除, 否则本帧仍输出绿框且 id 重分配闪烁
            for k in dropped:
                self._pts_cache.pop(k, None)
                self._song_plus_hist.pop(k, None)
                self._song_plus_last_t.pop(k, None)
                self.vrel_filters.pop(k, None)
                self._song_plus_vrel_out.pop(k, None)   # 输出限变锚点同步清理
                self._song_plus_vrel_out_t.pop(k, None)  # 2026-11-09 斜率时间锚同步清理
                self._song_plus_dout.pop(k, None)        # 2026-11-09 dRel 包络锚同步清理
                self._pts_not_seen.pop(k, None)
                self._song_plus_native_id.pop(k, None)   # 同步清理原生航迹ID缓存, 防无限增长
                self._song_plus_birth.pop(k, None)       # 揭示期判龄缓存, 同步清理
                self._song_plus_birth_cnt.pop(k, None)   # 2026-11-10 出生确认计数同步
                self._song_plus_expiring.pop(k, None)    # 130 收尾标记, 同步清理

        # 更新帧间聚类中心 (供下一帧关联)
        self._song_plus_tracks = new_tracks

        # 清理过期目标: 存活判定 = 本帧匹配到 OR 保留在聚类中心(隔帧轮询)
        # 目标真正消失(多帧未轮询)才删除。_pts_cache 未覆盖的旧 track 走 NOT_SEEN 递减
        if self.frame % 2 == 0:
            stale = [k for k in self.pts if k not in self._pts_cache and k not in self._song_plus_tracks]
            for k in stale:
                # 递减用 get 兜底: 新 trackId 可能未在 _pts_not_seen 中初始化 (初始仅 range(255))
                self._pts_not_seen[k] = self._pts_not_seen.get(k, 0) - 1
                if self._pts_not_seen[k] <= 0:
                    del self.pts[k]
                    self._song_plus_hist.pop(k, None)
                    self._song_plus_last_t.pop(k, None)
                    self.vrel_filters.pop(k, None)
                    self._song_plus_vrel_out.pop(k, None)  # 输出限变锚点同步清理
                    self._song_plus_vrel_out_t.pop(k, None)  # 2026-11-09 斜率时间锚同步清理
                    self._song_plus_dout.pop(k, None)        # 2026-11-09 dRel 包络锚同步清理
                    self._pts_not_seen.pop(k, None)  # 删 pts 时同步删键, 防 trackId 自增致字典无限增长
                    # 【2026-10-22 审计修复 M3】原生航迹 ID 缓存同步清理。
                    #   MAX_OBJECTS 截断路径已加, 但 stale 清理路径漏了 → 长时间行驶
                    #   trackId 自增致 _song_plus_native_id 无界增长 (内存泄漏)。
                    self._song_plus_native_id.pop(k, None)
                    self._song_plus_birth.pop(k, None)   # 揭示期判龄缓存, 同步清理 (M3 同款)
                    self._song_plus_birth_cnt.pop(k, None)   # 2026-11-10 出生确认计数同步
                    self._song_plus_expiring.pop(k, None)  # 130 收尾标记, 同步清理

        self.pts.update(self._pts_cache)

        # 契约 (官方 test_car_interfaces): CAN 无效时不产出 RadarData——
        # radard 无新帧 → tracks 自然老化; 若照发空/陈旧点会被当有效雷达目标。
        if not self.rcp.can_valid:
            return None
        ret = structs.RadarData()
        ret.errors.canError = False
        ret.points = list(self.pts.values())
        return ret

    # -----------------------------------------------------------------
    #  宋Plus 23款: 0x374 RADAR_MRR 多目标 MUX 解码 (2026-08-19 重写)
    #  参考: /home/lidin/下载/byd_han_2021.dbc BO_ 884 MRR_0x374 (2021汉 DBC, MUX 多目标)
    #  帧结构 (2021汉 DBC 语义, H宋23日志 全46段实测验证):
    #    byte0 = Child_ID (MUX 选择器, 目标ID; 非低2位 TID!)
    #          空闲槽: 9/10/11/12/13 (byte1=100/0, byte2=100 固定, 距离0%有效)
    #          真实目标: 25/26/27/28 等 (byte1/2/3 变化, 距离有效 98.6~100%)
    #    byte1 = Lock_Target_Abscissa (横向坐标, 偏移-100; 2021汉DBC单位"pot"非米,
    #            主目标实测 ±7 集中/侧目标±70, 平滑有效, scale 待实车标定, 输出原值)
    #    byte2 = Lock_Target_Ordinate (纵向距离, 偏移-100, 量级=米, 雷达内部滤波)
    #    byte3 = Goal_Length_Relative_Spd (纵向相对速度档 0~12; 值越大目标越近/本车越慢)
    #    byte4 = Goal_Late_Relative_Spd (横向相对速度, 宋23 恒 0 未用)
    #    byte5 = Target_State_B6(bit0-1)+B1(bit3)+Target_Type(bit4-7) (宋23 恒 0xFC, Type=15)
    #    byte6 高4位 = AliveCounter (帧计数 0~15)
    #    byte7 = CheckSum = (sum(byte0..byte6) ^ 0xFF) % 256 (5000/5000 全中)
    #  2026-08-18 H宋23日志实测: 0x374 原车发 bus2(MPC), src=2 直收(每段~6000帧);
    #  panda fwd 转发到 bus0(段1实测 src=128 与 src=2 同刻96%同内容)。
    #  多目标: 同一时刻最多4个 ChildID (33次4帧/消息), 活跃目标可同时出现 (26,27)54次等。
    #  速度: 用 byte3 档位 (0~12, 单调反映接近/减速程度), 不用距离差分 (距离被雷达滤波,
    #  差分只能反映 4~14% 真实相对速度, 已证伪)。
    #  方案A: 指纹=BYD_SONG_PLUS_DMI_23 时直接走此方法 (用户指定车型)。
    # -----------------------------------------------------------------
    SONG_PLUS_23_VREL_WIN = 0.5   # 23款 距离差分窗口 (s) (保留, 兼容外部引用)
    SONG_PLUS_23_MIN_DELTA = 1.0  # 最小距离变化 (m) (保留, 兼容外部引用)
    # 0x374 MUX 目标判定 (2021汉 DBC Child_ID 语义, H宋23 日志实测)
    SONG_PLUS_23_IDLE_CHILD_IDS = frozenset((9, 10, 11, 12, 13))  # 空闲槽 (b1=100/b2=100 或 0)
    SONG_PLUS_23_CHILD_OFFSET = 100    # byte1 横向 pot 偏移 (实测 pot=0=正前, 保持100)
    # byte2 距离偏移 (2026-08-19 独立破解! 非2021汉DBC的100):
    #   dRel = d[2] - SONG_PLUS_23_DIST_OFFSET
    #   104.0 = 101.5(硬件真实零点, 9174停车样本回归: cid26=101.57/cid42=101.45, 斜率1.006)
    #        + 2.5(RADAR_TO_CAMERA 相机参考对齐, radard match_vision_to_track 的
    #           offset_vision_dist = lead.x - 2.5)
    #   101.5 时雷达输出的是"相机参考系"距离(≈视觉x), 但 radard 期望 dRel ≈ 视觉x - 2.5
    #   (保险杠参考), 导致雷达目标出匹配窗口 → 匹配失败橙框消失(25%匹配率)。
    #   加 2.5m 后 dRel 进入匹配窗口 → 匹配率 25%→81%(橙框恢复, 官方车标准保险杠参考)。
    #   用 2021汉 DBC 的 -100 套宋23 是错的(偏移差1.5m = "雷达比视觉远1.5m"真因)。
    SONG_PLUS_23_DIST_OFFSET = 104.0
    SONG_PLUS_23_DIST_MIN = 3.0        # 有效距离下限 (m)
    SONG_PLUS_23_DIST_MAX = 150.0      # 有效距离上限 (m)
    # ---- byte1 (Lock_Target_Abscissa) 横向标定 (2026-08-19 8192 日志决定性复测) ----
    # 直接模型 y = -0.06*pot (无CENTER!): pot 就是真实横向位置。
    #   cid26/42 (正前): pot中位≈0 → y≈0; cid25/27/43 (侧向): pot中位+30/-33/-37
    #   → y -1.8/+2.0/+2.2m (真实旁车)。同一pot桶不同距离视觉y不变 → 横向位置非方位角。
    # 之前的"CENTER 归一化"(槽位固定方位中心假说) 是错的: 侧向槽归一化后 y≈0 把旁车
    # 洗成正前 → SCC 误识别旁车(8192 路测红框指向旁车, cid27 直接y=+2.0m 被归一化成0)。
    # 直接模型下 SCC |y|<0.5 自然排除旁车(25/27/43 全 >1.8m), 保留正前(26/42 96-99%)。
    SONG_PLUS_23_YREL_K = -0.06    # 统一斜率 (m/pot), cid26 桶回归 -0.058~-0.083 收敛 -0.06
    SONG_PLUS_23_YREL_CENTER = {}  # 已废弃: 无CENTER (直接 y=K*pot), 保留空dict兼容外部引用
    # byte3 = Goal_Length_Relative_Spd (2021汉DBC: km/h 纵向相对速度, 宋23固件重缩放)。
    # 2026-08-19 终极标定 (819+H宋23 两套日志, CID26 主目标 3万+ 视觉lead.v真值配对):
    #   b3=0 → vRel≈+0.0 m/s (同速/跟随, 非接近!)
    #   b3>=1 → vRel≈+1.3~+5.2 m/s (前车远离), 查表:
    #     b3=1→+1.3, 2→+2.3, 3→+3.2, 4→+4.2, 5→+5.1 (819/H宋23 两套一致, 线性斜率≈0.94)
    # 之前标定 (b3=0→-2.6 接近, b3>=1→1.18*b3-0.32) 是错误的: 用被滤波距离差分当真值
    # (距离滤波只反映 4~14% 真实速度, 已证伪), 把速度信息抹掉导致误判。视觉 lead.v 真值
    # 显示 b3 就是相对速度, 单调映射。教训: 标定速度必须用平滑速度真值(视觉 lead.v),
    # 不能用被滤波距离的差分。
    SONG_PLUS_23_SPD_NEAR = 0.0        # b3=0 时 vRel (m/s): 同速/跟随 (819+H宋23 实测 vRel≈+0.0)
    SONG_PLUS_23_SPD_TABLE = {1: 1.3, 2: 2.3, 3: 3.2, 4: 4.2, 5: 5.1,
                              6: 5.6, 7: 6.0, 8: 6.5, 9: 7.0, 10: 7.5,
                              11: 8.0, 12: 8.5, 13: 9.0, 14: 9.5, 15: 10.0,
                              16: 10.5, 17: 11.0, 18: 11.5}  # b3>=1 查表 (m/s), 高档线性外推
    SONG_PLUS_23_SPD_K = 1.18          # 保留兼容外部引用 (已废弃)
    SONG_PLUS_23_SPD_B = -0.32         # 保留兼容外部引用 (已废弃)

    def _crc_374(self, d) -> bool:
        """0x374 byte7 校验: (sum(byte0..byte6) ^ 0xFF) % 256 == byte7 (2021汉 DBC CheckSum)"""
        return ((sum(d[:7]) & 0xFF) ^ 0xFF) % 256 == d[7] if len(d) >= 8 else True

    def _decode_radar_mrr_from_can(self, can_strings) -> bool:
        """从 can_strings 手动解析 0x374 RADAR_MRR (src=2), 输出多目标 (ChildID 分组)。
        返回是否找到有效目标。"""
        if not can_strings or not isinstance(can_strings[0], tuple) or len(can_strings[0]) < 2:
            return False
        records = can_strings[0][1]
        # 【2026-11-10 时基统一】内部时间全部用日志时间 (logMonoTime):
        #   回放/离线全速跑时墙钟间隔≈0 → 差分/长窗/关联预测全部失效 (回放与实车行为
        #   不一致的根因)。logMonoTime 与 monotonic 在实车差固定偏移 → 窗口差值等价,
        #   实车行为不变; 回放则可复现。
        try:
          _log_t = float(can_strings[0][0]) / 1e9
        except (TypeError, ValueError, IndexError):
          _log_t = time.monotonic()
        now = _log_t
        v_ego = self.v_ego if hasattr(self, 'v_ego') else 0.0

        # 取本帧所有 0x374 目标 (src=2, bus2; ChildID 分组), 更新累积缓存
        # 0x374 是轮询广播 (每条消息 1~4 个 ChildID, 序列 9,10,27,12,13 循环),
        # 单条消息只有部分目标 → 必须跨消息累积, 输出所有"最近有效"目标。
        updated = False
        for r in records:
            a, d, s = r[0], r[1], r[2]
            if a != 0x374 or s != 2 or len(d) < 8:
                continue
            child_id = d[0]  # byte0 完整值 = Child_ID (MUX 目标标识)
            if child_id in self.SONG_PLUS_23_IDLE_CHILD_IDS:
                continue  # 空闲槽 (固定坐标, 无目标)
            if not self._crc_374(d):
                continue  # 校验失败, 丢弃错帧
            dist = d[2] - self.SONG_PLUS_23_DIST_OFFSET
            if not (self.SONG_PLUS_23_DIST_MIN <= dist <= self.SONG_PLUS_23_DIST_MAX):
                continue
            # 横向 byte1 (Lock_Target_Abscissa): 直接模型 y = -0.06*(d[1]-100), 无CENTER!
            # 2026-08-19 8192 日志决定性复测: pot 就是真实横向位置(米级线性)。
            #   ① cid26/42 (正前): pot中位≈0 → y≈0 (正前, 96-99% 落 |y|<0.5)
            #   ② cid25/27/43 (侧向槽): pot中位 +30/-33/-37 → y -1.8/+2.0/+2.2m (旁车!)
            #   ③ 同一pot桶不同距离, 视觉y几乎不变 → pot是横向位置非方位角
            #   ④ cid27 时间序列: pot 平滑变化跟随目标移动 (横向位置特征)
            # 之前"CENTER 归一化"(y=K*(pot-CENTER)) 是错的: 把侧向槽的槽位中心(-33等)
            # 当"正前", 把旁车洗成正前 y≈0 → SCC 误识别旁车(8192路测红框指向旁车)!
            # 正确: 侧向槽 pot 大 = 真实旁车横向, SCC |y|<0.5 自然排除。
            pot = d[1] - self.SONG_PLUS_23_CHILD_OFFSET
            yrel = self.SONG_PLUS_23_YREL_K * pot
            if not (-89.0 <= yrel <= 89.0):
                continue
            spd_byte = d[3]  # 相对速度档 0~18
            # 速度: byte3 = Goal_Length_Relative_Spd (2026-08-19 终极标定)。
            # b3=0 → 同速/跟随 (vRel≈+0.0); b3>=1 → 前车远离 (查表 +1.3~+5.2m/s 起)。
            # 注: byte3 只编码"远离", 接近/同速都给 0。接近感知靠 dRel 缩短(radard 距离主导
            # 减速) + 主目标提升, 不在 vRel 用距离差分(雷达滤波重 + 1m量化测不出低速接近,
            # 2026-08-19 已试差分补负值: 0.5s窗 0.2m/s真实速度仅移0.1m<3LSB门槛, 差分不触发)。
            vrel = 0.0
            if spd_byte > 0:
                vrel = self.SONG_PLUS_23_SPD_TABLE.get(spd_byte,
                       self.SONG_PLUS_23_SPD_TABLE[max(self.SONG_PLUS_23_SPD_TABLE)] + (spd_byte - max(self.SONG_PLUS_23_SPD_TABLE)) * 0.5)
            if abs(v_ego) < 1.0:
                vrel = 0.0  # 低速/停稳归零: 前后都停时相对速度应≈0
            self._mrr23_tracks[child_id] = (now, dist, yrel, vrel)
            updated = True
        # 清理超时目标 (轮询~100Hz, 0.5s 未更新视为目标消失)
        # 必须放在 not updated return 之前: 雷达断流(无0x374帧)时也执行, 防 _mrr23_tracks 内存残留
        expired = [cid for cid, (lt, *_rest) in self._mrr23_tracks.items() if now - lt > self._mrr23_track_timeout]
        for cid in expired:
            self._mrr23_tracks.pop(cid, None)
            self._pts_cache.pop(cid, None)  # 防御: 与 update() 的 _pts_cache.clear() 配合, 断流时同步清理

        if not updated:
            return False

        # 输出所有最近有效目标 (累积多目标)
        for child_id, (lt, dist, yrel, vrel) in self._mrr23_tracks.items():
            track_id = child_id  # ChildID 直接作 trackId (唯一且稳定)
            if track_id not in self._pts_cache:
                self._pts_cache[track_id] = structs.RadarData.RadarPoint()
                self._pts_cache[track_id].trackId = track_id
            self._pts_not_seen[track_id] = NOT_SEEN_INIT
            pt = self._pts_cache[track_id]
            pt.dRel = dist
            pt.yRel = yrel
            pt.vRel = vrel
            pass  # vLead: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # aRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # yvRel: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
            pass  # measured: 现代 RadarPoint 无此字段 (dRel/yRel/vRel/trackId only)
        return True

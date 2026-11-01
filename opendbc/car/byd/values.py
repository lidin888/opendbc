from dataclasses import dataclass, field
from enum import IntFlag
from opendbc.car import Bus, DbcDict, PlatformConfig, Platforms, CarSpecs
from opendbc.car.structs import CarParams
from opendbc.car.docs_definitions import CarHarness, CarDocs, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu

class CarControllerParams:
  STEER_MAX = 300
  STEER_DELTA_UP = 7
  STEER_DELTA_DOWN = 10

  STEER_DRIVER_ALLOWANCE = 68
  STEER_DRIVER_MULTIPLIER = 3
  STEER_DRIVER_FACTOR = 1
  # 2026-08-13: 驾驶员与OP反向对拉退避锁时长(50Hz帧数)。seg27实测: 驾驶员持续施力
  # drv=+50~+84 vs OP 输出 -99~-151 对拉5s → EPS TorqueFailed。触发反向保护后锁零
  # 扭矩0.5s, 期间 softstart 从0重新爬升, 防止恢复大扭矩再次对拉。
  DRIVER_BACKOFF_FRAMES = 25
  STEER_ERROR_MAX = 50

  # 低速停车 EPS 静止扭矩上限保护（v9-BYD seg52 复现场）
  # BYD EPS 在车近乎静止(vEgo<STEER_LOW_SPEED_V)时最大可承受电机扭矩约60,
  # 超过即永久 TorqueFailed(需重启)。车在动时 EPS 有路面阻尼, 扭矩到 -100+ 都安全。
  # 此处仅在 vEgo<0.6 m/s 时把最终下发扭矩限到 ±STEER_LOW_SPEED_MAX(40),
  # 对应最坏 eps~45(<60 上限, 留~15 余量)。余量充足, 仍保留足够低速回正手感,
  # 经 seg52 回放验证不触发锁死。不影响过弯/行驶手感(只限停车段)。
  STEER_LOW_SPEED_V = 0.6      # m/s, 进入静止保护的车速阈值
  STEER_LOW_SPEED_MAX = 40     # raw 扭矩单位(= motor torque), 对应最坏eps~45(留~15余量)

  STEER_STEP = 2  #100/2=50hz
  STEER_SOFTSTART_STEP = 6 # 20ms(50Hz) * 300 / 6 = 1000ms. This means the clip ceiling will be increased to 300 in 1000ms

  ACC_STEP = 2    #50hz

  ACCEL_MAX = 2.0
  ACCEL_MIN = -3.5

  K_DASHSPEED = 0.0719088 #convert pulse to kph

  USE_STEERING_SPEED_LIMITER = False

  # op long control
  K_accel_jerk_upper = 0.1   # ⚠️ 2026-10-08 起已停用 (accel>=0 分支改用 K_jerk_up_*; 全仓无引用, 勿再启用)
  K_accel_jerk_lower = 0.5
  K_jerk_xp =            [   4,   10,   20,   40,   80]  # meters
  K_jerk_base_lower_fp = [-2.3, -1.8, -1.4, -1.0, -0.4]
  K_jerk_base_upper_fp = [ 0.8,  0.7,  0.6,  0.3,  0.2]

  # ---- 泄压(加速方向) jerk 上限: 2026-10-08 门槛式放宽, 根治"前车提速走了我们还在刹车" ----
  # 起因 (实车 .comma(7) route29 seg17, 全程只有 t=50~60s 这 10s OP 纵向在控, 事件就在其中):
  #   前车 vRel -4.5 → +3.9 (前车加速拉远), dRel 13→18m; 我们 aCmd 56.1s 已转正并升到 +1.75,
  #   但 aEgo 从 -1.59 爬回 0 用了 2.0s、到 +0.67 用了 2.5s —— 请求早就是正的, 车还在减速。
  #   实测有效 jerk = 0.94 m/s^3 ≈ 当时下发的 JerkUpperLimit(0.8→1.0) → **ESC 严格 1:1 跟随,
  #   没有隐藏延迟** ⇒ 拖住我们的就是自己给的斜率上限(旧值恒 0.8, 建压:泄压 ≈ 3:1~4.5:1)。
  # 参照 (jerk 信号位偏移/因子与 byd_han_dmev_2020 完全一致, 数值可比):
  #   原车相机实测恒发 JerkUpper/JerkLower = 0/0 (不限 jerk; 4 批日志 ~52 万帧) → 限幅是我们自加的;
  #   官方车: 现代 SCC clip(jerk*2, 0.5, 5.0) / 大众 ±4.0 / 特斯拉 ±4.9(5.0 会 ACC fault);
  #   同门 BYD 高阶商业包 = clip(accel*6.0/2.5, 0.6, 6.0) (量纲同, 车型 DBC 不同 byd_generic_pt)。
  # 取法: 照抄高阶包的"∝请求 + 地板"形状; 地板取 0.8 (与旧值持平, **任何 accel 下都不比现状慢**):
  #   accel 0.1→0.8 / 0.3→0.8 / 0.5→1.2 / 1.0→2.4 / 1.5→3.6 / 2.0→4.8
  #   (ACCEL_MAX=2.0 → 6.0 这一档实际用不到, 留作余量)
  #   跟车微调区(accel<=0.33)与旧值完全一致 → 零回归; 放开只发生在真加速请求(accel>=0.5)。
  #   (10-08 审核修正: 地板原拟 0.6 会在"请求由负转正的头 0.1~0.2s"比旧值更慢, 恰卡在最需要快
  #    的松刹过渡段 → 提到 0.8 消除该反向风险; 高阶包地板是 0.6, 但其 ComfortBand 语义不同, 不照抄)
  # 注意: ACCEL_SLEW_RATE=0.3/帧@20Hz = 6 m/s^3 是**请求侧**斜率上限, ju 放到 6.0 时它才成为绑定约束。
  # 路试只看三点: ①跟车锯齿/过冲 ②起步与急加速后顿挫 ③深刹回正时间(仿真预期提前~0.5s, 少丢~1.7m/s;
  #   ⚠️ 若 ESC 内部自有 jerk 上限(如 2.0)则收益封顶——找 accel>=1.0 窗口量 aEgo 斜率即可判定)。
  # 回退: 只改 K_jerk_up_max 一个数 (2.0 保守档 / 3.0 中档 / 6.0 高阶包同款)。
  K_jerk_up_gain = 2.4     # ⚠️ 2026-10-09 起停用 (泄压上限改用 K_jerk_up_from_mpc_*)
  K_jerk_up_min  = 1.6     # ⚠️ 停用
  K_jerk_up_max  = 6.0     # ⚠️ 停用
  K_jerk_rel_base_upper = 1.6   # ⚠️ 停用 (曾被 accel<0 分支使用, 现由 K_jerk_up_from_mpc_min 替代)

  # ===========================================================================
  # 泄压(松刹)方向 jerk 上限 —— 2026-10-09 改为**照官方现代 HyundaiJerk 的自适应式**:
  #   opendbc/car/hyundai/carcontroller.py:613-645
  #     jerk_u = min(max(0.5, actuators.jerk * 2.0), 5.0)     # key = MPC 解出的 jerk
  #     jerk_l = min(max(1.0, -actuators.jerk * 2.0), 5.0)    # (非CANFD 分支; CANFD 是 *4.0)
  # **根本差异: key 用 MPC 的意图(jerk), 不用当前命令值(accel)。**
  #   "想松开"发生在 aCmd 仍是负数的时候 —— 那时 MPC.jerk 已经 +1.37~+1.70,
  #   按 accel 查表会取到 accel<0 的值(旧 0.8 / 4ebeaf28c5 的 0.8), 按 jerk 则给 2.7~3.4。
  # 选型依据: 27 个真实事件离线仿真(tools/debug/p7_jerk_ab.py + p8c.py,
  #   指标=命令由负转正 → aEgo 回到 0 的耗时), 相对旧代码的中位改善:
  #     旧代码 1.77s | B=4ebeaf28c5 clip(accel*2.4,0.8,6.0) +0.04s(仅 11/27 事件有效)
  #     | C=B+accel<0→1.6 +1.12s(27/27) | D=官方原式(地板0.5) +0.89s(**4 个事件反而变差**)
  #     | **F=本式 clip(jerk*2,1.6,5.0) +1.21s(27/27)**  ← 选它
  #   F 对 ESC 内部上限不敏感: 封顶 2.0 仍 +1.19s, 封顶 1.5 仍 +1.00s。
  # 地板取 1.6 而非官方的 0.5: 0.5 比旧值 0.8 还紧 → 在"MPC.jerk≈0 或为负"的片段会**反而更慢**
  #   (实测 2 个事件崩到 24.3s / 10.9s), 1.6 消除该风险。
  # 上限 5.0 = 官方同值; 交叉参照 Tesla JERK_LIMIT_MAX=4.9 (5.0 会 ACC fault)。
  # 限制: 只改泄压方向; jerk_lower(刹车多快)一律不动 → 刹车力度与旧行为完全一致。
  # 回退: 把 K_jerk_up_from_mpc_gain 改 0 且 min/max 都改 0.8 = 恢复旧的恒 0.8;
  #       或让 bydcan.acc_cmd 重新用 jerk_base_upper / K_jerk_up_* (三处常量仍在)。
  # 路试判据: 找 aCmd>=1.0 的窗口量 aEgo 斜率 —— 跟着涨=ESC 无内部上限(收益全额);
  #       卡在某个值=ESC 内部封顶(收益封顶)。同时看跟车锯齿/顿挫。
  # ===========================================================================
  K_jerk_up_from_mpc_gain = 2.0   # ju = MPC.jerk * gain
  K_jerk_up_from_mpc_min  = 1.6   # 地板 (不用官方的 0.5, 见上)
  K_jerk_up_from_mpc_max  = 5.0   # 上限 (官方同值; Tesla 4.9 交叉印证)

  def __init__(self, CP):
    pass

#FD to be added later
class BydSafetyFlags(IntFlag):
  HAN_TANG_DMEV = 0x1 #pre 2021 models with veoneer mpc/radar solution
  TANG_DMI = 0x2 #note tang dmi is not tang dm
  SONG_PLUS_DMI = 0x4 #note song pro is similar but not song dmi
  QIN_PLUS_DMI = 0x8
  YUAN_PLUS_DMI_ATTO3 = 0x10 #yuan plus is atto3
  OP_LONG = 0x40 #OP纵向模式: panda controls_allowed 由OP的ACC_CMD驱动, 不依赖原车ACC激活


# 车型报文变体位 (存入 CarParams.flags, carstate 按位分支)。
# 移植自 mypilot 社区包 (高阶版 values.py BydFlags); 位定义必须与数据源一致。
class BydFlags(IntFlag):
  CANFD = 1                        # CAN-FD 平台标记 (解析路径无分支, 仅语义标记)
  ANGLE_CONTROL = 2                # 角度控制车型 (P2 阶段, 当前未接入)
  ALT_INDICATOR = 4                # 转向灯取 TURN_SIGNAL_SWITCH==3/5 (而非 Left/RightIndicator 位)
  SETSPEED_X10 = 8                 # 仪表 SetSpeed 为 x10 编码
  BCM_SEAL = 16                    # 门/安全带来自 660 报文 (2bit 语义: 门!=1开, 带!=2松)
  ALT_ACC_EPS = 32                 # EPS 备用源 (源实现中未消费, 保留语义)
  ALT_ACC_EPS_SEAL = 64            # EPS 来自 0x1FC(508) 报文 (P2 角度车型, 当前未接入)
  ALT_ACC_CURISE_MODE = 128        # 巡航显示变体
  ALT_ACC_ACCON1 = 256             # AccOn1 变体
  ALT_PCM_BTN = 512                # 取消键语义改为 setCruise (而非 cancel)
  ACC_STOP_FIX = 1024              # 跟停修复标记
  ANGLE_DRIVER_OVERRIDE_RELEASE = 2048  # 角度模式驾驶员接管释放 (P2)


@dataclass
class BydCarDocs(CarDocs):
  package: str = "All"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.custom]))
  #todo add docs and harness info

@dataclass
class BydPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {Bus.pt: "byd_han_dmev_2020"})
  #todo add dbc for other models

class CAR(Platforms):
  BYD_HAN_DM_20 = BydPlatformConfig(
    [BydCarDocs("BYD HAN DM 2020")],
    CarSpecs(mass=2080., wheelbase=2.920, steerRatio=16.8, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
  )
  BYD_HAN_EV_20 = BydPlatformConfig(
    [BydCarDocs("BYD HAN EV 2020")],
    CarSpecs(mass=2100., wheelbase=2.959, steerRatio=16.8, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
  )

  #The following parameters are likely be incorrect, developers please fill and fix them.

  BYD_TANG_DM = BydPlatformConfig(
    [BydCarDocs("BYD TANG DM 2019-20")],
    CarSpecs(mass=2250., wheelbase=2.820, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    dbc_dict={Bus.pt: "byd_tang_dm"},
  )

  BYD_TANG_DMI_21 = BydPlatformConfig(
    [BydCarDocs("BYD TANG DMI 2021")],
    CarSpecs(mass=2153., wheelbase=2.820, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_INDICATOR,  # 源 flags=1077 含 bit2: 转向灯走 TURN_SIGNAL_SWITCH
  )

  BYD_SONG_PLUS_DMI_21 = BydPlatformConfig(
    [BydCarDocs("BYD SONG PLUS DMI 2021")],
    CarSpecs(mass=1785., wheelbase=2.765, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_INDICATOR,  # 源 flags=4: 转向灯走 TURN_SIGNAL_SWITCH (实车验证)
  )

  BYD_SONG_PLUS_DMI_22 = BydPlatformConfig(
    [BydCarDocs("BYD SONG PLUS DMI 2022")],
    CarSpecs(mass=1785., wheelbase=2.765, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_INDICATOR,  # 源 flags=564 含 bit2
  )

  BYD_SONG_PLUS_DMI_23 = BydPlatformConfig(
    [BydCarDocs("BYD SONG PLUS DMI 2023")],
    CarSpecs(mass=1785., wheelbase=2.765, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_INDICATOR,  # 同宋Plus平台 (源无此车型, 按 21/22 延续)
  )

  BYD_SONG_PRO_DMI_22 = BydPlatformConfig(
    [BydCarDocs("BYD SONG PRO DMI 2022")],
    CarSpecs(mass=1670., wheelbase=2.712, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
  )

  BYD_QIN_PLUS_DMI_23 = BydPlatformConfig(
    [BydCarDocs("BYD QIN PLUS DMI 2023")],
    CarSpecs(mass=1580., wheelbase=2.718, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
  )

  BYD_YUAN_PLUS_DMI_22 = BydPlatformConfig(
    [BydCarDocs("BYD YUAN PLUS DMI 2022")],
    CarSpecs(mass=1625., wheelbase=2.720, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_INDICATOR,  # 源 flags=565 含 bit2
  )

  # ===== P1 批次 (2026-11-01): mypilot 社区包移植的 7 款扭矩模式车型 =====
  # flags 来自源 values.py 实测值; 规格 (mass/wheelbase) 来自源 CarSpecs。
  # 注意: HAN_DMI_22R / HAN_DMI_25 / QIN_PRO 源指纹库缺 CAN 指纹 → 无法自动识别,
  #       需 "强制车型识别" 手动选择 (见 fingerprints.py 注释)。
  BYD_HAN_DMI_22 = BydPlatformConfig(
    [BydCarDocs("BYD HAN DMI 2022")],
    CarSpecs(mass=2080., wheelbase=2.920, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_PCM_BTN | BydFlags.ALT_ACC_EPS | BydFlags.BCM_SEAL | BydFlags.SETSPEED_X10 | BydFlags.ALT_INDICATOR | BydFlags.CANFD,
  )

  BYD_HAN_DMI_22R = BydPlatformConfig(
    [BydCarDocs("BYD HAN DMI 2022-22")],
    CarSpecs(mass=2080., wheelbase=2.920, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_PCM_BTN | BydFlags.ALT_ACC_CURISE_MODE | BydFlags.ALT_ACC_EPS | BydFlags.BCM_SEAL | BydFlags.SETSPEED_X10 | BydFlags.ALT_INDICATOR | BydFlags.CANFD,
  )

  BYD_HAN_DMI_25 = BydPlatformConfig(
    [BydCarDocs("BYD HAN DMI 2025")],
    CarSpecs(mass=2080., wheelbase=2.920, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_PCM_BTN | BydFlags.ALT_ACC_EPS | BydFlags.BCM_SEAL | BydFlags.ALT_INDICATOR | BydFlags.CANFD,
  )

  BYD_TANG_DMI_24 = BydPlatformConfig(
    [BydCarDocs("BYD TANG DMI 2024")],
    CarSpecs(mass=2250., wheelbase=2.820, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_PCM_BTN | BydFlags.ALT_ACC_EPS | BydFlags.BCM_SEAL | BydFlags.ALT_INDICATOR | BydFlags.CANFD,
  )

  BYD_TANG_DMP_22 = BydPlatformConfig(
    [BydCarDocs("BYD TANG DMP 2022")],
    CarSpecs(mass=2445., wheelbase=2.820, steerRatio=15.0, centerToFrontRatio=0.60, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_PCM_BTN | BydFlags.ALT_ACC_CURISE_MODE | BydFlags.ALT_ACC_EPS | BydFlags.BCM_SEAL | BydFlags.SETSPEED_X10 | BydFlags.ALT_INDICATOR | BydFlags.CANFD,
  )

  BYD_TANG_DMP_23 = BydPlatformConfig(
    [BydCarDocs("BYD TANG DMP 2023")],
    CarSpecs(mass=2445., wheelbase=2.820, steerRatio=15.0, centerToFrontRatio=0.60, tireStiffnessFactor=1.0),
    flags=BydFlags.ALT_PCM_BTN | BydFlags.ALT_ACC_EPS | BydFlags.BCM_SEAL | BydFlags.ALT_INDICATOR | BydFlags.CANFD,
  )

  BYD_QIN_PRO = BydPlatformConfig(
    [BydCarDocs("BYD QIN PRO 2018-20")],
    CarSpecs(mass=2250., wheelbase=2.820, steerRatio=15.0, centerToFrontRatio=0.44, tireStiffnessFactor=1.0),
    flags=0,  # 早期平台, 老协议 (沿用默认解析路径)
  )


class LKASConfig:
  DISABLE = 0
  ALARM = 1
  LKA = 2
  ALARM_AND_LKA = 3

class CanBus:
  ESC = 0
  MRR = 1
  MPC = 2

FW_QUERY_CONFIG = FwQueryConfig(
  fw_version_regex=br"DUMMYDATA",  # 2026-11-01: 匹配 FW_VERSIONS 占位值（过 test_fw_version_format）；实测数据后更新
  requests=[
    Request(
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_REQUEST],
      [StdQueries.MANUFACTURER_SOFTWARE_VERSION_RESPONSE],
      bus=CanBus.ESC,
    ),
  ],
)

# P1 新增车型归属 (safety 侧只看 SEAL(0x20)/OP_LONG(0x40) 位, 平台位仅为历史语义)
PLATFORM_NEWGEN_TORQUE = {CAR.BYD_HAN_DMI_22, CAR.BYD_HAN_DMI_22R, CAR.BYD_HAN_DMI_25, CAR.BYD_TANG_DMI_24,
                          CAR.BYD_TANG_DMP_22, CAR.BYD_TANG_DMP_23, CAR.BYD_QIN_PRO}

PLATFORM_HANTANG_DMEV = {CAR.BYD_HAN_DM_20, CAR.BYD_HAN_EV_20, CAR.BYD_TANG_DM} | PLATFORM_NEWGEN_TORQUE
PLATFORM_TANG_DMI = {CAR.BYD_TANG_DMI_21}
PLATFORM_SONG_PLUS_DMI = {CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22, CAR.BYD_SONG_PLUS_DMI_23, CAR.BYD_SONG_PRO_DMI_22}
PLATFORM_QIN_PLUS_DMI = {CAR.BYD_QIN_PLUS_DMI_23}
PLATFORM_YUAN_PLUS_DMI_ATTO3 = {CAR.BYD_YUAN_PLUS_DMI_22}

# power train canbus is located and accessible in in MPC connector
MPC_ACC_CAR = {CAR.BYD_HAN_DM_20, CAR.BYD_HAN_EV_20, CAR.BYD_TANG_DM}

# power train canbus contains mrr radar info
PT_RADAR_CAR = {CAR.BYD_HAN_DM_20, CAR.BYD_HAN_EV_20, CAR.BYD_TANG_DM, CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22, CAR.BYD_SONG_PLUS_DMI_23}

# MRR 雷达 0x380~0x3FF 直解车型 (V11 实测: 唐DM/汉EV/宋Plus 21/22 全系通用大陆 ARS4xx 同款雷达)
# 汉DM 与汉EV 同平台(PLATFORM_HANTANG_DMEV)推断同款, 未实测待验证。
# 注意: 此集合仅决定雷达直解(radar_interface/carstate 雷达逻辑), 与 is_tang_dm(横向 318 编码)语义无关!
# 宋Plus 23款 (2026-08-14 实测): 雷达为 0x374 RADAR_MRR 单目标报文(原车发 bus2/MPC,
# panda fwd_hook 转发到 bus0/ESC, 2026-08-18 H宋23日志 src=2 直收实证),
# 不走 0x380 直解, 单独归入 RADAR_MRR_CAR。
MRR_DIRECT_DECODE_CAR = {CAR.BYD_TANG_DM, CAR.BYD_HAN_DM_20, CAR.BYD_HAN_EV_20, CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22}

# 宋Plus 走 0x380 博世 MRRevo14 多目标解码 (2026-08-13 破解: 0.0625m/LSB, 多目标)
# 唐DM/汉DM/汉EV 仍用 0x380~0x3FF V11 私有协议 (同一片地址, 布局不同, 不能混用)
SONG_PLUS_380_DECODE_CAR = {CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22}

# 宋Plus 23款: 雷达 = 0x374 RADAR_MRR MUX 多目标报文 (2026-08-19 破解定论, 参考2021汉DBC BO_ 884)
# 帧结构: byte0=Child_ID(完整值, 空闲槽9~13, 真实目标25/26/27/28等) / byte1=横向(偏移-100)
#         / byte2=纵向距离(偏移-100, m, 雷达内部强滤波) / byte3=相对速度档(0~12, 重缩放)
#         / byte5=Target_State/Type(宋23恒0xFC) / byte6高4位=AliveCounter / byte7=CheckSum
# 无 IsValid 位(byte5 恒 0xFC), 有效判定 = ChildID 非空闲槽 + LongDist 在 3~150m + CRC 通过。
# 总线: 原车发 bus2(MPC), panda fwd_hook 转发到 bus0(ESC); radar_interface 走 bus=MPC(2) 读 src=2。
RADAR_MRR_CAR = {CAR.BYD_SONG_PLUS_DMI_23}

# use torque lat control, otherwise use angle mode
# 2026-11-01 P1: 新增 7 款扭矩车型 (源 TORQUE_LAT_CAR 全集一致); 无 siglin 表的走线性扭矩路径
TORQUE_LAT_CAR = {CAR.BYD_HAN_DM_20, CAR.BYD_HAN_EV_20, CAR.BYD_TANG_DM, CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22, CAR.BYD_SONG_PLUS_DMI_23} | PLATFORM_NEWGEN_TORQUE

# use experimental long mode
EXP_LONG_CAR = {CAR.BYD_HAN_DM_20, CAR.BYD_HAN_EV_20, CAR.BYD_TANG_DM, CAR.BYD_SONG_PLUS_DMI_21, CAR.BYD_SONG_PLUS_DMI_22, CAR.BYD_SONG_PLUS_DMI_23} | PLATFORM_NEWGEN_TORQUE

DBC = CAR.create_dbc_map()

if __name__ == "__main__":
  cars = []
  for platform in CAR:
    for doc in platform.config.car_docs:
      cars.append(doc.name)
  cars.sort()
  for c in cars:
    print(c)

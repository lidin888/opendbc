#!/usr/bin/env python3
from math import exp

import numpy as np

from opendbc.car import get_safety_config, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType, LateralAccelFromTorqueCallbackType
from opendbc.car.byd.values import (
  CAR,
  CanBus,
  BydSafetyFlags,
  MPC_ACC_CAR,
  PT_RADAR_CAR,
  TORQUE_LAT_CAR,
  EXP_LONG_CAR,
  PLATFORM_HANTANG_DMEV,
  PLATFORM_TANG_DMI,
  PLATFORM_SONG_PLUS_DMI,
  PLATFORM_QIN_PLUS_DMI,
  PLATFORM_YUAN_PLUS_DMI_ATTO3,
)
from opendbc.car.byd.carcontroller import CarController
from opendbc.car.byd.carstate import CarState
from opendbc.car.byd.radar_interface import RadarInterface

from openpilot.common.params import Params

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter
TransmissionType = structs.CarParams.TransmissionType
NetworkLocation = structs.CarParams.NetworkLocation

NON_LINEAR_TORQUE_PARAMS = {
  CAR.BYD_HAN_DM_20: [1.807, 1.674, 0.04],
  CAR.BYD_TANG_DM: [1.807, 1.674, 0.04],
  CAR.BYD_HAN_EV_20: [1.807, 1.674, 0.04],
  CAR.BYD_SONG_PLUS_DMI_21: [1.807, 1.674, 0.04],
  CAR.BYD_SONG_PLUS_DMI_22: [1.807, 1.674, 0.04],
  CAR.BYD_SONG_PLUS_DMI_23: [1.335, 2.197, 0.0282],
}

# 雷达可用性按车型集合判定 (2026-11-01): 原 BYD_RADAR 环境变量在设备侧从未设置,
# 导致 radarUnavailable 恒 True (该字段 controls 不消费, 但语义应正确)。
# PT_RADAR_CAR 的雷达解析均有实车依据; 非该集合车型 = 无雷达数据。


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  # BYD stock latAccelFactor (torque_data/params.toml [2.2, 2.5, 0.145] 首值)。
  # 用于把 torque_params.latAccelFactor 旋钮(手调/Self-Tune 自调谐)映射到 siglin 曲线增益。
  STOCK_LAT_ACCEL_FACTOR = 2.2

  @staticmethod
  def _siglin_curve(lateral_acceleration: float, abc: tuple[float, float, float]) -> float:
    # BYD 非线性扭矩模型 (ABC) 曲线本体: torque = sig(latAccel·a)·b + latAccel·c
    def sig(val):
      # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
      if val >= 0:
        return 1 / (1 + exp(-val)) - 0.5
      else:
        z = exp(val)
        return z / (1 + z) - 0.5

    # The "lat_accel vs torque" relationship is assumed to be the sum of "sigmoid + linear" curves
    # An important thing to consider is that the slope at 0 should be > 0 (ideally >1)
    # This has big effect on the stability about 0 (noise when going straight)
    a, b, c = abc
    return float((sig(lateral_acceleration * a) * b) + (lateral_acceleration * c))

  def torque_from_lateral_accel_siglin(
    self,
    lateral_acceleration: float,
    torque_params: structs.CarParams.LateralTorqueTuning,
  ) -> float:
    # friction 由 SP 的 latcontrol_torque ff 侧处理, 此处只回曲线 (否则双重摩擦)。
    # 曲线固定为 v9 实车标定值, 不随 latAccelFactor 缩放 (2026-11-03 去掉 scale:
    # 避免"调 factor 却在缩放曲线"的语义混乱; 摩擦仍由 torque_params.friction 单独调节)。
    abc = NON_LINEAR_TORQUE_PARAMS[self.CP.carFingerprint]
    return self._siglin_curve(lateral_acceleration, abc)

  @staticmethod
  def _siglin_enabled() -> bool:
    # 只有自调校(LiveTorqueParamsToggle)切线性 —— 它学的是线性模型三值, 必须进线性;
    # 自定义调校(CustomTorqueParams)不切: 曲线已固定(见 torque_from_lateral_accel_siglin),
    # 它的 friction 滑条在 siglin 下照样生效, 只有 factor 滑条无效果 (2026-11-03)。
    return not Params().get_bool("LiveTorqueParamsToggle")

  def _use_siglin(self) -> bool:
    return self.CP.carFingerprint in NON_LINEAR_TORQUE_PARAMS and self._siglin_enabled()

  def lateral_accel_from_torque(self) -> LateralAccelFromTorqueCallbackType:
    if self._use_siglin():
      # GM 同款查表反演 (2026-11-01 补: 此前缺反函数 → update_limits 的 PID 限幅
      # 按线性反算, 与 siglin 前向不一致)。曲线单调(a,b,c>0), np.interp 反查。
      abc = NON_LINEAR_TORQUE_PARAMS[self.CP.carFingerprint]
      lataccel_grid = np.arange(-5.0, 5.0, 0.01)
      curve_grid = [self._siglin_curve(x, abc) for x in lataccel_grid]

      def lateral_accel_from_torque_siglin(torque: float, torque_params: structs.CarParams.LateralTorqueTuning) -> float:
        return float(np.interp(torque, curve_grid, lataccel_grid))
      return lateral_accel_from_torque_siglin
    else:
      return self.lateral_accel_from_torque_linear

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    if self._use_siglin():
      return self.torque_from_lateral_accel_siglin
    else:
      return self.torque_from_lateral_accel_linear

  @staticmethod
  def init(CP, CP_SP, can_recv, can_send):
    # 原车仪表车距 4 档 (SetDistance 1~4 = 1BAR~4BAR, 汉/唐 DBC VAL_ 铁证)。
    # 基类默认 LongitudinalPersonalityMax=3 → OP personality 只有 3 档,
    # leadDistanceBars=raw+1 ∈ [1,3], 0x32D 注入永远到不了 4BAR。
    # 覆盖为 4: personality raw∈[0,3] (cereal aggressive/standard/relaxed/moreRelaxed),
    # bars=1~4 与原车一一对应 (aggressive=1BAR最近 ~ moreRelaxed=4BAR最远) (2026-08-19)。
    Params().put('LongitudinalPersonalityMax', 4)

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    # BYD OP 纵向自带按钮拦截/注入(_get_button_injection), 与 SP 的 ICBM 自动按键冲突, 禁用。
    ret.intelligentCruiseButtonManagementAvailable = False
    # pcmCruiseSpeed 必须与 openpilotLongitudinalControl 对称 (2026-11-03 修):
    # 之前写死 False → controlsd 的 longActive 变成 `enabled or not pcmCruiseSpeed` 恒 True
    # → carcontroller 的按钮拦截分支命中(清零 SET/RES/Cancel) → 没开 OP 纵向时原车收不到
    # 按钮, 仪表车速加不上去 (v9 无此字段, longActive 只看 openpilotLongitudinalControl)。
    # 没开 OP 纵向时速度该由原车管 → True; 开 OP 纵向时 OP 自己管速度 → False。
    ret.pcmCruiseSpeed = not stock_cp.openpilotLongitudinalControl
    return ret

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "byd"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.byd)]

    # 有 siglin 表且走曲线模式时, jerk 的扭矩空间换算是线性的, 与曲线不一致 → 自动关掉;
    # 只有线性模式(开了自调校/自定义调校)下 jerk 才可用 (2026-11-03)。
    if candidate in NON_LINEAR_TORQUE_PARAMS and CarInterface._siglin_enabled():
      Params().put_bool("LateralJerkTorqueController", False)

    ret.dashcamOnly = False
    ret.radarUnavailable = candidate not in PT_RADAR_CAR

    ret.minEnableSpeed = -1.0
    ret.enableBsm = 0x418 in fingerprint[CanBus.ESC]
    ret.transmissionType = TransmissionType.direct

    ret.minEnableSpeed = -1.0
    ret.minSteerSpeed = 0.1 * CV.KPH_TO_MS

    ret.steerActuatorDelay = 0.05
    ret.steerLimitTimer = 0.4

    if candidate in PLATFORM_HANTANG_DMEV:
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.HAN_TANG_DMEV.value
    elif candidate in PLATFORM_TANG_DMI:
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.TANG_DMI.value
    elif candidate in PLATFORM_SONG_PLUS_DMI:
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.SONG_PLUS_DMI.value
    elif candidate in PLATFORM_QIN_PLUS_DMI:
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.QIN_PLUS_DMI.value
    elif candidate in PLATFORM_YUAN_PLUS_DMI_ATTO3:
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.YUAN_PLUS_DMI_ATTO3.value

    if candidate in MPC_ACC_CAR:
      ret.networkLocation = NetworkLocation.fwdCamera

    use_torque_lat = candidate in TORQUE_LAT_CAR

    if use_torque_lat:
      CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)
    else:
      ret.lateralTuning.init("pid")
      ret.lateralTuning.pid.kpBP, ret.lateralTuning.pid.kiBP = [[8.3, 27.8], [8.3, 27.8]]
      ret.lateralTuning.pid.kpV, ret.lateralTuning.pid.kiV = [[0.6, 0.3], [0.2, 0.1]]
      ret.lateralTuning.pid.kf = 0.000072

    use_experimental_long = candidate in EXP_LONG_CAR

    ret.alphaLongitudinalAvailable = use_experimental_long
    ret.openpilotLongitudinalControl = alpha_long and ret.alphaLongitudinalAvailable
    # OP纵向(pcmCruise=False): OP用按钮自己启停, 原车ACC保持待机(AccControlActive=0),
    # 相机不进入指令-响应监督 → 不报ACC_ERROR (2026-08-15 日志验证: OP纵向时原车ACC
    # 被注入SET激活后才报错; 待机时相机AccCtrl=0不监督)。
    # 非OP纵向(雷达纵向): pcmCruise=True, 跟随原车ACC, 行为不变。
    ret.pcmCruise = not ret.openpilotLongitudinalControl
    # OP纵向模式传给panda: controls_allowed由OP的ACC_CMD驱动, 不依赖原车ACC激活
    # (2026-08-16 实测: 全程拦截原车ACC待机后 pcm_cruise_check 永远清controls_allowed
    #  → OP激活2s后 controlsMismatch 控制模块不匹配 立即接管)。
    if ret.openpilotLongitudinalControl:
      ret.safetyConfigs[0].safetyParam |= BydSafetyFlags.OP_LONG.value

    # sptn: kpBP/kpV 在 LongitudinalPIDTuning 已 deprecated 且 longcontrol 硬编码 kp=0,
    # 只设 live 字段 ki (v9-byd 的 kpV=1.5 属阶段3 longcontrol 对拍项)。
    ret.longitudinalTuning.kiBP = [0.0]
    ret.longitudinalTuning.kiV = [0.3]

    # model specific parameters
    # Todo: Developers please fill or add more models.
    if candidate in (
      CAR.BYD_HAN_DM_20,
      CAR.BYD_HAN_EV_20,
      CAR.BYD_TANG_DM,
      CAR.BYD_SONG_PLUS_DMI_21,
      CAR.BYD_TANG_DMI_21,
      CAR.BYD_SONG_PLUS_DMI_22,
      CAR.BYD_SONG_PLUS_DMI_23,
      CAR.BYD_SONG_PRO_DMI_22,
      CAR.BYD_QIN_PLUS_DMI_23,
      CAR.BYD_YUAN_PLUS_DMI_22,
    ):
      ret.minSteerSpeed = 0
      ret.autoResumeSng = True
      # 起步 accel 1.2 (2026-09-02 原车对照): 原车 ACC 起步 accel 渐进爬峰 0.7~1.75(均~1.2+),
      # OP 旧值 0.8 起步请求偏弱, ESC 慢热期蠕行更明显。1.2 = 原车峰值中位, 仍低于 ACCEL_MAX。
      # sptn: startingState/startAccel/vEgoStarting/vEgoStopping 已进 capnp deprecated 组且
      # 全链无消费者 (起步语义在 stopping_controller), 阶段3 对拍时在此落点等效实现。
      ret.stopAccel = -0.5
      ret.longitudinalActuatorDelay = 0.5
    else:
      ret.dashcamOnly = True

    return ret


# byd tuning suggestions
# torque mode (linear):
#   -Start with P = 1.0 and I = 0, FRICTION = 0, change the LAT_ACCEL_FACTOR until
#    vehicle can just about to go thought a medium curve at medium speed(say 50kph)
#   -Find a straight road and tune FRICTION until car stay in same position of lane,
#    don't worry if car hugs to left or right.
#   -I = 0.1 (default) then try. Keep tuning all values until you get a good enough
#    result. Commaai's PI loop have a funny anti-wind up method.
#   -Add the steering angle speed (angle slew rate) to stop the control loop from
#    tuning wheel too sharp.

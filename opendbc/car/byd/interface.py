#!/usr/bin/env python3
from math import exp

from opendbc.car import get_safety_config, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType
from opendbc.car.byd.values import (
  CAR,
  CanBus,
  BydSafetyFlags,
  MPC_ACC_CAR,
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

import os

BYD_RADAR = os.getenv("BYD_RADAR") is not None


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  def torque_from_lateral_accel_siglin(
    self,
    lateral_acceleration: float,
    torque_params: structs.CarParams.LateralTorqueTuning,
  ) -> float:
    # BYD 非线性扭矩模型 (ABC): torque = sig(latAccel·a)·b + latAccel·c
    # friction 由 SP 的 latcontrol_torque ff 侧处理, 此处只回曲线 (否则双重摩擦)。
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
    non_linear_torque_params = NON_LINEAR_TORQUE_PARAMS.get(self.CP.carFingerprint)
    if non_linear_torque_params is None:
      raise ValueError(f"NON_LINEAR_TORQUE_PARAMS missing for {self.CP.carFingerprint}")
    a, b, c = non_linear_torque_params
    steer_torque = (sig(lateral_acceleration * a) * b) + (lateral_acceleration * c)
    return float(steer_torque)

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    if self.CP.carFingerprint in NON_LINEAR_TORQUE_PARAMS:
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
    Params().put('LongitudinalPersonalityMax', "4")

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    # BYD OP 纵向自带按钮拦截/注入(_get_button_injection), 与 SP 的 ICBM 自动按键冲突, 禁用。
    ret.intelligentCruiseButtonManagementAvailable = False
    ret.pcmCruiseSpeed = False
    return ret

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "byd"
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.byd)]

    ret.dashcamOnly = False
    # disable simple pt radar due to mpc solver issue in official OP. It works with carrot/sunny/forg.
    if BYD_RADAR:
      ret.radarUnavailable = False
    else:
      ret.radarUnavailable = True  # candidate not in PT_RADAR_CAR

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

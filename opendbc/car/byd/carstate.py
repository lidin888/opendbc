import copy
import numpy as np

# -----disabled for pytests-----
# from datetime import datetime, timedelta
# import subprocess
# from openpilot.common.time_helpers import system_time_valid
# from openpilot.common.swaglog import cloudlog

from opendbc.can import CANDefine, CANParser

from opendbc.car.common.conversions import Conversions as CV

# from opendbc.car.common.numpy_fast import mean
from opendbc.car import Bus, create_button_events, structs
from opendbc.car.interfaces import CarStateBase
from opendbc.car.byd.debug_log import acc_debug_write
from opendbc.car.byd.values import DBC, CanBus, LKASConfig, CarControllerParams, CAR, PLATFORM_SONG_PLUS_DMI, MRR_DIRECT_DECODE_CAR, RADAR_MRR_CAR, PT_RADAR_CAR, BydFlags

# 雷达数据源开关 (2026-11-01): 原为 BYD_RADAR 环境变量, 改为按车型集合判断——
# PT_RADAR_CAR 的雷达解析路径均已实车验证 (0x380 直解 / 0x374 单目标), 不再依赖
# 外部 env (设备上从未设置, 导致 mrr_leading_dist 恒 0 → jerk 恒取最紧档 0.8)。

ButtonType = structs.CarState.ButtonEvent.Type


class CarState(CarStateBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)

    can_define = CANDefine(DBC[CP.carFingerprint][Bus.pt])

    self.shifter_values = can_define.dv["DRIVE_STATE"]["Gear"]

    self.speed_kph = 0

    self.mpc_lkas_config = 0

    self.acc_hud_adas_counter = 0
    self.acc_mpc_state_counter = 0
    self.acc_cmd_counter = 0

    self.eps_warning = False

    self.acc_active_last = False
    self.low_speed_alert = False
    self.lkas_allowed_speed = False

    # 0x318 byte0 低2位: 唐DM 用 2bit LKAS_State 枚举(0=NotReady/1=Ready/2=Active/3=TemporaryFail)。
    # 宋Plus(共用 byd_han_dmev_2020.dbc) 的 EPS 同样按 2bit 枚举发送, 但 DBC 把低2位拆成
    # LKAS_Prepared(bit0)+CruiseActivated(bit1) 两个 1bit 信号, 需合成回 2bit 枚举。
    # 汉DM/汉EV 的 EPS 按独立 bit 语义(bit0=prepared), 保持原逻辑不变(2026-08-08 确认汉EV 正常)。
    self.is_tang_dm = CP.carFingerprint == CAR.BYD_TANG_DM
    self.is_mrr_direct = CP.carFingerprint in MRR_DIRECT_DECODE_CAR  # 雷达 0x380 直解车型 (与横向 is_tang_dm 无关)
    self.is_radar_mrr = CP.carFingerprint in RADAR_MRR_CAR  # 宋Plus 23款: 0x374 单目标雷达 (无 IsValid 位)
    self.is_song_plus_dmi = CP.carFingerprint in PLATFORM_SONG_PLUS_DMI
    self.has_pt_radar = CP.carFingerprint in PT_RADAR_CAR  # 原车雷达数据可用 (见模块头注释)
    self.lkas_state = 0  # 0=NotReady/1=Ready/2=Active/3=TemporaryFail
    self.lkas_prepared = False  # 318, EPS to OP
    self.acc_state = 0
    self.adas_set_dist = 0

    self.mpc_laks_output = 0
    self.mpc_laks_active = False
    self.mpc_laks_reqprepare = False

    self.acc_enhance_request = False
    self.acc_enhance_accel = 0.0
    self.acc_enhance_scene = 0
    self.acc_enhance_ending = False

    # self.prev_angle = 0

    self.cam_lkas = None
    self.cam_acc = None
    self.esc_eps = None

    self.mrr_leading_dist = 0
    self.acc_has_lead = False
    self.acc_leading_dist = 0
    self.acc_close_warning = False

    self.btn_acc_cancel = 0
    self.btn_acc_set_reset = 0
    self.btn_acc_dist_inc = 0
    self.btn_acc_dist_dec = 0

    self.pcm_buttons_vals = {}

  def update(self, can_parsers) -> tuple[structs.CarState, structs.CarStateSP]:
    cp = can_parsers[Bus.pt]
    cp_cam = can_parsers[Bus.cam]

    ret = structs.CarState()
    ret_sp = structs.CarStateSP()

    if self.is_tang_dm:
      # 0x318 bits0~1 = 2bit LKAS_State: 0 NotReady / 1 Ready / 2 Active / 3 TemporaryFail
      self.lkas_state = int(cp.vl["ACC_EPS_STATE"]["LKAS_State"])
      self.lkas_prepared = self.lkas_state in (1, 2)  # Ready / Active
    elif self.is_song_plus_dmi:
      # 宋Plus: 合成低2位枚举, 与唐DM 的 LKAS_State 完全等价。
      # 背景(2026-08-08 LKAS 分析): 宋Plus 后装L2 报 TorqueFailed 锁死的根因是
      # 旧逻辑只读 LKAS_Prepared(bit0), 正常 Active(2) 时 bit0=0 导致 lkas_prepared 恒 False;
      # 更致命的是 EPS 预警 TemporaryFail(3) 时 bit0=1, 被误读为 lkas_prepared=True(以为已就绪),
      # OP 不退避继续顶扭矩 76→112 → 升级 TorqueFailed 永久锁死(需重启车)。
      # 合成 2bit 后, 3=TempFail 不再算 prepared, 配合 steerFaultTemporary 立即平滑退避。
      # 注: 仅宋Plus 需要此逻辑; 汉DM/汉EV 的 EPS 按独立 bit 语义, 保持原逻辑(用户确认汉EV 正常)。
      self.lkas_state = int(cp.vl["ACC_EPS_STATE"]["LKAS_Prepared"]) | (int(cp.vl["ACC_EPS_STATE"]["CruiseActivated"]) << 1)
      self.lkas_prepared = self.lkas_state in (1, 2)  # Ready/Active 算 prepared; 0 NotReady/3 TemporaryFail 不算
    else:
      # 汉DM/汉EV: 2026-11-01 汉EV 实车日志(20汉EV/.comma, 3.3万帧)证明 0x318 低2位
      # **同为 2bit 枚举**: Active(值2) 时 bit0(LKAS_Prepared)=0, bit1(CruiseActivated)=1;
      # Ready 仅 5 帧(bit0=1)。若只读 bit0 → lkas_prepared 恒 False → 横向永远无法激活。
      # 与宋Plus 相同处理(合成 bit1:bit0)。对"bit0=1 且 bit1=0"的固件行为不变(值1→prepared)。
      self.lkas_state = int(cp.vl["ACC_EPS_STATE"]["LKAS_Prepared"]) | (int(cp.vl["ACC_EPS_STATE"]["CruiseActivated"]) << 1)
      self.lkas_prepared = self.lkas_state in (1, 2)  # Ready/Active 算 prepared; 0 NotReady/3 TemporaryFail 不算

    self.mpc_lkas_config = int(cp_cam.vl["ACC_MPC_STATE"]["LKAS_Config"])
    lkas_config_isAccOn = self.mpc_lkas_config != LKASConfig.DISABLE
    lkas_isMainSwOn = bool(cp.vl["PCM_BUTTONS"]["BTN_TOGGLE_ACC_OnOff"])

    lkas_hud_AccOn1 = bool(cp_cam.vl["ACC_HUD_ADAS"]["AccOn1"])
    self.acc_state = cp_cam.vl["ACC_HUD_ADAS"]["AccState"]
    self.adas_set_dist = cp_cam.vl["ACC_HUD_ADAS"]["SetDistance"]
    # 原车ACC自报的前车状态(HasLead/LeadingDistance/CloseWarning), 用于前车过滤替代 gap=setKph-vEgo 猜测。
    # 原车ACC用自己的雷达/视觉判断是否在跟车, 比 gap 判断可靠得多(爬坡/缓行时gap大但无前车)。
    # 实车rlog验证(2026-08-01): HasLead 可靠, 跟车段=1(如0000000e--14 HasLead=1:5828), 自由巡航=0。
    self.acc_has_lead = bool(cp_cam.vl["ACC_HUD_ADAS"]["HasLead"])
    self.acc_leading_dist = int(cp_cam.vl["ACC_HUD_ADAS"]["LeadingDistance"])
    self.acc_close_warning = bool(cp_cam.vl["ACC_HUD_ADAS"]["CloseWarning"])
    # ===== 诊断日志: 原车ACC状态变化(放在 acc_has_lead 赋值后, 打印当前帧值) =====
    _notify = int(cp_cam.vl["ACC_HUD_ADAS"]["Notify"])
    if _notify != getattr(self, '_prev_notify', -1) or int(self.acc_state) != getattr(self, '_prev_acc_state', -1):
      acc_debug_write(f"DBG_ACC 813变化: AccState={int(self.acc_state)} Notify={_notify} HasLead={int(self.acc_has_lead)} "
                      f"vEgo={ret.vEgo*3.6:.1f}kmh cruiseEn={ret.cruiseState.enabled}\n")
      self._prev_notify = _notify
      self._prev_acc_state = int(self.acc_state)

    prev_btn_acc_cancel = self.btn_acc_cancel
    prev_btn_acc_set_reset = self.btn_acc_set_reset
    prev_btn_acc_dist_inc = self.btn_acc_dist_inc
    prev_btn_acc_dist_dec = self.btn_acc_dist_dec

    self.btn_acc_cancel = cp.vl["PCM_BUTTONS"]["BTN_AccCancel"]
    self.btn_acc_set_reset = cp.vl["PCM_BUTTONS"]["BTN_AccUpDown_Cmd"]
    self.btn_acc_dist_inc = cp.vl["PCM_BUTTONS"]["BTN_AccDistanceIncrease"]
    self.btn_acc_dist_dec = cp.vl["PCM_BUTTONS"]["BTN_AccDistanceDecrease"]

    self.pcm_buttons_vals = {
      "SETME_1": cp.vl["PCM_BUTTONS"]["SETME_1"],
      "BTN_AccUpDown_Cmd": cp.vl["PCM_BUTTONS"]["BTN_AccUpDown_Cmd"],
      "BTN_AccCancel": cp.vl["PCM_BUTTONS"]["BTN_AccCancel"],
      "BTN_TOGGLE_ACC_OnOff": cp.vl["PCM_BUTTONS"]["BTN_TOGGLE_ACC_OnOff"],
      "SETME2_1": cp.vl["PCM_BUTTONS"]["SETME2_1"],
      "BTN_AccDistanceDecrease": cp.vl["PCM_BUTTONS"]["BTN_AccDistanceDecrease"],
      "BTN_AccDistanceIncrease": cp.vl["PCM_BUTTONS"]["BTN_AccDistanceIncrease"],
      "Counter": int(cp.vl["PCM_BUTTONS"]["Counter"]),
    }

    # use wheels averages if you like
    # ret.wheelSpeeds = self.get_wheel_speeds(
    #     cp.vl["IPB"]["WheelSpeed_FL"],
    #     cp.vl["IPB"]["WheelSpeed_FR"],
    #     cp.vl["IPB"]["WheelSpeed_RL"],
    #     cp.vl["IPB"]["WheelSpeed_RR"],
    # )
    # speed_kph = mean([ret.wheelSpeeds.fl, ret.wheelSpeeds.fr, ret.wheelSpeeds.rl, ret.wheelSpeeds.rr])

    # use dash speedo as speed reference
    speed_raw = int(cp.vl["CARSPEED"]["CarDisplaySpeed"])
    self.speed_kph = speed_raw * CarControllerParams.K_DASHSPEED

    ret.vEgoRaw = float(self.speed_kph * CV.KPH_TO_MS)  # KPH to m/s
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    # ret.aEgo = cp.vl["AXAY"]["Ax"] - cp.vl["AXAY"]["AxOffset"]  # 纵向加速度 (来自底盘传感器)
    # ret.aEgoY = cp.vl["AXAY"]["AY"] - cp.vl["AXAY"]["AYOffset"]  # 横向加速度 (来自底盘传感器)

    # ret.yawRate = cp.vl["YAW_RATE"]["YawRate"] - cp.vl["YAW_RATE"]["YawRateOffset"]

    ret.standstill = speed_raw == 0

    if self.CP.minSteerSpeed > 0:
      if self.speed_kph > 0.5:
        self.lkas_allowed_speed = True
      elif self.speed_kph < 0.1:
        self.lkas_allowed_speed = False
    else:
      self.lkas_allowed_speed = True

    can_gear = int(cp.vl["DRIVE_STATE"]["Gear"])
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(can_gear, None))

    ret.genericToggle = bool(cp.vl["STALKS"]["HeadLight"])
    if self.CP.enableBsm:
      ret.leftBlindspot = bool(cp.vl["BSD_RADAR"]["LEFT_APPROACH"])
      ret.rightBlindspot = bool(cp.vl["BSD_RADAR"]["RIGHT_APPROACH"])

    # 转向灯: 车型差异由 DBC 承担, 代码统一读 LeftIndicator/RightIndicator (v9/VV9 原结构)。
    #   宋系 → byd_han_dmev_2020.dbc      (Left 37|1 / Right 38|1, 即 TURN_SIGNAL_SWITCH 的 bit1/bit2)
    #   汉唐等 → byd_han_dmev_2020_han.dbc (Left 4|1  / Right 5|1, 原厂 DBC 未定义的空闲位镜像;
    #            20汉EV 实车日志 3.3万帧证实: 左转 bit4=1 / 右转 bit5=1, 36|3 恒 0)
    # 两种位定义下读位均兼容短拨(值2=左短拨/值4=右短拨)与长拨(3=左/5=右)。
    ret.leftBlinker = bool(cp.vl["STALKS"]["LeftIndicator"])
    ret.rightBlinker = bool(cp.vl["STALKS"]["RightIndicator"])

    ret.steeringAngleOffsetDeg = 0
    ret.steeringAngleDeg = cp.vl["EPS"]["SteeringAngle"]

    self.steeringRateDegAbs = cp.vl["EPS"]["SteeringAngleRate"]
    ret.steeringRateDeg = self.steeringRateDegAbs

    ret.steeringTorque = cp.vl["ACC_EPS_STATE"]["SteerDriverTorque"]
    ret.steeringTorqueEps = cp.vl["ACC_EPS_STATE"]["MainTorque"]
    self.eps_warning = bool(cp.vl["ACC_EPS_STATE"]["SteerWarning"])  # Todo: some firmware have SteerWarning field asserted.
    self.steer_error_code = int(cp.vl["ACC_EPS_STATE"]["SteerErrorCode"])
    self.eps_state_counter = int(cp.vl["ACC_EPS_STATE"]["Counter"])

    ret.steeringPressed = self.update_steering_pressed(abs(ret.steeringTorque) > 59, 5)

    ret.parkingBrake = cp.vl["EPB"]["EPB_ActiveFlag"] == 1

    # 注意: BrakePedal factor=0.01 → 解析值=0.09~0.33。int() 会截断成0 → brakePressed恒False!
    # (2026-08-16 .comma(3)验证: 踩刹车PED byte1=9~33, OP读brake=0 → 不退出 → controlsMismatch)
    # 普通模式靠panda brake_pressed清ca退出没暴露; OP纵向靠OP自己检测刹车, 必须保留float。
    # sptn: CarState.brake 字段已移除, float 语义保留在局部变量 (brakePressed 判定不被截断)。
    brake_val = cp.vl["PEDAL"]["BrakePedal"]
    ret.brakePressed = brake_val > 0

    if self.CP.flags & BydFlags.BCM_SEAL:
      # 新平台门/安全带: 660 报文的 2bit 状态 (门==1 关闭 / !=1 开; 带==2 系好 / !=2 松开)
      door_msg = cp.vl["BELT"]
      ret.doorOpen = any([door_msg["DoorStateFL"] != 1, door_msg["DoorStateFR"] != 1,
                          door_msg["DoorStateRL"] != 1, door_msg["DoorStateRR"] != 1])
      ret.seatbeltUnlatched = door_msg["SeatBeltState"] != 2
    else:
      ret.seatbeltUnlatched = cp.vl["BELT"]["SeatBeat"] != 2  # 1:unfasten, 2:fasten
      ret.doorOpen = any([cp.vl["BCM"]["FrontLeftDoor"], cp.vl["BCM"]["FrontRightDoor"],
                          cp.vl["BCM"]["RearLeftDoor"], cp.vl["BCM"]["RearRightDoor"]])

    # 同样: AcceleratorPedal factor=0.01, int()截断成0 → gasPressed恒False。保留float。
    gas_val = cp.vl["PEDAL"]["AcceleratorPedal"]
    ret.gasPressed = gas_val > 0

    ret.cruiseState.available = lkas_isMainSwOn and lkas_config_isAccOn and lkas_hud_AccOn1
    ret.cruiseState.enabled = self.acc_state in (3, 5)
    ret.cruiseState.standstill = ret.standstill
    set_speed_kph = cp_cam.vl["ACC_HUD_ADAS"]["SetSpeed"]
    if self.CP.flags & BydFlags.SETSPEED_X10:
      set_speed_kph *= 10  # 新平台仪表 SetSpeed 为 x10 编码
    ret.cruiseState.speed = set_speed_kph * CV.KPH_TO_MS

    ret.latEnabled = lkas_isMainSwOn and self.lkas_allowed_speed

    # Todo: some firmware have these fields asserted.
    if self.is_tang_dm:
      # DM 固件 EPS 持续置位 SteerWarning=1 / SteerErrorCode=7 / TorqueFailed=0, 画面异常态而非真实故障;
      # 仅当 SteerWarning 持续非空闲错误(=7)时方才定为临时转向故障;
      # LKAS_State=3 (TemporaryFail) 是 EPS 自身的"临时故障"状态, 出现时应立即停止驱动转向。
      ret.steerFaultTemporary = bool((self.acc_state == 7) or (self.lkas_state == 3) or (self.eps_warning and self.steer_error_code != 7))
    elif self.is_song_plus_dmi:
      # 宋Plus: 识别 LKAS_State=3(TemporaryFail)。这是 EPS 预警(日志实证: 正常段 --67/--71
      # 出现 st=3 仅 60-90ms 且 err=0 自行恢复; 故障段 --73 持续 484ms 且 err 0→2→4 → TorqueFailed 锁死)。
      # 识别后走 openpilot 标准 steerFaultTemporary → latActive=False 平滑退避, EPS 恢复(2026-08-08 LKAS 分析)。
      # 注: 仅宋Plus 需要; 汉DM/汉EV EPS 按独立 bit 语义, 保持原逻辑。
      # 2026-08-16 .comma(4): 去掉 acc_state==7 —— AccState=7 是相机HUD报的 ACC_ERROR
      # (OP纵向退出后未补发Act=0导致), 此时 EPS 全正常(lkas_state=0/eps_warning=0/steer_err=0),
      # 误判 steerFaultTemporary → 第二次激活弹"转向暂时不可用"+SOFT_DISABLE被踢出。
      ret.steerFaultTemporary = bool((self.lkas_state == 3) or self.eps_warning)
    else:
      # 汉DM/汉EV: 保持原逻辑。
      ret.steerFaultTemporary = bool((self.acc_state == 7) or self.eps_warning)

    self.acc_active_last = ret.cruiseState.enabled

    self.mpc_laks_output = cp_cam.vl["ACC_MPC_STATE"]["LKAS_Output"]  # use to fool mpc
    self.mpc_laks_reqprepare = cp_cam.vl["ACC_MPC_STATE"]["LKAS_ReqPrepare"] != 0  # use to fool mpc
    self.mpc_laks_active = cp_cam.vl["ACC_MPC_STATE"]["LKAS_Active"] != 0  # use to fool mpc

    self.acc_hud_adas_counter = cp_cam.vl["ACC_HUD_ADAS"]["Counter"]
    self.acc_mpc_state_counter = cp_cam.vl["ACC_MPC_STATE"]["Counter"]
    self.acc_cmd_counter = cp_cam.vl["ACC_CMD"]["Counter"]

    self.cam_lkas = copy.copy(cp_cam.vl["ACC_MPC_STATE"])
    self.cam_adas = copy.copy(cp_cam.vl["ACC_HUD_ADAS"])
    self.cam_acc = copy.copy(cp_cam.vl["ACC_CMD"])
    self.esc_eps = copy.copy(cp.vl["ACC_EPS_STATE"])

    # ---- TEMP DEBUG: 诊断 "Steering Temporarily Unavailable" ----（异步写文件, 不用 print 防 pty 卡死）
    self._dbg_cnt = getattr(self, '_dbg_cnt', 0) + 1
    if self._dbg_cnt % 100 == 1:
      acc_debug_write(f"DBG BYD acc_state={self.acc_state} eps_warning={int(self.eps_warning)} steer_err={self.steer_error_code} "
                      f"torque_failed={int(cp.vl['ACC_EPS_STATE']['TorqueFailed'])} lkas_state={self.lkas_state} "
                      f"lkas_prepared={int(self.lkas_prepared)} hands={int(cp.vl['ACC_EPS_STATE']['ReportHandsNotOnSteeringWheel'])} "
                      f"steer_ang={cp.vl['EPS']['SteeringAngle']:.1f} speed={ret.vEgo*CV.MS_TO_KPH:.1f}\n")

    if self.has_pt_radar:
      if self.is_mrr_direct:
        # MRR 直解车型: 雷达由 radar_interface 直解 (0x380~0x3FF), 注入模式默认无前车(199)。
        self.mrr_leading_dist = 199
      elif self.is_radar_mrr:
        # 宋Plus 23款 (RADAR_MRR_CAR): 0x374 轮询广播, DBC TargetID 仅低2位无法区分目标槽
        # (真实目标 25→1/26→2/27→3/28→0, 空闲槽 10→2), 直接按 LongDist>0 判定——
        # 空闲槽 LongDist 实测恒无效(0%有效), 真实目标(25/26/27/28)才 >0。
        if int(cp_cam.vl["RADAR_MRR"]["LongDist"]) > 0:
          self.mrr_leading_dist = int(cp_cam.vl["RADAR_MRR"]["LongDist"])
        else:
          self.mrr_leading_dist = 199
      else:
        mrr_id = int(cp_cam.vl["RADAR_MRR"]["TargetID"])
        if mrr_id == 2:  # 1:left, 2:front, 3:right
          if bool(cp_cam.vl["RADAR_MRR"]["IsValid"]):
            self.mrr_leading_dist = int(cp_cam.vl["RADAR_MRR"]["LongDist"])
          else:
            self.mrr_leading_dist = 199

    ret.steerFaultPermanent = bool(cp.vl["ACC_EPS_STATE"]["TorqueFailed"])  # EPS give up all inputs until restart

    # if self.setTimeDelay == 0:
    #     if not system_time_valid():
    #         yyyy = int(cp.vl["DATETIME"]["YY"] + 2000)
    #         MM = int(cp.vl["DATETIME"]["MM"])
    #         DD = int(cp.vl["DATETIME"]["DD"])
    #         hh = int(cp.vl["DATETIME"]["hh"])
    #         mm = int(cp.vl["DATETIME"]["mm"])
    #         ss = int(cp.vl["DATETIME"]["ss"])
    #         china_time = datetime(yyyy,MM,DD,hh,mm,ss)
    #         china_utc_offset = timedelta(hours=8)
    #         utc_time = china_time - china_utc_offset
    #         cloudlog.debug(f"Setting time to {utc_time}")
    #         try:
    #             subprocess.run(f"TZ=UTC date -s '{utc_time}'", shell=True, check=True)
    #         except subprocess.CalledProcessError:
    #             cloudlog.exception("timed.failed_setting_time")
    # else:
    #     self.setTimeDelay = self.setTimeDelay - 1

    # ALT_PCM_BTN: 新平台取消键语义为 setCruise (源实测映射, 不影响我们的 res/cancel 注入路径)
    cancel_map = {1: ButtonType.setCruise} if (self.CP.flags & BydFlags.ALT_PCM_BTN) else {1: ButtonType.cancel}
    ret.buttonEvents = [
      *create_button_events(self.btn_acc_cancel, prev_btn_acc_cancel, cancel_map),
      *create_button_events(self.btn_acc_set_reset, prev_btn_acc_set_reset, {1: ButtonType.decelCruise, 3: ButtonType.accelCruise}),
      # 车距键方向: 上键(BTN_AccDistanceDecrease)=减车距=gapAdjustCruise(递减);
      #             下键(BTN_AccDistanceIncrease)=增车距=altButton2(cruise.py递增分支)。
      # 2026-08-28 修复: 原两键都映射gapAdjustCruise → 恒递减, 增键失效。
      *create_button_events(self.btn_acc_dist_inc, prev_btn_acc_dist_inc, {1: ButtonType.altButton2}),
      *create_button_events(self.btn_acc_dist_dec, prev_btn_acc_dist_dec, {1: ButtonType.gapAdjustCruise}),
    ]

    return ret, ret_sp

  @staticmethod
  def get_can_parsers(CP, CP_SP):
    pt_messages = [
      # sig_address, frequency
      ("EPS", 100),
      ("CARSPEED", 50),
      ("PEDAL", 50),
      ("EPB", 1),
      ("ACC_EPS_STATE", 50),
      ("DRIVE_STATE", 50),
      ("STALKS", 1),
      ("BCM", 1),
      ("PCM_BUTTONS", 20),
      ("DATETIME", 2),
      ("YAW_RATE", 50),
      ("AXAY", 50),
      ("BELT", 20),
    ]

    if CP.enableBsm:
      pt_messages.append(("BSD_RADAR", 20))

    cam_messages = [
      ("ACC_HUD_ADAS", 50),
      ("ACC_CMD", 50),
      ("ACC_MPC_STATE", 50),
    ]
    if CP.carFingerprint in PT_RADAR_CAR:
      if CP.carFingerprint in MRR_DIRECT_DECODE_CAR:
        # MRR 直解车型: 雷达由 radar_interface 直解 (0x380~0x3FF), carstate 不订阅。
        pass
      else:
        cam_messages.append(("RADAR_MRR", 60))

    return {
      Bus.pt: CANParser(DBC[CP.carFingerprint][Bus.pt], pt_messages, CanBus.ESC),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], cam_messages, CanBus.MPC),
    }

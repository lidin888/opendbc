import numpy as np
from opendbc.car import structs
from opendbc.car.byd.values import CanBus, CarControllerParams, CAR

GearShifter = structs.CarState.GearShifter
VisualAlert = structs.CarControl.HUDControl.VisualAlert


def byd_checksum(byte_key, dat):
  first_bytes_sum = sum(byte >> 4 for byte in dat)
  second_bytes_sum = sum(byte & 0xF for byte in dat)
  remainder = second_bytes_sum >> 4
  second_bytes_sum += byte_key >> 4
  first_bytes_sum += byte_key & 0xF
  first_part = (-first_bytes_sum + 0x9) & 0xF
  second_part = (-second_bytes_sum + 0x9) & 0xF
  return (((first_part + (-remainder + 5)) << 4) + second_part) & 0xFF


# MPC -> Panda -> EPS
def create_steering_control(packer, CP, cam_msg: dict, req_torque, req_prepare, active, hud_control, counter):
  values = {
    s: cam_msg[s]
    for s in [
      "AutoFullBeamState",
      "LeftLaneState",
      "LKAS_Config",
      "SETME2_0x1",
      "MPC_State",
      "AutoFullBeam_OnOff",
      "LKAS_Output",
      "LKAS_Active",
      "SETME3_0x0",
      "TrafficSignRecognition_OnOff",
      "SETME4_0x0",
      "SETME5_0x1",
      "RightLaneState",
      "LKAS_State",
      "TrafficSignRecognition_Result",
      "LKAS_AlarmType",
      "SETME7_0x3",
    ]
  }

  values["ReqHandsOnSteeringWheel"] = 0
  values["LKAS_ReqPrepare"] = req_prepare
  values["Counter"] = counter

  if active:
    mpc_state = values["MPC_State"]  # 2: Cancelling lkas control
    values.update(
      {
        "LKAS_Output": req_torque,
        "LKAS_Active": 1,
        "LKAS_State": 4 if (mpc_state == 2) else 2,
        "LeftLaneState": 3 if hud_control.leftLaneDepart else int(hud_control.leftLaneVisible) + 1,
        "RightLaneState": 3 if hud_control.rightLaneDepart else int(hud_control.rightLaneVisible) + 1,
      }
    )
  else:  # Note: This disables the stock AEB feature: turn steering wheel while close impacting obstacles in front.
    values.update(
      {
        "LKAS_Output": 0,
        "LKAS_Active": 0,
      }
    )

  data = packer.make_can_msg("ACC_MPC_STATE", CanBus.ESC, values)[1]
  values["CheckSum"] = byd_checksum(0xAF, data)
  return packer.make_can_msg("ACC_MPC_STATE", CanBus.ESC, values)


# op long control
def acc_cmd(packer, CP, cam_msg: dict, mrr_leaddist, accel, rfss, sss, longActive, jerk_mpc=0.0):
  values = {
    s: cam_msg[s]
    for s in [
      "AccelCmd",
      "ComfortBandUpper",
      "ComfortBandLower",
      "JerkUpperLimit",
      "SETME1_0x1",
      "JerkLowerLimit",
      "ResumeFromStandstill",
      "StandstillState",
      "BrakeBehaviour",
      "AccReqNotStandstill",
      "AccControlActive",
      "AccOverrideOrStandstill",
      "EspBehaviour",
      "Counter",
      "SETME2_0xF",
    ]
  }

  jerk_base_upper = np.interp(mrr_leaddist, CarControllerParams.K_jerk_xp, CarControllerParams.K_jerk_base_upper_fp)
  jerk_base_lower = np.interp(mrr_leaddist, CarControllerParams.K_jerk_xp, CarControllerParams.K_jerk_base_lower_fp)

  # ---------------------------------------------------------------------------
  # 2026-10-09 泄压(松刹)方向 jerk 上限 —— 照官方现代 HyundaiJerk 的自适应写法
  # (opendbc/car/hyundai/carcontroller.py:613-645 `jerk_u = clip(actuators.jerk*2.0, 0.5, 5.0)`)。
  # **key 用 MPC 解出的 jerk(= 意图), 不用当前 accel(= 命令值)。**
  # 原因(实测 .comma(9), 宋21 两个真实事件 4号16:56:49 / 6号07:55:47):
  #   前车急减速→我们深刹(aCmd -3.4)→前车提速走人. 排除了雷达/MPC/状态机/门控/Act平台
  #   (长链审计见 tools/debug/p7_jerk_ab.py、p8c.py; 全链 11 环逐环验过只有 jerk 绑定),
  #   实测 aEgo 回升斜率 1.0~1.05 m/s³ ≈ 当时下发的 ju=0.8~1.0 → ESC 1:1 跟随该上限。
  #   关键: "想松开"发生在 aCmd 仍是负数的时候 —— 那时 MPC.jerk 已经是 +1.37~+1.70,
  #   而旧公式按 accel<0 取 0.8(见 git 历史); 用 jerk 做 key 则立刻给 2.7~3.4。
  # 公式选型依据(27 个真实事件离线仿真, 指标=命令转正→aEgo 到 0 的耗时, 中位改善):
  #   旧代码 1.77s → B(accel*2.4,地板0.8) +0.04s(仅11/27事件有效) → C(+松刹1.6) +1.12s(27/27)
  #   → D(官方,地板0.5) +0.89s(4个事件反而变差) → **F=本式 clip(jerk*2,1.6,5.0) +1.21s(27/27)**
  #   对 ESC 内部上限不敏感: 封顶2.0仍+1.19s, 封顶1.5仍+1.00s。
  # 地板取 1.6(不取官方的 0.5): 0.5 比旧值 0.8 还紧 → 在"MPC.jerk 仍≈0/为负"的片段会反而更慢
  #   (实测 2 个事件崩到 24s/11s), 1.6 消除该风险且与 C 组一致。
  # 限制: **只改泄压方向**。jerk_lower(建压/刹车多快)一律不动 → 刹车力度与旧行为完全一致。
  # ---------------------------------------------------------------------------
  jerk_upper = float(np.clip(jerk_mpc * CarControllerParams.K_jerk_up_from_mpc_gain,
                             CarControllerParams.K_jerk_up_from_mpc_min,
                             CarControllerParams.K_jerk_up_from_mpc_max))
  if accel < 0:  # use lower factor
    jerk_lower = jerk_base_lower + accel * CarControllerParams.K_accel_jerk_lower
  else:
    jerk_lower = jerk_base_lower

  if longActive:
    values.update(
      {
        "AccelCmd": accel,
        "ComfortBandUpper": 0.05 if mrr_leaddist > 50 else 0.10,
        "ComfortBandLower": 0.05 if mrr_leaddist > 50 else 0.10,
        "JerkUpperLimit": jerk_upper,
        "JerkLowerLimit": jerk_lower,
        "ResumeFromStandstill": rfss,
        "StandstillState": sss,
        "AccControlActive": 1,  # 关键！触发 safety 层阻断原车 ACC_CMD
        "AccReqNotStandstill": 1,
      }
    )
  else:
    # 当acc_enhance退出时，确保不发送加速命令
    # 只允许保持或减速，不允许加速，避免与原车ACC冲突
    current_accel = values.get("AccelCmd", 0)
    values.update(
      {
        "AccelCmd": min(current_accel, 0),  # 不允许正值（加速）
        "AccControlActive": 0,  # 恢复原车 ACC 控制
        "AccReqNotStandstill": 0,
      }
    )

  data = packer.make_can_msg("ACC_CMD", CanBus.ESC, values)[1]
  values["CheckSum"] = byd_checksum(0xAF, data)
  return packer.make_can_msg("ACC_CMD", CanBus.ESC, values)


# OP纵向: 注入 0x32D (ACC_HUD_ADAS) 驱动仪表巡航显示 (2026-08-16 HUD注入方案)
# 背景: 原车ACC激活时相机广播 AccState=3 + SetSpeed=设定速度 → 仪表显示巡航速度/车距。
#       OP纵向让原车ACC待机, 相机广播的是死值(SetSpeed=0/SetDistance=3/AccState=1/2)
#       → 仪表不跟OP UI设定。本函数复制相机帧, 覆盖为原车ACC激活态形态:
#       SetSpeed=OP设定速度 / SetDistance=OP车距等级 / AccState=3(激活), 50Hz发ESC总线。
# 待机时(accstate=None)透传相机值, 等于"OP替代panda转发", 内容与原转发帧一致。
# 注: 当前 carcontroller 调用总传非 None accstate(激活=3, 待机=相机待机值1/2),
#     并显式传 setdistance(激活/待机均=OP UI车距等级 1~4), 故 accstate=None 分支仅作回退。
#     (2026-08-18: 待机 SetDistance 也注入 OP UI 车距等级——相机待机死值=3 是"默认3档",
#     透传会让暂停后仪表车距回3格与UI不同步。原车 SetDistance 为 1~4 四档,
#     byd_generic_pt.dbc VAL_ 813 铁证: 4"4BAR" 3"3BAR" 2"2BAR" 1"1BAR"。)
def create_acc_hud_adas(packer, cam_msg: dict, setspeed, setdistance, accstate, counter):
  values = {
    s: cam_msg[s]
    for s in [
      "SetSpeed",
      "HasLead",
      "SetDistance",
      "LeadingDistance",
      "AEB",
      "FCW",
      "SETME1_0x1",
      "AccState",
      "AccOn1",
      "CloseWarning",
      "SETME2_0x1",
      "Notify",
      "Status",
      "SETME3_0xFFF",
      "SETME4_0xF",
    ]
  }
  if accstate is None:
    # 待机: 透传相机值, 保持与panda转发帧一致(OP替代转发, 内容不变)
    values["SetSpeed"] = cam_msg["SetSpeed"]
    values["SetDistance"] = cam_msg["SetDistance"]
    values["AccState"] = cam_msg["AccState"]
  else:
    # 激活: 覆盖为原车ACC激活态 (commalog0811 铁证: state=3 + SetSpeed跟随设定)
    values["SetSpeed"] = setspeed
    values["SetDistance"] = setdistance
    values["AccState"] = accstate
  values["Counter"] = counter

  data = packer.make_can_msg("ACC_HUD_ADAS", CanBus.ESC, values)[1]
  values["CheckSum"] = byd_checksum(0xAF, data)
  return packer.make_can_msg("ACC_HUD_ADAS", CanBus.ESC, values)


# send fake torque feedback from eps to trick MPC, preventing DTC, so that safety features such as AEB still working
def create_fake_318(packer, CP, esc_msg: dict, faketorque, laks_reqprepare, laks_active, enabled, counter):
  is_tang_dm = CP.carFingerprint == CAR.BYD_TANG_DM

  # 唐DM 用 LKAS_State 2bit 枚举(0=NotReady/1=Ready/2=Active/3=TemporaryFail) 单字段;
  # 汉DM(及其他共用 byd_han_dmev_2020.dbc) 用旧字段 LKAS_Prepared + CruiseActivated 两信号。
  base_fields = [
    "TorqueFailed",
    "SETME1_0x1",
    "SteerWarning",
    "SteerErrorCode",
    "MainTorque",
    "SETME3_0x1",
    "SETME4_0x3",
    "SteerDriverTorque",
    "SETME5_0xFF",
    "SETME6_0xFFF",
  ]
  if is_tang_dm:
    values = {s: esc_msg[s] for s in ["LKAS_State"] + base_fields}
  else:
    values = {s: esc_msg[s] for s in ["LKAS_Prepared", "CruiseActivated"] + base_fields}

  values["ReportHandsNotOnSteeringWheel"] = 0
  values["Counter"] = counter

  if enabled:
    if laks_active:
      if is_tang_dm:
        values.update({"LKAS_State": 2, "MainTorque": faketorque})  # Active
      else:
        values.update({"LKAS_Prepared": 0, "CruiseActivated": 1, "MainTorque": faketorque})
    elif laks_reqprepare:
      if is_tang_dm:
        values.update({"LKAS_State": 1, "MainTorque": 0})  # Ready
      else:
        values.update({"LKAS_Prepared": 1, "MainTorque": 0})
    else:
      if is_tang_dm:
        values.update({"LKAS_State": 0, "MainTorque": 0})  # NotReady
      else:
        values.update({"LKAS_Prepared": 0, "MainTorque": 0})

  data = packer.make_can_msg("ACC_EPS_STATE", CanBus.MPC, values)[1]
  values["CheckSum"] = byd_checksum(0xAF, data)
  return packer.make_can_msg("ACC_EPS_STATE", CanBus.MPC, values)


def create_button_fwd(packer, pcm_vals):
  data = packer.make_can_msg("PCM_BUTTONS", CanBus.MPC, pcm_vals)[1]
  pcm_vals["CheckSum"] = byd_checksum(0xAF, data)
  return packer.make_can_msg("PCM_BUTTONS", CanBus.MPC, pcm_vals)

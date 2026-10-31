#!/usr/bin/env python3
"""
BYD 安全测试 —— 重点: OP纵向模式(OP_LONG flag) 的 panda 行为
1. tx_hook: OP发ACC_CMD(Act=1) → controls_allowed=true; Act=0 → false
2. tx_hook: ACC_CMD 只在 bus=ESC(OP注入)生效, 相机(MPC)不干扰
3. fwd_hook: byd_block_original_acc=true 时阻断原车ACC_CMD转发
4. OP纵向跳过 pcm_cruise_check (原车ACC待机不每帧清 controls_allowed)
5. HUD 0x32D 双向阻断 (仪表只收 OP 注入帧)

注: SP 的 controls_allowed_lateral(MADS) 双门控支持纵向off时转向, 转向强拦已恢复
    (v9-carrot 时代软化不再需要)。
"""
import unittest

from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety

FLAG_SONG_PLUS_DMI = 0x4
FLAG_OP_LONG = 0x40


class TestBydLongitudinalSafety(common.CarSafetyTest, common.MotorTorqueSteeringSafetyTest):

  TX_MSGS = [[0x32E, 0], [0x3B0, 2], [0x316, 0], [0x318, 2], [0x32D, 0]]
  STANDSTILL_THRESHOLD = 0.1
  RELAY_MALFUNCTION_ADDRS = {0: (0x316,)}
  # fwd_hook 黑名单 (匹配 modes/byd.h byd_fwd_hook):
  # ESC→MPC: ACC_EPS_STATE(0x318/0x1FC) + PCM_BUTTONS(0x3B0) + OP纵向0x32D(回灌阻断)
  # MPC→ESC: ACC_MPC_STATE(0x316/0x1E2) + OP纵向0x32D(相机帧阻断)
  FWD_BLACKLISTED_ADDRS = {0: [0x318, 0x1FC, 0x3B0, 0x32D], 2: [0x316, 0x1E2, 0x32D]}

  MAX_RATE_UP = 18
  MAX_RATE_DOWN = 18
  MAX_TORQUE_LOOKUP = ([0], [300])
  MAX_TORQUE_ERROR = 80

  MAX_RT_DELTA = 243

  DRIVER_TORQUE_ALLOWANCE = 15
  DRIVER_TORQUE_FACTOR = 1

  NO_STEER_REQ_BIT = False

  def setUp(self):
    self.packer = CANPackerSafety("byd_han_dmev_2020")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.byd, FLAG_OP_LONG | FLAG_SONG_PLUS_DMI)
    self.safety.init_tests()

  # ---- 消息构造 ----

  def _torque_meas_msg(self, torque):
    values = {"MainTorque": torque}
    return self.packer.make_can_msg_safety("ACC_EPS_STATE", 0, values)

  def _torque_driver_msg(self, torque):
    values = {"MainTorque": torque}
    return self.packer.make_can_msg_safety("ACC_EPS_STATE", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAS_Active": steer_req, "LKAS_Output": torque}
    return self.packer.make_can_msg_safety("ACC_MPC_STATE", 0, values)

  def _speed_msg(self, speed):
    values = {"CarDisplaySpeed": speed}
    return self.packer.make_can_msg_safety("CARSPEED", 0, values)

  def _user_brake_msg(self, brake):
    values = {"BrakePedal": brake}
    return self.packer.make_can_msg_safety("PEDAL", 0, values)

  def _user_gas_msg(self, gas):
    values = {"AcceleratorPedal": gas}
    return self.packer.make_can_msg_safety("PEDAL", 0, values)

  def _pcm_status_msg(self, enable):
    # 原车ACC待机(pcmCruise=False): HUD AccState=0/2 (OFF/ACC_ON), 不产生 cruise engaged
    values = {"AccState": 2 if enable else 0}
    return self.packer.make_can_msg_safety("ACC_HUD_ADAS", 2, values)

  def _acc_cmd_msg(self, act, bus=0):
    """构造 ACC_CMD(0x32E)。act=1: OP接管(Act=1); act=0: 干净退出"""
    values = {"AccControlActive": act, "AccelCmd": 0.0}
    return self.packer.make_can_msg_safety("ACC_CMD", bus, values)

  def _button_msg(self, cancel=False, updown=0):
    values = {"BTN_AccCancel": cancel, "BTN_AccUpDown_Cmd": updown}
    return self.packer.make_can_msg_safety("PCM_BUTTONS", 2, values)

  # ---- OP_LONG 语义下跳过的通用测试 (pcm_cruise_check 被设计性跳过) ----

  def test_cruise_engaged_prev(self):
    raise unittest.SkipTest("OP_LONG: pcm_cruise_check 设计性跳过 (controls_allowed 由 ACC_CMD 驱动)")

  def test_enable_control_allowed_from_cruise(self):
    raise unittest.SkipTest("OP_LONG: pcm_cruise_check 设计性跳过 (controls_allowed 由 ACC_CMD 驱动)")

  def test_disable_control_allowed_from_cruise(self):
    raise unittest.SkipTest("OP_LONG: pcm_cruise_check 设计性跳过 (controls_allowed 由 ACC_CMD 驱动)")

  # ---- OP纵向核心测试 ----

  def test_op_long_acc_cmd_controls_allowed(self):
    """OP纵向: OP发Act=1 → controls_allowed=true; Act=0 → false"""
    # 初始: 原车ACC待机, controls_allowed 应为 false
    self.assertFalse(self.safety.get_controls_allowed())
    # 发送原车ACC待机HUD, 不应清/置 controls_allowed (跳过pcm_cruise_check)
    self._rx(self._pcm_status_msg(False))
    self.assertFalse(self.safety.get_controls_allowed())

    # OP发Act=1 ACC_CMD (bus=ESC) → controls_allowed=true
    self.assertTrue(self._tx(self._acc_cmd_msg(1)))
    self.assertTrue(self.safety.get_controls_allowed())

    # OP发Act=0 → controls_allowed=false
    self.assertTrue(self._tx(self._acc_cmd_msg(0)))
    self.assertFalse(self.safety.get_controls_allowed())

  def test_op_long_camera_acc_cmd_doesnt_clear(self):
    """相机(MPC总线)发ACC_CMD 不被当成OP帧处理"""
    # 相机在 MPC 总线发 0x32E: TX_MSGS 不允许 → 应被拒绝(不触发controls_allowed逻辑)
    self.assertFalse(self._tx(self._acc_cmd_msg(1, bus=2)))
    self.assertFalse(self._tx(self._acc_cmd_msg(0, bus=2)))

    # OP 在 ESC 总线发 Act=1 → controls_allowed=true (正常)
    self.assertTrue(self._tx(self._acc_cmd_msg(1)))
    self.assertTrue(self.safety.get_controls_allowed())

    # 相机帧被拒后, OP 的 controls_allowed 不受影响
    self.assertFalse(self._tx(self._acc_cmd_msg(0, bus=2)))
    self.assertTrue(self.safety.get_controls_allowed())

  def test_op_long_blocks_original_acc_fwd(self):
    """OP接管(byd_block_original_acc=true)时, 原车ACC_CMD不转发 (fwd_hook)"""
    self.assertTrue(self._tx(self._acc_cmd_msg(1)))  # OP接管
    # fwd_hook: MPC 来的原车 ACC_CMD 应被阻断(返回-1)
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, 0x32E))

    # OP退控(Act=0)后恢复转发
    self.assertTrue(self._tx(self._acc_cmd_msg(0)))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, 0x32E))

  def test_op_long_buttons(self):
    """OP纵向模式按钮安全: 非cancel需要controls_allowed"""
    self.safety.set_controls_allowed(0)
    # 未激活: 只能 cancel
    self.assertTrue(self._tx(self._button_msg(cancel=True)))
    # speed adjust 需要 controls_allowed
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._tx(self._button_msg(updown=1)))

  def test_op_long_hud_adas_fwd_blocked(self):
    """OP纵向: 0x32D(ACC_HUD_ADAS) 双向阻断 (HUD注入方案)
    - MPC(相机)来的0x32D不转ESC: 仪表改收OP注入帧
    - ESC(OP注入)来的0x32D不转MPC: 防carstate读到注入帧AccState=3误判ACC enabled
    """
    # 确保未接管(byd_block_original_acc=false): OP发Act=0清零
    self.assertTrue(self._tx(self._acc_cmd_msg(0)))
    self.assertEqual(-1, self.safety.safety_fwd_hook(2, 0x32D))  # 相机→ESC 阻断
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0x32D))  # ESC→MPC 阻断
    # 非0x32D报文不受影响: 相机ACC_CMD(0x32E)未接管时仍转发到ESC
    self.assertEqual(0, self.safety.safety_fwd_hook(2, 0x32E))
    # OP注入0x32D到ESC被TX白名单放行
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("ACC_HUD_ADAS", 0, {"SetSpeed": 60.0, "SetDistance": 2, "AccState": 3})))

  def test_op_long_pcm_status_not_affect(self):
    """OP纵向: 原车ACC待机(HUD acc_state=OFF/ON) 不触发 pcm_cruise_check"""
    self.assertTrue(self._tx(self._acc_cmd_msg(1)))
    self.assertTrue(self.safety.get_controls_allowed())
    # 原车HUD报 OFF, 不应清 controls_allowed (OP_LONG跳过pcm_cruise_check)
    self._rx(self._pcm_status_msg(False))
    self.assertTrue(self.safety.get_controls_allowed())
    # HUD报 ACC_ON, 也不该清
    self._rx(self._pcm_status_msg(True))
    self.assertTrue(self.safety.get_controls_allowed())


if __name__ == "__main__":
  unittest.main()

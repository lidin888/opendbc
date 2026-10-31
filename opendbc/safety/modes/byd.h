#pragma once

#include "opendbc/safety/declarations.h"

// BYD safety — 行为基准: v9-byd safety_byd.h (2026-10-31 逐字对拍移植, 新框架 API)
// 关键语义 (勿回退):
//   ① CruiseActivated = (data[0] & 0x3) == 2 合成枚举, 不是 bit1 单判 (TemporaryFail=3 不算激活)
//   ② OP纵向 (OP_LONG=0x40): controls_allowed 由 OP 的 ACC_CMD AccControlActive 驱动,
//      rx 不走 pcm_cruise_check (原车 ACC 待机会每帧清 ca → controlsMismatch)
//   ③ AccControlActive=1 持续阻断原车 ACC_CMD (MPC→ESC) 转发; =0 恢复
//   ④ OP纵向 HUD 拓扑: 阻断 OP 注入 0x32D 回灌 MPC + 阻断相机 0x32D→ESC (仪表只收 OP 帧)
//   ⑤ AEB 0x32F 双向放行 (保原厂 AEB)
//   ⑥ 转向 steer_req = LKAS_Active || CruiseActivated (OR 语义); steer 检查强拦
//      (SP 的 controls_allowed_lateral/MADS 双门控支持纵向off时转向, 不需要 v9-carrot 时代的软化)

#define BYD_CANADDR_IPB               0x1F0
#define BYD_CANADDR_ACC_MPC_STATE     0x316
#define BYD_CANADDR_ACC_MPC_STATE_SEAL 0x1E2
#define BYD_CANADDR_ACC_EPS_STATE     0x318
#define BYD_CANADDR_ACC_EPS_STATE_SEAL 0x1FC
#define BYD_CANADDR_ACC_HUD_ADAS      0x32D
#define BYD_CANADDR_ACC_CMD           0x32E
#define BYD_CANADDR_ACC_AEB           0x32F
#define BYD_CANADDR_PCM_BUTTONS       0x3B0
#define BYD_CANADDR_DRIVE_STATE       0x242
#define BYD_CANADDR_PEDAL             0x342
#define BYD_CANADDR_EPS               0x11F
#define BYD_CANADDR_CARSPEED          0x121

#define BYD_CANBUS_ESC  0
#define BYD_CANBUS_MRR  1
#define BYD_CANBUS_MPC  2

static bool byd_eps_cruiseactivated = false;
static bool byd_block_original_acc = false;  // 持续阻断原车 ACC_CMD 的标志
static bool byd_op_long = false;             // OP纵向模式: controls_allowed 由 OP 的 ACC_CMD 驱动

typedef enum {
  HAN_TANG_DMEV,
  SEAL
} BydPlatform;
static BydPlatform byd_platform;

static void byd_rx_hook(const CANPacket_t *msg) {
  int bus = msg->bus;
  int addr = msg->addr;

  if (bus == BYD_CANBUS_ESC) {
    if (addr == BYD_CANADDR_PEDAL) {
      gas_pressed = (msg->data[0] != 0U);
      brake_pressed = (msg->data[1] != 0U);
    } else if (addr == BYD_CANADDR_CARSPEED) {
      int speed_raw = (((msg->data[1] & 0x0FU) << 8) | msg->data[0]);
      vehicle_moving = (speed_raw != 0);
    } else if (addr == BYD_CANADDR_ACC_EPS_STATE) {
      // ① 低2位合成枚举: 0=NotReady/1=Ready/2=Active/3=TemporaryFail, 仅 ==2 算激活
      byd_eps_cruiseactivated = ((msg->data[0] & 0x3U) == 2U);

      int angle_meas_new = ((msg->data[3] << 8) | msg->data[2]);
      angle_meas_new = to_signed(angle_meas_new, 16) * 10;
      update_sample(&angle_meas, angle_meas_new);

      // MainTorque: bit8 起 12 位 = byte1 低4位 | byte2 (宋Plus/汉DM 同读, 否则转向力矩限制形同虚设)
      int torque_motor = (((msg->data[2] & 0x0FU) << 8) | msg->data[1]);
      torque_motor = to_signed(torque_motor, 12);
      update_sample(&torque_meas, torque_motor);
    } else if (addr == BYD_CANADDR_ACC_EPS_STATE_SEAL) {
      byd_eps_cruiseactivated = ((msg->data[0] & 0x3U) == 2U);
      int torque_motor = (((msg->data[5] & 0x0FU) << 8) | msg->data[4]);
      torque_motor = to_signed(torque_motor, 12);
      update_sample(&torque_meas, torque_motor);
    } else {
      //empty
    }
  } else if (bus == BYD_CANBUS_MPC) {
    if (addr == BYD_CANADDR_ACC_HUD_ADAS) {
      // ② OP纵向: 不走 pcm_cruise_check, controls_allowed 由 tx_hook 驱动
      if (!byd_op_long) {
        if (byd_platform == SEAL) {
          bool cruise_engaged = GET_BIT(msg, 22U);  // AccOn1
          pcm_cruise_check(cruise_engaged);
        } else {
          unsigned int accstate = ((msg->data[2] >> 3) & 0x07U);
          bool cruise_engaged = (accstate == 3U) || (accstate == 5U);  // 3=acc_active, 5=user force accel
          pcm_cruise_check(cruise_engaged);
        }
      }
    } else {
      //empty
    }
  } else {
    //empty
  }
}

static bool byd_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits BYD_STEERING_LIMITS = {
    .max_torque = 300,
    .max_rate_up = 18,
    .max_rate_down = 18,
    .max_rt_delta = 243,  // 18 * 250/20 = 225 + 18
    .max_torque_error = 80,
    .type = TorqueMotorLimited,
  };

  const AngleSteeringLimits BYD_SEAL_STEERING_LIMITS = {
    .angle_deg_to_can = 100,
    .angle_rate_up_lookup = {
      {0., 5., 15.},
      {5., .8, .15}
    },
    .angle_rate_down_lookup = {
      {0., 5., 15.},
      {5., 3.5, .4}
    },
  };

  bool tx = true;
  int bus = msg->bus;
  int addr = msg->addr;

  // ③ AccControlActive 检测: 只认 OP 注入的 ACC_CMD(ESC 总线), 相机的(MPC)不走此逻辑
  //    —— 否则相机 Act=0 帧会误清 controls_allowed (2026-08-16 .comma(3) 实测)
  if ((addr == BYD_CANADDR_ACC_CMD) && (bus == BYD_CANBUS_ESC)) {
    int acc_control_active = (msg->data[5] >> 4) & 0x1U;
    if (acc_control_active == 1) {
      byd_block_original_acc = true;
      if (byd_op_long) {
        controls_allowed = true;
      }
    } else {
      byd_block_original_acc = false;
      if (byd_op_long) {
        controls_allowed = false;
      }
    }
  }

  if (bus == BYD_CANBUS_ESC) {
    // PCM_BUTTONS: cancel 无条件放行, 速度加减需 controls_allowed
    if (addr == BYD_CANADDR_PCM_BUTTONS) {
      int btn_up_down = (msg->data[0] >> 3) & 0x3U;
      int btn_cancel = (msg->data[0] >> 6) & 0x1U;

      bool allowed_cancel = (btn_cancel == 1);
      bool allowed_speed_adj = ((btn_up_down == 1) || (btn_up_down == 3)) && controls_allowed;

      if (!allowed_cancel && !allowed_speed_adj && (btn_up_down != 0)) {
        tx = false;
      }
    }

    if (byd_platform == SEAL) {
      if (addr == BYD_CANADDR_ACC_MPC_STATE_SEAL) {
        int desired_angle = (((msg->data[3] & 0x07U) << 8U) | msg->data[2]);
        desired_angle = to_signed(desired_angle, 11);
        bool steer_req = GET_BIT(msg, 28U) || byd_eps_cruiseactivated;  // ⑥ OR 语义

        if (steer_angle_cmd_checks(desired_angle, steer_req, BYD_SEAL_STEERING_LIMITS)) {
          tx = false;  // 恢复强拦 (2026-10-31): SP 的 controls_allowed_lateral(MADS) 双门控已支持纵向off时转向,
                       // 无需 carrot 时代 aol_allowed 的软化 (v9-byd "disable this" 是 carrot 全局设计的妥协)
        }
      }
    } else {
      if (addr == BYD_CANADDR_ACC_MPC_STATE) {
        int desired_torque = (((msg->data[3] & 0x07U) << 8U) | msg->data[2]);
        desired_torque = to_signed(desired_torque, 11);
        bool steer_req = GET_BIT(msg, 28U) || byd_eps_cruiseactivated;  // ⑥ OR 语义

        if (steer_torque_cmd_checks(desired_torque, steer_req, BYD_STEERING_LIMITS)) {
          tx = false;  // 恢复强拦 (2026-10-31): SP 的 controls_allowed_lateral(MADS) 双门控已支持纵向off时转向,
                       // 无需 carrot 时代 aol_allowed 的软化 (v9-byd "disable this" 是 carrot 全局设计的妥协)
        }
      }
    }
  }

  return tx;
}

static bool byd_fwd_hook(int bus, int addr) {
  bool block_msg = false;

  if (bus == BYD_CANBUS_ESC) {
    // 来自 ESC: 阻断 EPS 状态/按键回灌 MPC; ④ OP纵向时阻断 OP 注入的 0x32D 回灌
    block_msg = (addr == BYD_CANADDR_ACC_EPS_STATE)
             || (addr == BYD_CANADDR_ACC_EPS_STATE_SEAL)
             || (addr == BYD_CANADDR_PCM_BUTTONS)
             || (byd_op_long && (addr == BYD_CANADDR_ACC_HUD_ADAS));
  } else if (bus == BYD_CANBUS_MPC) {
    // 来自 MPC(相机): 阻断相机转向命令/转向状态; ④ OP纵向时阻断相机 0x32D (仪表改收 OP 注入帧);
    // ③ 持续阻断原车 ACC_CMD; ⑤ AEB 0x32F 不在列表 = 双向放行
    block_msg = (addr == BYD_CANADDR_ACC_MPC_STATE)
             || (addr == BYD_CANADDR_ACC_MPC_STATE_SEAL)
             || (byd_op_long && (addr == BYD_CANADDR_ACC_HUD_ADAS))
             || (byd_block_original_acc && (addr == BYD_CANADDR_ACC_CMD));
  } else {
    //empty
  }

  return block_msg;
}

static safety_config byd_init(uint16_t param) {
  const uint32_t FLAG_SEAL = 0x20U;
  const uint32_t FLAG_OP_LONG = 0x40U;

  safety_config ret;

  // ② OP纵向模式: controls_allowed 由 OP 的 ACC_CMD 驱动 (rx_hook 跳过 pcm_cruise_check)
  byd_op_long = GET_FLAG(param, FLAG_OP_LONG);

  static RxCheck byd_handm_rx_checks[] = {
    {.msg = {{BYD_CANADDR_PEDAL,          BYD_CANBUS_ESC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_CARSPEED,       BYD_CANBUS_ESC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_EPS_STATE,  BYD_CANBUS_ESC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_HUD_ADAS,   BYD_CANBUS_MPC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_MPC_STATE,  BYD_CANBUS_MPC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  static RxCheck byd_seal_rx_checks[] = {
    {.msg = {{BYD_CANADDR_PEDAL,               BYD_CANBUS_ESC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_CARSPEED,            BYD_CANBUS_ESC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_EPS_STATE_SEAL,  BYD_CANBUS_ESC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_HUD_ADAS,        BYD_CANBUS_MPC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_MPC_STATE,       BYD_CANBUS_MPC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
    {.msg = {{BYD_CANADDR_ACC_MPC_STATE_SEAL,  BYD_CANBUS_MPC, 8, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},
  };

  static const CanMsg BYD_HANDM_TX_MSGS[] = {
    {BYD_CANADDR_PCM_BUTTONS,     BYD_CANBUS_MPC, 8, .check_relay = false},
    {BYD_CANADDR_ACC_CMD,         BYD_CANBUS_ESC, 8, .check_relay = false},
    {BYD_CANADDR_ACC_MPC_STATE,   BYD_CANBUS_ESC, 8, .check_relay = true},
    {BYD_CANADDR_ACC_EPS_STATE,   BYD_CANBUS_MPC, 8, .check_relay = false},
    {BYD_CANADDR_ACC_HUD_ADAS,    BYD_CANBUS_ESC, 8, .check_relay = false},  // ④ OP纵向 HUD 注入
  };

  static const CanMsg BYD_SEAL_TX_MSGS[] = {
    {BYD_CANADDR_PCM_BUTTONS,        BYD_CANBUS_MPC, 8, .check_relay = false},
    {BYD_CANADDR_ACC_CMD,            BYD_CANBUS_ESC, 8, .check_relay = false},
    {BYD_CANADDR_ACC_MPC_STATE,      BYD_CANBUS_ESC, 8, .check_relay = true},
    {BYD_CANADDR_ACC_MPC_STATE_SEAL, BYD_CANBUS_ESC, 8, .check_relay = true},
    {BYD_CANADDR_ACC_EPS_STATE_SEAL, BYD_CANBUS_MPC, 8, .check_relay = false},
    {BYD_CANADDR_ACC_HUD_ADAS,       BYD_CANBUS_ESC, 8, .check_relay = false},  // ④ OP纵向 HUD 注入
  };

  bool use_seal = GET_FLAG(param, FLAG_SEAL);
  if (use_seal) {
    byd_platform = SEAL;
    ret = BUILD_SAFETY_CFG(byd_seal_rx_checks, BYD_SEAL_TX_MSGS);
  } else {
    byd_platform = HAN_TANG_DMEV;
    ret = BUILD_SAFETY_CFG(byd_handm_rx_checks, BYD_HANDM_TX_MSGS);
  }

  return ret;
}

const safety_hooks byd_hooks = {
  .init = byd_init,
  .rx = byd_rx_hook,
  .tx = byd_tx_hook,
  .fwd = byd_fwd_hook,
};

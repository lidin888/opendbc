import time
import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.lateral import apply_driver_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.byd import bydcan
from opendbc.car.byd.debug_log import acc_debug_write
from opendbc.car.byd.values import CarControllerParams
from openpilot.common.params import Params

VisualAlert = structs.CarControl.HUDControl.VisualAlert
ButtonType = structs.CarState.ButtonEvent.Type
LongCtrlState = structs.CarControl.Actuators.LongControlState


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)

    self.packer = CANPacker(dbc_names[Bus.pt])

    self.frame = 0
    self.last_steer_frame = 0
    self.last_acc_frame = 0


    self.apply_torque_last = 0

    self.mpc_lkas_counter = 0
    self.mpc_acc_counter = 0
    self.eps_fake318_counter = 0

    # 泄压 jerk 内部估算状态: sptn Actuators 无 jerk 字段, 用请求加速度差分替代 CC.actuators.jerk
    self._jerk_est = 0.0
    self._last_accel_t_ns = None

    self.lkas_req_prepare = 0
    self.lkas_active = 0
    self.lat_safeoff = 0
    self.driver_backoff_frames = 0

    # 后装L2 EPS 保护总开关 (2026-08-13 建, 2026-09-04 扩展): 0=关(默认), 1=开。
    # 作用(开关=1 时全部生效, 且仅对后装L2车 PLATFORM_SONG_PLUS_DMI 生效):
    #   ①驾驶员与OP反向施力对抗时, 清空softstart + 0.5s退避锁保持零扭矩;
    #   ②EPS 报 steerFaultTemporary/Permanent 时立即清 lkas_active 当帧归零退出
    #     (不走 lat_safeoff 渐降, 避免 tmp→perm 决策窗内收到 Act=1+渐降 矛盾状态);
    #   ③宋Plus 扭矩输出前/中检查 EPS prepared(与唐DM 对齐), 未就绪即退避重握手;
    #   ④故障期间禁止重新激活(防 Act 乒乓发矛盾帧)。
    # 背景: TorqueFailed/报lkas 只出在后装L2车(低配EPS固件对扭矩冲突/矛盾状态敏感),
    #       高配原厂L2 EPS 不敏感。高配车(唐/汉/秦/元等)无论开关状态都保持原行为,
    #       完全不走此开关(2026-09-04 用户要求); 后装车用户自行开启。
    self.params = Params()
    self.lateral_backoff_enabled = self.params.get_int("LateralTorqueDriverBackoff") == 1
    self._backoff_param_frame = 0

    self.steer_softstart_limit = 0
    self.steerRateLimActive = False
    self.steerRateLim = 1.0

    # 启停场景计时器：停车后累计帧数，超过阈值强制清零状态
    # 防止长时间停车后 EPS 内部状态漂移，保留 lkas_active 反而引发异常
    self.lat_inactive_frames = 0

    self.first_start = True
    self.rfss = 0
    self.sss = 0

    self.apply_accel_last = 0

    # --- OP纵向退出沿补发 Act=0 (2026-08-16 .comma(4) 验证) ---
    # 根因: 旧逻辑 should_send_acc_cmd 只在 CC.longActive 时发 ACC_CMD, OP 退出后立即
    # 停发 → ESC 检测到 ACC_CMD 流中断 → 报 AccState=7(ACC_ERROR, Notify=36) → 原车
    # 屏幕报错 + 宋Plus carstate 误判 steerFaultTemporary → 第二次快速激活被踢出。
    # 修复: 记录 longActive 下降沿, 补发 20 帧(1s@20Hz) Act=0 干净退出 ACC 状态。
    self._long_active_prev = False
    self._long_active_exit_frames = 0

    self.last_pcm_counter = -1

    # --- 仪表HUD注入 0x32D (OP纵向 2026-08-16) ---
    # OP纵向时相机待机广播死值(SetSpeed=0/AccState=1/2), 仪表不跟随OP UI设定。
    # 本注入: 50Hz(100Hz每2帧)发 ACC_HUD_ADAS 到ESC, 激活时覆盖为原车ACC激活态
    # (SetSpeed=OP设定/SetDistance=OP车距/AccState=3), 待机按原车ACC开关状态
    # 决定 SetDistance: ACC OFF(AccState=0)透传相机值(不显示档位)/
    # ACC ON(AccState=1/2)注入OP档位(与UI同步, 08-18行为)。
    self.hud_counter = 0

    # --- 旧 ACC Enhance 软阻断（当前注入模式下已废弃，保留作回退） ---
    self.acc_enhance_active = False
    self.acc_enhance_ending = False
    self.acc_enhance_accel_cache = 0.0
    # --- 按钮注入 ---

    self.btn_inject_cmd = None
    self.btn_inject_frames_left = 0
    self.btn_cooldown_frames = 0
    self.btn_resume_compensate = False
    self.btn_inject_scene = 0  # 注入发起时记录的scene，用于补偿判断（避免注入期间scene变化导致误补偿）

    # --- 测试辅助: OP纵向状态机汇总日志(状态变化时打印一行, 便于测试人员定位) ---
    # 目的: 让测试人员扫一眼 debug log 就能确认 OP 纵向是否正常工作, 不用翻代码。
    # 检测三件事: (1) OP纵向开关是否开启 (2) 原车ACC是否被成功拦截(待机) (3) OP是否激活并发ACC_CMD。
    self._long_dbg_last = {}          # 上次状态, 变化才打印
    self._long_dbg_init_logged = False
    self._long_dbg_orig_acc_conflict = 0  # OP激活期间原车ACC冲突(拦截失败)标志

    # --- 测试辅助: OP纵向异常计数(应发未发/原车ACC被激活) ---
    self._long_dbg_should_send_miss = 0   # OP激活但本帧未发ACC_CMD(条件不满足)计数
    self._long_dbg_acc_cmd_sent = 0       # OP纵向已发ACC_CMD累计

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    if (self.frame - self.last_steer_frame) >= CarControllerParams.STEER_STEP:
      if self.first_start:
        self.mpc_lkas_counter = int(CS.acc_mpc_state_counter + 1) & 0xF
        self.mpc_acc_counter = int(CS.acc_cmd_counter + 1) & 0xF
        self.eps_fake318_counter = int(CS.eps_state_counter + 1) & 0xF
        self.first_start = False

      apply_torque = 0

      # 定期刷新开关(设置页改动实时生效, 50帧@100Hz≈0.5s 一次)
      self._backoff_param_frame += 1
      if self._backoff_param_frame % 50 == 0:
        self.lateral_backoff_enabled = self.params.get_int("LateralTorqueDriverBackoff") == 1

      # 后装L2车判定: 开关保护只对后装车生效, 高配原厂车(唐/汉/秦/元等)无论开关
      # 状态都走原行为 (2026-09-04 用户: 高配车不要走这个开关)。
      retrofit_protect = self.lateral_backoff_enabled and getattr(CS, 'is_song_plus_dmi', False)

      # ===== EPS 临时故障(TemporaryFail)立即退避 (2026-09-04 20260904 行车日志分析) =====
      # 现场: 12:56:28.265 EPS 报 lkas_state=3(TemporaryFail) 预警, controlsd 同帧撤 latActive,
      #       但 CC 走 lat_safeoff 衰减路径期间 lkas_active 仍=1 → EPS 在 0.5s tmp→perm 决策
      #       窗口内收到 "LKAS_Active=1 + LKAS_Output 84→0 渐降" 矛盾状态 → 28.762 升级
      #       TorqueFailed 永久锁死(需重启车)。
      # 修复: 检测到 tmp/perm 故障立即清 lkas_active(create_steering_control 发 Act=0+Output=0
      #       干净退出信号) + 清 softstart/退避锁 + apply_torque_last 当帧归零(不走 lat_safeoff
      #       渐降, 避免决策窗内仍收到非零扭矩请求)。perm 也一并清: EPS 已死, 不应再声称激活。
      #       panda 侧 BYD 扭矩检查失败不拦截(safety_byd.h tx_hook 注释), 瞬时归零安全。
      # 门控(2026-09-04): 仅后装L2车 + LateralTorqueDriverBackoff=1 生效 —— TorqueFailed 只出在
      #       后装L2车; 开关关闭或高配原厂车都保持原 lat_safeoff 渐降行为。
      if retrofit_protect and (CS.out.steerFaultTemporary or CS.out.steerFaultPermanent) and self.lkas_active:
        acc_debug_write(f"{time.monotonic():.3f} CC LKAS: steerFault tmp={int(CS.out.steerFaultTemporary)} "
                        f"perm={int(CS.out.steerFaultPermanent)} state={CS.lkas_state} force lkas_active=0\n")
        self.lkas_active = 0
        self.steer_softstart_limit = 0
        self.driver_backoff_frames = 0
        self.apply_torque_last = 0

      if CC.latActive:
        # 起步恢复：重置启停计时器
        self.lat_inactive_frames = 0
        # 唐DM: 输出扭矩前额外检查 EPS 实时 LKAS_State prepared(1/2) (原行为不变)。
        # 宋Plus: 同样合成 2bit 枚举, 但仅 LateralTorqueDriverBackoff=1 时才查 prepared
        # (2026-09-04 用户: TorqueFailed 只出在后装L2车, 与对拉退避共用开关;
        #  关闭时宋Plus 保持旧行为不查 prepared)。
        # 汉DM/汉EV: lkas_prepared 来自 LKAS_Prepared 1bit 字段，语义不同，保持原逻辑不检查。
        check_lkas_prepared = getattr(CS, 'is_tang_dm', False) or retrofit_protect
        if self.lkas_active and (CS.lkas_prepared or not check_lkas_prepared):
          steer_desire = CC.actuators.torque

          if CarControllerParams.USE_STEERING_SPEED_LIMITER:
            rate_limit = np.interp(CS.out.aEgo, [8.3, 27.8], [132, 64])
            delta_rate = CS.steeringRateDegAbs - rate_limit

            if delta_rate < 0:
              self.steerRateLim -= 0.005 * delta_rate

              if delta_rate < -0.05:
                self.steerRateLimActive = False

              if self.steerRateLim > 1.0:
                self.steerRateLim = 1.0
                self.steerRateLimActive = False

            else:
              if self.steerRateLimActive:
                self.steerRateLim -= 0.005 * delta_rate
              else:
                self.steerRateLim = steer_desire
                self.steerRateLimActive = True

              if self.steerRateLim < 0:
                self.steerRateLim = 0

            new_steer_pu = np.clip(steer_desire, -self.steerRateLim, self.steerRateLim)
          else:
            new_steer_pu = steer_desire

          new_steer = int(round(new_steer_pu * CarControllerParams.STEER_MAX))

          if self.steer_softstart_limit < CarControllerParams.STEER_MAX:
            self.steer_softstart_limit = self.steer_softstart_limit + CarControllerParams.STEER_SOFTSTART_STEP
            new_steer = np.clip(new_steer, -self.steer_softstart_limit, self.steer_softstart_limit)

          # 防止OP/LKAS与驾驶员反向抢盘触发EPS TorqueFailed(报lkas / steerUnavailable)。
          # 现象:过匝道时OP仍在输出横向扭矩,驾驶员强行打方向,两股相反扭矩持续对拉,
          #       EPS被顶到TorqueFailed永久故障(需重启车辆才能恢复)。
          # apply_driver_steer_torque_limits允许OP在驾驶员对抗时仍输出一定反向扭矩
          # (allowance=68,multiplier=3 -> 驾驶员+148时仍允许OP输出-60),不足以避免故障。
          # 这里在最终下发前判定:当驾驶员扭矩与命令扭矩方向相反且|驾驶员扭矩|超过允许量时,
          # 视为驾驶员强制接管,立即将命令归零,交由apply_driver_steer_torque_limits按速率限制
          # 平滑回零(STEER_DELTA_DOWN),EPS不再被反向顶。同向(驾驶员帮着打)时不干预,保留LKAS助力。
          # 注:latcontrol_torque在steeringPressed时已冻结PID积分,归零期间不会积分饱和。
          driver_torque = CS.out.steeringTorque
          if (new_steer != 0 and driver_torque != 0 and
              (new_steer > 0) != (driver_torque > 0) and
              abs(driver_torque) > CarControllerParams.STEER_DRIVER_ALLOWANCE):
            # 2026-08-13 v9-BYD seg27 复现: 驾驶员持续反向施力(drv=+50~+84) vs OP 持续
            # 输出(-99~-151)对拉5s → EPS 从 state=2 直接 TorqueFailed(无 state=3 预警,
            # 3f60ff79 的 TemporaryFail 退避路径未走)。原保护仅 new_steer=0, 但 softstart
            # 已满(300)时下下帧又恢复 -300, 回零仅10/帧 → 对拉平均扭矩仍大(seg27 实测
            # |out|mean=72), EPS 被顶死。
            # 增强(后装L2车 + LateralTorqueDriverBackoff=1 时, 默认关闭, 后装L2车自行开启):
            # 清空 softstart + 0.5s 退避锁, 驾驶员持续对抗期间保持零扭矩, EPS 不再被顶。
            # 开关关闭或高配原厂车保持原行为(new_steer=0 不锁)。
            new_steer = 0
            if retrofit_protect:
              self.steer_softstart_limit = 0
              self.driver_backoff_frames = CarControllerParams.DRIVER_BACKOFF_FRAMES
          elif self.driver_backoff_frames > 0:
            # 退避锁(仅后装L2车 + LateralTorqueDriverBackoff=1 触发): 驾驶员持续对抗期间保持零扭矩,
            # softstart 也从 0 重新爬升(恢复大扭矩需重新软启动), 防止保护间歇触发时
            # new_steer 又瞬时恢复大值。
            self.driver_backoff_frames -= 1
            new_steer = 0
            self.steer_softstart_limit = 0

          # 低速停车 EPS 静止扭矩上限保护（v9-BYD seg52 复现场）
          # 根因: BYD EPS 在 vEgo<0.6 时最大可承受电机扭矩约60, 超过即永久 TorqueFailed(需重启)。
          #       seg52 在 vEgo≈0.3 仍下发 +64 回正扭矩, eps 稳态维持64~70 → 锁死。
          #       (车在动时 eps 到 -100+ 都安全, 故只限停车段。)
          # 措施: vEgo<STEER_LOW_SPEED_V 时把最终下发扭矩限到 ±STEER_LOW_SPEED_MAX(40),
          #       对应最坏 eps~45(<60 上限, 留~15 余量)。hands-off 时 eps≈cmd(1:1, 实测EPS放大≤13%),
          #       限 cmd 即限 eps。余量充足, 仍保留足够低速回正手感, 经 seg52 回放验证不触发锁死。
          if CS.out.vEgo < CarControllerParams.STEER_LOW_SPEED_V:
            new_steer = int(np.clip(new_steer, -CarControllerParams.STEER_LOW_SPEED_MAX, CarControllerParams.STEER_LOW_SPEED_MAX))

          apply_torque = apply_driver_steer_torque_limits(new_steer, self.apply_torque_last, CS.out.steeringTorque, CarControllerParams)

        else:
          # 唐DM 原行为 + 宋Plus(开关开): 若 lkas_active=1 但 EPS 已撤 prepared(未就绪/已
          # TemporaryFail)，立即退避清零，避免持续驱动未就绪的 EPS 触发 TorqueFailed 保护。
          # 汉DM/汉EV 保持原逻辑。
          self.driver_backoff_frames = 0  # 未进入扭矩输出分支, 清退避锁
          if check_lkas_prepared and self.lkas_active and not CS.lkas_prepared:
            acc_debug_write(f"{time.monotonic():.3f} CC LKAS: EPS not prepared, backing off to prepare mode\n")
            self.lkas_active = 0.0
          # 故障期间禁止重新激活 (2026-09-04 审核修复, 仅后装L2车+开关开时生效): tmp 与
          # prepared 可同时为真(宋Plus SteerWarning=1+state=2), 若不拦, 上方故障清零后
          # 此处立刻置回 1 → 50Hz 乒乓, 发出 Act=1+Output=0 矛盾帧。
          if CS.lkas_prepared and not (retrofit_protect and
                                       (CS.out.steerFaultTemporary or CS.out.steerFaultPermanent)):
            self.lkas_active = 1.0
            self.steerRateLimActive = False
            self.steerRateLim = 1.0
            self.lkas_req_prepare = 0
            self.steer_softstart_limit = 0
            self.lat_safeoff = 1
          else:
            self.lkas_req_prepare = 1
            # 注意: prepare等待期不能强制同步mpc_laks_active设lkas_active=1。
            # 原因: lkas_active=1会让bydcan发CruiseActivated=1(谎报已激活)，但实际未控车
            # (apply_torque_last=0)，EPS检测到"声称激活但无扭矩"会拒绝LKAS_Prepared=1，
            # 导致prepare握手永远完不成→横向永久丢失。
            # UI崩溃重启后card重新初始化(lkas_active=0)，若mpc_laks_active仍=1(原车MPC残留)，
            # 会卡死在此分支。正确做法: prepare等待期保持lkas_active=0，让EPS正常握手。

      elif self.lat_safeoff:
        self.driver_backoff_frames = 0  # 退避完成中, 清退避锁
        if self.apply_torque_last == 0:
          self.lat_safeoff = 0
        apply_torque = apply_driver_steer_torque_limits(0, self.apply_torque_last, CS.out.steeringTorque, CarControllerParams)

      else:
        self.driver_backoff_frames = 0  # 横向未激活(停车/握手), 清退避锁
        self.lkas_req_prepare = 0
        self.steerRateLimActive = False
        self.steerRateLim = 1.0
        # 启停场景（CC.latActive因standstill短暂为False）：保留lkas_active和
        # steer_softstart_limit，起步时直接输出扭矩，避免softstart从0爬升
        # 导致前几百毫秒方向盘没力、横向丢失。
        # 时间窗保护：停车超过 15000 帧（约5分钟@50Hz）后强制清零，
        # 覆盖几乎所有红绿灯场景，仅在极端长停时降级到原始 prepare 流程。
        # 仅当车确实在动（vEgo>0.1）或非standstill或超时才真正退出并清零。
        #
        # EPS TorqueFailed 永久故障保护（v9-BYD seg52复现修复）：
        # 根因: BYD EPS 在 CruiseActivated=1 + vEgo<0.3 + OP发lkas_active=1+LKAS_Output=0
        #       的矛盾状态下，给40ms宽限期后置位TorqueFailed永久故障（需重启车）。
        #       正常stop-and-go中EPS在vEgo≈0.93就提前撤了CruiseActivated=0，不触发。
        #       异常状态（EPS不撤CruiseActivated）下持续到vEgo<0.3就会锁死。
        # 修复: 监测EPS反馈的CruiseActivated，仅在EPS仍激活+CruiseAct=1+vEgo<0.3时
        #       立即清零lkas_active释放EPS，避免矛盾状态持续触发perm。
        #       正常stop-and-go中EPS已撤CruiseAct=0，不触发清零，lkas_active保留，
        #       起步瞬时激活，不影响跟车。
        # 唐DM: 无CruiseActivated字段(新DBC用LKAS_State枚举), 用 lkas_state==2(Active) 判断EPS激活。
        self.lat_inactive_frames += 1
        if getattr(CS, 'is_tang_dm', False):
          cruise_activated = CS.lkas_state == 2  # LKAS_State Active
        else:
          cruise_activated = bool(CS.esc_eps.get('CruiseActivated', 0)) if CS.esc_eps else False
        if (CS.out.vEgo <= 0.3 and cruise_activated) or CS.out.vEgo > 0.3 or self.lat_inactive_frames > 15000:
          self.lkas_active = 0
          self.steer_softstart_limit = 0
          self.lat_inactive_frames = 0
        # 注意: 此处不强制同步mpc_laks_active。
        # 原因: else分支覆盖"latActive=False且非safeoff"场景(行驶中横向未激活/低速过渡/握手失败)。
        # 若强制lkas_active=1，OP未控车却发CruiseActivated=1，EPS检测双源冲突→高频steerFaultTemporary
        # →controlsd高频处理+UI高频刷新告警→CPU占满→画面卡死(别人车行驶中偶发)。
        # 等红灯LKAS故障问题(standstill场景)由lat_inactive_frames>15000超时清零+重启握手解决。
      self.apply_torque_last = apply_torque

      self.mpc_lkas_counter = int(self.mpc_lkas_counter + 1) & 0xF
      self.eps_fake318_counter = int(self.eps_fake318_counter + 1) & 0xF
      self.last_steer_frame = self.frame

      can_sends.append(
        bydcan.create_steering_control(
          self.packer, self.CP, CS.cam_lkas, self.apply_torque_last, self.lkas_req_prepare, self.lkas_active, CC.hudControl, self.mpc_lkas_counter
        )
      )

      # 唐DM: fake 318 的 LKAS_State 必须反映 OP 真实控车状态(self.lkas_active),
      # 不能沿用原车 MPC 的 LKAS_Active(CS.mpc_laks_active)——唐DM 原车 MPC 该位恒 0,
      # 若照抄会发 LKAS_State=0(NotReady)+TorqueFailed, 与原车 EPS(Ready/Active)矛盾,
      # 原车 ACC 检测到伪造 EPS 状态异常 → 19s 后主动 DISENGAGE 报"自适应巡航受限"。
      fake318_active = self.lkas_active if getattr(CS, 'is_tang_dm', False) else CS.mpc_laks_active
      fake318_reqprep = self.lkas_req_prepare if getattr(CS, 'is_tang_dm', False) else CS.mpc_laks_reqprepare
      can_sends.append(
        bydcan.create_fake_318(self.packer, self.CP, CS.esc_eps, CS.mpc_laks_output, fake318_reqprep, fake318_active, True, self.eps_fake318_counter)
      )

    # --- 仪表HUD注入 0x32D (仅OP纵向, 50Hz = 100Hz循环每2帧一帧) ---
    # 相机 0x32D 是 50Hz, 待机时内容不变; OP注入用同样频率替代转发。
    # panda fwd 已阻断相机帧转ESC + 阻断OP帧回灌MPC, 仪表只收OP注入帧。
    # 激活(CC.enabled): AccState=3 + SetSpeed=OP设定速度 + SetDistance=OP车距等级
    # 待机: SetSpeed=OP保留的vCruise(暂停前设定), SetDistance=OP UI车距等级(与UI同步),
    #       AccState=相机待机值(1/2)。
    #   (2026-08-18 修复: 原透传相机死值 SetSpeed=0 → 仪表暂停后归0; 改为显示保留速度,
    #    与原车ACC cancel后仪表保留SetSpeed一致; 同版修复 SetDistance: 相机待机死值=3,
    #    透传会导致仪表暂停后回3格与UI不同步, 改为注入OP UI车距等级)
    #   (2026-08-18 档位: 原车 SetDistance 1~4 四档(1BAR~4BAR, 汉DBC
    #    byd_han_dmev_2020.dbc:347 VAL_ 813 铁证, 唐 byd_tang_dm.dbc 同),
    #    默认3档(相机待机死值=3); OP UI bars 1~4 与之一一对应 → 直通注入, 不压缩。
    #    仅非法值(0/负/超4)兜底到3。)
    if not self.CP.pcmCruise and self.frame % 2 == 0:
      cam_adas = getattr(CS, 'cam_adas', None)
      if cam_adas:
        self.hud_counter = (self.hud_counter + 1) & 0xF
        setspeed = getattr(CS.out, 'vCruise', 0.0) or 0.0
        setdist = int(CC.hudControl.leadDistanceBars)
        if not (1 <= setdist <= 4):
          setdist = 3
        if CC.enabled:
          can_sends.append(bydcan.create_acc_hud_adas(self.packer, cam_adas, float(setspeed), int(setdist), 3, self.hud_counter))
        else:
          # 待机: SetSpeed=OP vCruise(保留暂停前设定), AccState 透传相机待机值。
          # SetDistance 按"原车 ACC 是否开启"区分(2026-08-26 后装车修复, 08-28 修订):
          #   - 原车 ACC 关闭 (cam AccState=0, 相机 SetDist=0=关闭标记): 透传相机值,
          #     仪表不显示档位。原逻辑注入 OP UI 档位 → OP纵向一开/关闭ACC 仪表仍显示。
          #   - 原车 ACC 开启待机 (cam AccState=1/2): 注入 OP UI 档位
          #     (08-18 行为: 暂停后档位与 UI 同步, 防相机待机死值回跳)。
          if cam_adas['AccState'] == 0:
            can_sends.append(bydcan.create_acc_hud_adas(self.packer, cam_adas, float(setspeed), int(cam_adas['SetDistance']), cam_adas['AccState'], self.hud_counter))
          else:
            can_sends.append(bydcan.create_acc_hud_adas(self.packer, cam_adas, float(setspeed), int(setdist), cam_adas['AccState'], self.hud_counter))

    if (self.frame + 1 - self.last_acc_frame) >= CarControllerParams.ACC_STEP:
      cruise_enabled = CS.out.cruiseState.enabled

      # --- 旧 ACC Enhance 软阻断逻辑（当前 acc_enhance_request 恒 False，本段不会触发；保留作回退） ---
      if CS.acc_enhance_ending:
        self.acc_enhance_ending = True
        CS.acc_enhance_ending = False

      if CS.acc_enhance_request and cruise_enabled:
        if not self.acc_enhance_active:
          self.acc_enhance_active = True
        self.acc_enhance_accel_cache = CS.acc_enhance_accel
      elif not CS.acc_enhance_request and self.acc_enhance_active:
        self.acc_enhance_active = False
        self.acc_enhance_ending = True

      if not cruise_enabled:
        if self.acc_enhance_active:
          self.acc_enhance_active = False
          self.acc_enhance_ending = True

      # OP纵向(pcmCruise=False, 原车ACC待机): CC.longActive 即发 ACC_CMD, 不依赖原车ACC enabled。
      # (原车ACC待机时 cruiseState.enabled=False, 若保留 cruise_enabled and CC.longActive 将发不出命令)
      should_send_acc_cmd = (cruise_enabled and self.acc_enhance_active) or self.acc_enhance_ending or CC.longActive

      # OP纵向退出沿: longActive 1→0 时补发 Act=0 ACC_CMD, 让 ESC 干净退出 ACC 状态,
      # 否则 ACC_CMD 流中断 → ESC 报 AccState=7(ACC_ERROR) → 原车报错/误判转向故障。
      if CC.longActive:
        self._long_active_prev = True
      elif self._long_active_prev:
        # 下降沿: 启动 1s(20Hz*20帧) Act=0 补发
        self._long_active_prev = False
        self._long_active_exit_frames = 20

      if self._long_active_exit_frames > 0:
        should_send_acc_cmd = True
        self._long_active_exit_frames -= 1
        long_active_exit = True

      accel_to_send = 0.0
      long_active = False

      if self.acc_enhance_ending:
        # 旧 ACC Enhance 结束清理（当前不会触发，保留作回退）
        can_sends.append(bydcan.acc_cmd(self.packer, self.CP, CS.cam_acc, CS.mrr_leading_dist, 0, 0, 0, False))
        self.acc_enhance_ending = False
        self.acc_enhance_accel_cache = 0.0

      elif should_send_acc_cmd:
        if CC.longActive:
          accel_to_send = np.clip(CC.actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)
          long_active = True

          stopping = CC.actuators.longControlState == LongCtrlState.stopping

          if stopping and accel_to_send < -0.1:
            self.rfss = 0
            self.sss = CS.out.standstill
          elif accel_to_send > 0.1 and CS.out.vEgo < 0.8:
            # 起步 rfss 保持 (2026-09-02 .comma(12) 实锤): starting 及 starting→pid 过渡期
            # (车速仍<0.8m/s) 都必须 rfss=1 不撤。原车 ACC 起步全程 rfss=1 不撤回;
            # 旧逻辑 pid 切换瞬间 rfss 掉 0 → ESC 重新拉起 standstill 保持 → 0.4km/h 爬行中
            # 二次停死 ~1s (19:53:20-23 起步抖动实锤)。原车对照: 原车起步 rfss 连续到车动。
            self.rfss = 1
            self.sss = 0
          else:
            self.rfss = 0
            self.sss = 0

        elif self.acc_enhance_active:
          # 旧 ACC Enhance 加速度分支（当前不会触发，保留作回退）
          accel_to_send = np.clip(self.acc_enhance_accel_cache, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)
          long_active = True
          self.rfss = 0
          self.sss = CS.out.standstill if accel_to_send < -0.1 else 0

        if long_active:
          # 2026-10-31: sptn 无 CC.actuators.jerk, 用请求加速度对时间的差分估算意图 jerk
          # (语义对拍 v9-byd 的 MPC.jerk: 泄压上限 key, 见 bydcan.acc_cmd 注释)。
          if self._last_accel_t_ns is not None and now_nanos > self._last_accel_t_ns:
            dt = (now_nanos - self._last_accel_t_ns) * 1e-9
            if dt > 0.0:
              self._jerk_est = 0.7 * self._jerk_est + 0.3 * ((accel_to_send - self.apply_accel_last) / dt)
          self._last_accel_t_ns = now_nanos
          can_sends.append(bydcan.acc_cmd(self.packer, self.CP, CS.cam_acc, CS.mrr_leading_dist,
                                          accel_to_send, self.rfss, self.sss, True,
                                          jerk_mpc=self._jerk_est))
          self.apply_accel_last = accel_to_send
          # ===== 诊断日志: OP纵向发ACC_CMD =====（异步写文件, 不用 print 防止 pty 卡死控制循环）
          if self.frame % 50 == 0:
            acc_debug_write(f"DBG_LONG OP发814: frame={self.frame} accel={accel_to_send:.2f} longActive={CC.longActive} "
                            f"cruiseEn={CS.out.cruiseState.enabled} vEgo={CS.out.vEgo:.1f} acc_state={CS.acc_state} "
                            f"mrr_dist={CS.mrr_leading_dist} lkas_active={self.lkas_active}\n")

        elif long_active_exit:
          # OP纵向退出沿补发 Act=0, 让 ESC 干净退出 ACC 状态 (2026-08-16 .comma(4) 验证)
          can_sends.append(bydcan.acc_cmd(self.packer, self.CP, CS.cam_acc, CS.mrr_leading_dist, 0.0, 0, 0, False))

      self.last_acc_frame = self.frame + 1

      # ===== 测试辅助: OP纵向状态机汇总日志(状态变化时打印一行, 便于测试人员定位) =====
      # 在 ACC_STEP 块内执行, 与 long_active/accel_to_send 同帧, 20Hz。
      # 打印字段让测试人员一眼确认:
      #   sw=OP纵向开关 longAct=OP是否激活 origAcc=原车ACC是否被激活(应为0=拦截成功)
      #   accCmd=OP是否在发ACC_CMD(激活时应为1) miss=应发未发计数
      try:
        if not self.CP.pcmCruise:
          if not self._long_dbg_init_logged:
            self._long_dbg_init_logged = True
            acc_debug_write(f"OP_LONG MODE: pcmCruise={self.CP.pcmCruise} openpilotLongitudinalControl={self.CP.openpilotLongitudinalControl} "
                            f"拦截方式=全程(not pcmCruise) 测试检查点: [sw=1] [origAcc=0] [accCmd=1 when activating]\n")
          state = {
            "longAct": int(CC.longActive),
            "origAcc": int(CS.out.cruiseState.enabled),
            "accCmd": int(long_active),
          }
          # 异常检测: OP激活但本帧没发ACC_CMD(条件意外不满足)
          if CC.longActive and not long_active:
            self._long_dbg_should_send_miss += 1
          # 汇总: OP纵向激活期间累计发了多少次ACC_CMD
          if long_active:
            self._long_dbg_acc_cmd_sent += 1
          # 状态变化: 打印 + OP激活结束(1->0)时打印累计统计供测试核对
          prev_la = self._long_dbg_last.get('longAct', 0)
          if state != self._long_dbg_last:
            acc_debug_write(f"OP_LONG {time.monotonic():.3f} sw=1 longAct={state['longAct']} origAcc={state['origAcc']} "
                            f"accCmd={state['accCmd']} miss={self._long_dbg_should_send_miss} "
                            f"vEgo={CS.out.vEgo*3.6:.0f} acc_state={CS.acc_state} accel={accel_to_send:+.2f}\n")
            self._long_dbg_last = state
            # 异常高亮: OP纵向激活期间原车ACC被激活(拦截失败) —— 测试人员最需要看到的
            if state['longAct'] == 1 and state['origAcc'] == 1:
              self._long_dbg_orig_acc_conflict = 1
              acc_debug_write(f"!!! OP_LONG 异常: OP激活期间原车ACC也被激活(origAcc=1)!! 拦截失效, 会触发相机监督报错。"
                              f" vEgo={CS.out.vEgo*3.6:.0f} acc_state={CS.acc_state}\n")
          if prev_la == 1 and state['longAct'] == 0:
            acc_debug_write(f"OP_LONG 结束: 本次OP纵向激活共发ACC_CMD {self._long_dbg_acc_cmd_sent} 帧, "
                            f"应发未发miss={self._long_dbg_should_send_miss}(应=0) "
                            f"期间原车ACC冲突过? {self._long_dbg_orig_acc_conflict}(应=0)\n")
            # 复位本次会话累计, 供下次激活重新统计
            self._long_dbg_acc_cmd_sent = 0
            self._long_dbg_orig_acc_conflict = 0
      except Exception:
        pass

    if CS.pcm_buttons_vals:
      cur_counter = CS.pcm_buttons_vals.get("Counter", -1)
      _cnt_chg = cur_counter != self.last_pcm_counter
      # 诊断: 注入流程日志(每帧记录, 看20Hz处理帧+门控原因+注入状态) —— 异步写盘, 不阻塞 100Hz 循环
      # cnt=1时是20Hz处理帧(counter变化), gate=_get_button_injection返回原因
      # gate: s1_set=注入SET/- s2_res=注入RES/+ no_en=enabled挡 btn=按键挡 cd=冷却 none=无scene
      acc_debug_write(f"{time.monotonic():.3f} INJECT cnt={1 if _cnt_chg else 0} left={self.btn_inject_frames_left} cd={self.btn_cooldown_frames} gate={getattr(self,'_last_gate','na')} scene={CS.acc_enhance_scene} en={CS.out.cruiseState.enabled} btn={CS.btn_acc_set_reset} resume={CC.cruiseControl.resume}\n")
      if _cnt_chg:
        self.last_pcm_counter = cur_counter
        # OP纵向(pcmCruise=False): 全程拦截拨杆按钮(激活前也拦), 让原车ACC永远无法启用。
        # 2026-08-16 .comma(6) 日志验证: 旧逻辑(仅CC.longActive时拦)在OP激活前驾驶员
        # 第一次按SET/RES会穿透 → 原车ACC被启用(en=True, setKph=30) → 相机进入监督。
        # 改为 OP纵向模式下无条件拦截(not self.CP.pcmCruise): 原车永远收不到激活指令
        # → en保持False → 相机始终待机 → 不报ACC_ERROR。OP自己的按钮启停从carstate
        # 读原始按钮(carstate.py buttonEvents), 不受fwd拦截影响, 照常工作。
        if not self.CP.pcmCruise or CC.longActive:
          # 拦截原车按钮(拨杆SET/RES/车距/cancel), 由OP控制发什么给相机。
          # 只透传 toggle(ACC开关保持on), 拨杆清零 → 原车ACC保持待机(AccControlActive=0)
          # → 相机不进入指令-响应监督 → 不报ACC_ERROR。
          fwd_vals = dict(CS.pcm_buttons_vals)
          # 只拦"拨杆"按钮: BTN_AccUpDown_Cmd(SET/RES=激活/调速) + BTN_AccCancel(退出)。
          # 保留 BTN_TOGGLE_ACC_OnOff 透传: ACC开关保持on(可用), 只是不激活。
          # 相机保持待机(AccControlActive=0) → 不监督 → 不报ACC_ERROR。
          fwd_vals["BTN_AccUpDown_Cmd"] = 0
          fwd_vals["BTN_AccCancel"] = 0
          fwd_vals["BTN_AccDistanceDecrease"] = 0
          fwd_vals["BTN_AccDistanceIncrease"] = 0
        else:
          fwd_vals = dict(CS.pcm_buttons_vals)

        if self.btn_cooldown_frames > 0:
          self.btn_cooldown_frames -= 1

        if not self.CP.pcmCruise or CC.longActive:
          # OP纵向: 拦截原车按钮后整个注入机制跳过, 只发中性帧(已在上面 fwd_vals 初始化)。
          # 防止残留 btn_inject_frames_left/btn_resume_compensate 覆盖中性值激活原车ACC。
          self.btn_inject_frames_left = 0
          self.btn_resume_compensate = False
        elif self.btn_inject_frames_left > 0:
          ud, cancel, dinc, ddec = self.btn_inject_cmd
          fwd_vals["BTN_AccUpDown_Cmd"] = ud
          fwd_vals["BTN_AccCancel"] = cancel
          fwd_vals["BTN_AccDistanceIncrease"] = dinc
          fwd_vals["BTN_AccDistanceDecrease"] = ddec
          self.btn_inject_frames_left -= 1
          if self.btn_inject_frames_left == 0:
            # Resume补偿: RES/+完成后立即注入SET/-减回速度
            # 补偿判断只看 btn_inject_scene(注入发起时记录的scene), 不看 CS.acc_enhance_scene:
            # - standstill时 _update_acc_enhance 不被调用(enabled=False), scene 是残留值不可靠
            # - btn_inject_scene==0: resume 注入, 必须补偿(否则SetSpeed飙升)
            # - btn_inject_scene==2: 弯道恢复的RES/+, 不补偿(需要升速)
            # - btn_inject_scene==1: 弯道减速的SET/-, 不会是 ud==3
            if ud == 3 and not self.btn_resume_compensate and self.btn_inject_scene != 2:
              self.btn_resume_compensate = True
            else:
              self.btn_resume_compensate = False
              # 冷却从7降到4(v=32): 原7帧+3帧注入=500ms/轮只降5km/h, 80→50要6轮3s,
              # 高速接近弯道时降速太慢(减速来不及, 弯中扭矩饱和UI提示)。
              # 4帧冷却+3帧注入=350ms/轮, 6轮从3s缩到2.1s。原车ACC对350ms间隔SET/-
              # 的响应需实车验证(跟停RES/+连续注入已验证可靠)。
              # v=36: scene=1且剩余gap>10km/h时冷却2帧(250ms/轮, 75→55从1.4s缩到1.0s),
              # 收紧弯抢减速时间; gap≤10或scene=2恢复保持4帧(温和节奏, 防按过头)。
              if self.btn_inject_scene == 1 and \
                 (CS.out.cruiseState.speed * 3.6 - CS.acc_enhance_accel) > 10:
                self.btn_cooldown_frames = 2
              else:
                self.btn_cooldown_frames = 4
        elif self.btn_resume_compensate:
          # 注入SET/-补偿RES/+导致的速度+10
          # btn_resume_compensate 只在 btn_inject_scene!=2 时设为True, 所以这里直接补偿。
          # 不再检查 CS.acc_enhance_scene==2: standstill时scene是残留值不可靠,
          # 且补偿循环期间 scene 可能变化, 用注入时的 btn_inject_scene 判断更准确。
          self.btn_inject_cmd = (1, 0, 0, 0)
          self.btn_inject_frames_left = 3
          self.btn_resume_compensate = False
        else:
          inject = self._get_button_injection(CC, CS)
          if inject is not None:
            self.btn_inject_cmd = inject
            self.btn_inject_frames_left = 3
            # 注入发起时记录 scene, 用于补偿判断。
            # resume 注入记录 scene=0: standstill起步的RES/+必须补偿SET/-,
            # 不能因为跟停前 scene 残留=2(弯道恢复)而跳过补偿导致 SetSpeed 飙升。
            # scene=1/2 注入记录实际 scene: 弯道恢复(scene=2)的RES/+不补偿。
            if CC.cruiseControl.resume:
              self.btn_inject_scene = 0
            else:
              self.btn_inject_scene = CS.acc_enhance_scene
            ud, cancel, dinc, ddec = inject
            fwd_vals["BTN_AccUpDown_Cmd"] = ud
            fwd_vals["BTN_AccCancel"] = cancel
            fwd_vals["BTN_AccDistanceIncrease"] = dinc
            fwd_vals["BTN_AccDistanceDecrease"] = ddec

        can_sends.append(bydcan.create_button_fwd(self.packer, fwd_vals))

    new_actuators = CC.actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / CarControllerParams.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    new_actuators.accel = float(self.apply_accel_last)
    new_actuators.steeringAngleDeg = float(CS.out.steeringAngleDeg)

    self.frame += 1
    return new_actuators, can_sends

  def _get_button_injection(self, CC, CS):
    # OP纵向(pcmCruise=False)下绝不注入任何按钮 (2026-08-15/16 日志验证):
    # OP纵向下 _update_acc_enhance已return但acc_enhance_scene残留, 残留scene会触发
    # 下方 SET/RES 注入 → 原车ACC收到SET被激活(AccControlActive=1) → 相机进入
    # 指令-响应监督 → 发现命令被OP拦截 → ACC_ERROR退出。
    # .comma(6): 改为全程拦截(不只在激活时)——OP纵向模式下任何按钮注入都禁止,
    # 原车ACC保持待机(AccControlActive=0) → 相机不监督 → 不报错。
    if not self.CP.pcmCruise or CC.longActive:
      self._last_gate = "op_long"
      return None

    # cancel 优先于所有检查: 驾驶员退出巡航时需无条件注入 cancel 按钮给原车ACC。
    # (CC.cruiseControl.cancel 自身已含 cruiseState.enabled 条件, 不会误触发)
    if CC.cruiseControl.cancel:
      self.btn_cooldown_frames = 0
      self._last_gate = "cancel"
      return (0, 1, 0, 0)

    # cruise已退出时不注入 scene（_update_acc_enhance不再被调用，scene可能残留旧值）。
    if not CS.out.cruiseState.enabled and not CC.cruiseControl.resume:
      self._last_gate = "no_en"
      return None

    # 驾驶员正在按键时不注入
    if CS.btn_acc_set_reset != 0 or CS.btn_acc_cancel != 0:
      self._last_gate = "btn"
      return None

    if self.btn_cooldown_frames > 0:
      self._last_gate = "cd"
      return None

    # 跟停恢复: 注入 RES/+ 让原车ACC起步。
    if CC.cruiseControl.resume:
      self._last_gate = "resume"
      return (3, 0, 0, 0)

    # 导航/弯道减速: 注入 SET/- 让原车ACC降SetSpeed
    if CS.acc_enhance_scene == 1:
      current_set_kph = CS.out.cruiseState.speed * 3.6
      target_kph = CS.acc_enhance_accel  # 复用字段传 target_speed
      if current_set_kph > target_kph + 2:
        self._last_gate = "s1_set"
        return (1, 0, 0, 0)
      self._last_gate = "s1_skip"

    # 出弯/限速恢复: 注入 RES/+ 让原车ACC升SetSpeed
    elif CS.acc_enhance_scene == 2:
      current_set_kph = CS.out.cruiseState.speed * 3.6
      target_kph = CS.acc_enhance_accel
      if current_set_kph < target_kph - 2:
        self._last_gate = "s2_res"
        return (3, 0, 0, 0)
      self._last_gate = "s2_skip"
    else:
      self._last_gate = "none"

    return None

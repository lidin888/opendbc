"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntFlag


class BYDFlagsSP(IntFlag):
  """BYD sunnypilot 专属标志位。"""
  # 【2026-11-18】纵向调校开关（UI: 车辆 → BYD → 纵向调校）
  #   ON  → planner 使用 BYD 运动加速表 (2.0/1.6/1.1/0.8) + 减速 clip −0.9 (回收为主, 节能)
  #   OFF → 完全官方 stock 行为 (1.6/1.2/0.8/0.6, 减速 −1.2)
  # 只对 CP.brand == 'byd' 设置, 其他车型永远 OFF。
  LONG_TUNING = 1

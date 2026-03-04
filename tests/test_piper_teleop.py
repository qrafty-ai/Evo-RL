from types import SimpleNamespace

import lerobot.utils.piper_sdk as piper_sdk_utils
import lerobot.robots.piper_follower.piper_follower as piper_follower_module
import lerobot.teleoperators.piper_leader.piper_leader as piper_leader_module
from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.piper_leader import PiperLeader, PiperLeaderConfig
from lerobot.teleoperators.utils import make_teleoperator_from_config


class FakeLogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SILENT = "SILENT"


class FakePiperInterface:
    def __init__(self, can_name, judge_flag=False, can_auto_init=True, logger_level=None):
        self.can_name = can_name
        self.judge_flag = judge_flag
        self.can_auto_init = can_auto_init
        self.logger_level = logger_level
        self.connected = False
        self.mode_commands = []
        self.role_commands = []
        self.last_joint = None
        self.last_gripper = None
        self.enable_calls = 0
        self.disable_calls = 0

        self._joint_ctrl = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            joint_ctrl=SimpleNamespace(
                joint_1=10000,
                joint_2=20000,
                joint_3=30000,
                joint_4=40000,
                joint_5=50000,
                joint_6=60000,
            ),
        )
        self._joint_state = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            joint_state=SimpleNamespace(
                joint_1=11000,
                joint_2=21000,
                joint_3=31000,
                joint_4=41000,
                joint_5=51000,
                joint_6=61000,
            ),
        )
        self._gripper_ctrl = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            gripper_ctrl=SimpleNamespace(grippers_angle=42000, grippers_effort=1500, status_code=0x01),
        )
        self._gripper_state = SimpleNamespace(
            time_stamp=1.0,
            Hz=120.0,
            gripper_state=SimpleNamespace(grippers_angle=43000, grippers_effort=1400, status_code=0x01),
        )

    def ConnectPort(self):
        self.connected = True

    def DisconnectPort(self, thread_timeout=0.1):
        del thread_timeout
        self.connected = False

    def MotionCtrl_2(self, *args):
        self.mode_commands.append(args)

    def MasterSlaveConfig(self, *args):
        self.role_commands.append(args)

    def EnablePiper(self):
        self.enable_calls += 1
        return True

    def DisableArm(self, motor_num):
        del motor_num
        self.disable_calls += 1

    def JointCtrl(self, *args):
        self.last_joint = args

    def GripperCtrl(self, *args):
        self.last_gripper = args

    def GetArmJointCtrl(self):
        return self._joint_ctrl

    def GetArmJointMsgs(self):
        return self._joint_state

    def GetArmGripperCtrl(self):
        return self._gripper_ctrl

    def GetArmGripperMsgs(self):
        return self._gripper_state


def patch_fake_sdk(monkeypatch):
    fake_loader = lambda: (FakePiperInterface, FakeLogLevel)
    monkeypatch.setattr(piper_sdk_utils, "get_piper_sdk", fake_loader)
    monkeypatch.setattr(piper_follower_module, "get_piper_sdk", fake_loader)
    monkeypatch.setattr(piper_leader_module, "get_piper_sdk", fake_loader)


def test_piper_leader_follower_teleop_roundtrip(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop_cfg = PiperLeaderConfig(
        port="can1",
        set_leader_mode_on_connect=True,
        manual_control=False,
        sync_gripper=True,
    )
    robot_cfg = PiperFollowerConfig(
        port="can0",
        set_follower_mode_on_connect=True,
        sync_gripper=True,
    )

    teleop = make_teleoperator_from_config(teleop_cfg)
    robot = make_robot_from_config(robot_cfg)

    assert isinstance(teleop, PiperLeader)
    assert isinstance(robot, PiperFollower)

    teleop.connect()
    robot.connect()
    try:
        action = teleop.get_action()
        sent = robot.send_action(action)
        obs = robot.get_observation()

        assert robot.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert robot.arm.last_gripper == (
            42000,
            robot_cfg.gripper_effort_default,
            robot_cfg.gripper_status_code,
            0x00,
        )
        assert sent["joint_1.pos"] == 10.0
        assert sent["gripper.pos"] == 42.0
        assert obs["joint_1.pos"] == 11.0
        assert obs["gripper.pos"] == 43.0

        teleop.send_feedback(action)
        assert teleop.arm.last_joint == (10000, 20000, 30000, 40000, 50000, 60000)
        assert teleop.arm.last_gripper == (
            42000,
            teleop_cfg.gripper_effort_default,
            teleop_cfg.gripper_status_code,
            0x00,
        )
    finally:
        teleop.disconnect()
        robot.disconnect()


def test_piper_calibration_is_noop(monkeypatch):
    patch_fake_sdk(monkeypatch)

    teleop = PiperLeader(PiperLeaderConfig(port="can1"))
    robot = PiperFollower(PiperFollowerConfig(port="can0"))

    assert teleop.is_calibrated
    assert robot.is_calibrated

    teleop.calibrate()
    robot.calibrate()

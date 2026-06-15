import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


# ################################
# Joint name heuristics (embedded chassis vs NullMobileBase vs arm)
# ################################
def _is_mobile_base_joint(joint_name: str) -> bool:
    return "mobile" in joint_name.lower()


def _is_chassis_passive_joint(joint_name: str) -> bool:
    name = joint_name.lower()
    return "steering" in name or "wheel" in name


def _is_arm_joint(joint_name: str) -> bool:
    return "cr10_joint" in joint_name.lower()


# ################################
# End joint name heuristics
# ################################


# ################################
# RangerboxCR10Lidar robot model
# ################################
class RangerboxCR10Lidar(ManipulatorModel):
    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(
            xml_path_completion("robots/rangerboxcr10lidar/robot.xml"), idn=idn
        )

    @property
    def default_base(self):
        return "NullMobileBase"

    @property
    def default_gripper(self):
        # AG95 joints/actuators live in robot.xml; use adapter (no MJCF merge).
        return {"right": "EmbeddedAG95"}

    @property
    def default_controller_config(self):
        return {"right": "default_rangerboxcr10lidar"}

    @property
    def init_qpos(self):
        # ################################
        # CR10 home-ish pose (rad); gripper finger1 open at 0
        # ################################
        qpos = np.zeros(len(self.joints))
        home = {
            "cr10_joint1": 0.0,
            "cr10_joint2": -0.3,
            "cr10_joint3": 0.75,
            "cr10_joint4": 0.0,
            "cr10_joint5": 0.45,
            "cr10_joint6": 0.0,
            "gripper_finger1_joint": 0.0,
        }
        for i, joint in enumerate(self.joints):
            for key, val in home.items():
                if key in joint:
                    qpos[i] = val
                    break
        return qpos

    # ################################
    # Minimal required manipulator metadata
    # ################################
    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        # Arm flange body in converted MJCF (gripper mounted on cr10_Link6).
        return {"right": "cr10_Link6"}

    @property
    def base_xpos_offset(self):
        # ################################
        # Larger footprint than PandaOmron; stand farther from counters/tables.
        # ################################
        return {
            "bins": (-0.9, -0.1, 0.0),
            "empty": (-1.1, 0.0, 0.0),
            "table": lambda table_length: (-0.35 - table_length / 2, 0.0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0.0, 1.2))

    @property
    def _horizontal_radius(self):
        # ################################
        # ~0.9 m half-width of Ranger chassis (placement / collision checks).
        # ################################
        return 0.95

    def update_joints(self):
        # ################################
        # Partition embedded chassis vs CR10 arm (not default all-non-base → arm)
        # ################################
        self._base_joints = []
        self._torso_joints = []
        self._head_joints = []
        self._legs_joints = []
        self._arms_joints = []

        for joint in self.all_joints:
            if "torso" in joint:
                self._torso_joints.append(joint)
            elif "head" in joint:
                self._head_joints.append(joint)
            elif "leg" in joint or _is_chassis_passive_joint(joint):
                self._legs_joints.append(joint)
            elif _is_mobile_base_joint(joint):
                self._base_joints.append(joint)
            elif _is_arm_joint(joint):
                self._arms_joints.append(joint)
            # passive mimic gripper joints: equality-driven, not in arm list

    def update_actuators(self):
        # ################################
        # Actuators: CR10 + gripper finger1 only (matches robot.xml)
        # ################################
        self._base_actuators = []
        self._torso_actuators = []
        self._head_actuators = []
        self._legs_actuators = []
        self._arms_actuators = []

        for actuator in self.all_actuators:
            if _is_mobile_base_joint(actuator):
                self._base_actuators.append(actuator)
            elif "cr10_joint" in actuator:
                self._arms_actuators.append(actuator)

    # ################################
    # End minimal required metadata
    # ################################


# ################################
# End of RangerboxCR10Lidar model
# ################################

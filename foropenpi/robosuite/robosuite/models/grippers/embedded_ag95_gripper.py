# ################################
# Python: AG95 gripper embedded in Rangerbox MJCF (no separate merge) begin
# ################################
"""
Gripper adapter for AG95 already present in rangerboxcr10lidar/robot.xml.
Does not merge extra MJCF; wires SimpleGrip to gripper_finger1_joint_act.
"""

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class EmbeddedAG95Gripper(GripperModel):
    embedded_in_robot = True

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/null_gripper.xml"), idn=idn)
        # Single-robot Kitchen demos use robot0_ prefix on the manipulator MJCF.
        self._robot_prefix = "robot0_"

    @property
    def dof(self):
        return 1

    @property
    def joints(self):
        return [f"{self._robot_prefix}gripper_finger1_joint"]

    @property
    def actuators(self):
        return [f"{self._robot_prefix}gripper_finger1_joint_act"]

    @property
    def important_sites(self):
        site = f"{self._robot_prefix}right_center"
        return {"grip_site": site, "grip_cylinder": site}

    @property
    def sites(self):
        return [f"{self._robot_prefix}right_center"]

    def set_sites_visibility(self, sim, visible):
        site = f"{self._robot_prefix}right_center"
        sid = sim.model.site_name2id(site)
        if (visible and sim.model.site_rgba[sid][3] < 0) or (
            not visible and sim.model.site_rgba[sid][3] > 0
        ):
            sim.model.site_rgba[sid][3] = -sim.model.site_rgba[sid][3]

    @property
    def important_sensors(self):
        # Sensors are on robot MJCF (not gripper0_* prefixed).
        return {
            "force_ee": f"{self._robot_prefix}force_ee",
            "torque_ee": f"{self._robot_prefix}torque_ee",
        }

    @property
    def init_qpos(self):
        return np.array([0.0])

    def format_action(self, action):
        # GRIP + position actuator: keep teleop as open (-1) / close (+1).
        a = float(np.asarray(action).reshape(-1)[0])
        return np.array([-1.0 if a <= 0 else 1.0])


# ################################
# Python: AG95 gripper embedded in Rangerbox MJCF end
# ################################

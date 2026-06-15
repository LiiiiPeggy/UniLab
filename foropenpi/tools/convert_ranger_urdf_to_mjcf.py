#!/usr/bin/env python3
# ################################
# Python: ranger URDF -> MJCF converter begin
# ################################

import argparse
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import trimesh


# ################################
# Python: AG95 gripper mimic -> MuJoCo joint equality begin
# ################################
def _parse_urdf_gripper_mimics(urdf_path: Path) -> list[tuple[str, str, float]]:
    """
    Parse URDF <mimic> tags (same semantics as ROS robot_state_publisher / Gazebo plugins).
    Returns (slave_joint, master_joint, multiplier) with q_slave = mult * q_master + offset.
    """
    specs: list[tuple[str, str, float]] = []
    tree = ET.parse(urdf_path)
    for joint in tree.findall(".//joint"):
        slave = joint.get("name")
        mimic = joint.find("mimic")
        if slave is None or mimic is None:
            continue
        master = mimic.get("joint")
        if not master:
            continue
        mult = float(mimic.get("multiplier", "1"))
        specs.append((slave, master, mult))
    return specs


def _inject_gripper_equalities(
    root: ET.Element, joint_names: set, urdf_path: Path
) -> None:
    """
    MuJoCo joint equality: polycoef(q_master) = q_slave  =>  q_slave = mult * q_master.
    Matches URDF mimic and Gazebo libroboticsgroup_gazebo_mimic_joint_plugin (leader: finger1).
    """
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    existing = {(e.get("joint1"), e.get("joint2")) for e in equality.findall("joint")}
    for slave, master, mult in _parse_urdf_gripper_mimics(urdf_path):
        if slave not in joint_names or master not in joint_names:
            continue
        key = (master, slave)
        if key in existing:
            continue
        # URDF: q_slave = mult * q_master (ROS/Gazebo). MuJoCo polycoef(q_m)=q_s uses
        # linear_coef * q_master = q_slave. Symmetric knuckle (finger2) matches URDF mult;
        # internal continuous joints need 1/mult due to opposite axis in converted MJCF.
        linear_coef = mult if slave == "gripper_finger2_joint" else (1.0 / mult)
        ET.SubElement(
            equality,
            "joint",
            {
                "joint1": master,
                "joint2": slave,
                "polycoef": f"0 {linear_coef} 0 0 0",
                "solref": "0.0005 1",
                "solimp": "0.99 0.999 0.001",
            },
        )


def _wrap_worldbody_single_root(root: ET.Element) -> None:
    """Robosuite requires exactly one root body under worldbody."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    bodies = [c for c in worldbody if c.tag == "body"]
    if len(bodies) <= 1:
        return
    wrapper = ET.Element("body", {"name": "base", "pos": "0 0 0"})
    for child in list(worldbody):
        worldbody.remove(child)
        wrapper.append(child)
    worldbody.append(wrapper)


# ################################
# Python: robosuite / mjviewer visual geom groups (group 1) begin
# ################################
def _tag_robosuite_visual_geoms(root: ET.Element) -> None:
    """
    MjviewerRenderer sets viewer.opt.geomgroup[0] = 0 (hide collision group).
    Panda uses group=\"1\" for visuals; URDF->MJCF defaults to group 0 (invisible).
    Do not set contype/conaffinity to 0 here — that removes geoms from mass inference
    and breaks Kitchen load (base has no <inertial>).
    """
    for geom in root.findall(".//worldbody//geom"):
        if geom.get("group") == "0":
            continue
        geom.set("group", "1")


# ################################
# Python: base_link inertial for merged MJCF base body begin
# ################################
def _inject_base_body_inertial(root: ET.Element) -> None:
    """URDF base_link inertial; required when visual geoms cannot supply body mass."""
    base = root.find('.//body[@name="base"]')
    if base is None or base.find("inertial") is not None:
        return
    ET.SubElement(
        base,
        "inertial",
        {
            "pos": "-0.0169024 0.00678181 0.0577617",
            "quat": "1 0 0 0",
            "mass": "88.7576",
            "diaginertia": "1.71234 4.90028 6.39425",
        },
    )


# ################################
# Python: base_link inertial end
# ################################


# ################################
# Python: robosuite / mjviewer visual geom groups end
# ################################


# ################################
# Python: AG95 gripper mimic -> MuJoCo joint equality end
# ################################


def _rewrite_urdf_mesh_paths(urdf_text: str) -> str:
    """
    Rewrite package mesh URIs to local OBJ basenames so MuJoCo can parse reliably.
    Example:
      package://rangerboxcr10lidar_description/meshes/ranger_meshes/ranger_base_link.STL
      -> ranger_base_link.obj
    """
    return re.sub(
        r'package://rangerboxcr10lidar_description/meshes/(?:.*/)?([^/"\']+)\.STL',
        r"\1.obj",
        urdf_text,
    )


def _export_obj_meshes(mesh_root: Path, output_mesh_dir: Path) -> int:
    output_mesh_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for stl_file in mesh_root.rglob("*.STL"):
        obj_file = output_mesh_dir / f"{stl_file.stem}.obj"
        if obj_file.exists():
            continue
        mesh = trimesh.load_mesh(stl_file, force="mesh")
        mesh.export(obj_file, file_type="obj")
        converted += 1
    return converted


def convert(
    urdf_path: Path, mesh_root: Path, out_xml: Path, out_mesh_dir: Path
) -> None:
    urdf_text = urdf_path.read_text(encoding="utf-8")
    rewritten = _rewrite_urdf_mesh_paths(urdf_text)

    # Use a temp work dir so MuJoCo resolves local OBJ files next to URDF.
    with tempfile.TemporaryDirectory(prefix="ranger_mjcf_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_mesh_dir = tmp_dir / "meshes"
        _export_obj_meshes(mesh_root, tmp_mesh_dir)

        # Copy all OBJ files flat into temp root (matching rewritten basenames)
        for obj_file in tmp_mesh_dir.glob("*.obj"):
            shutil.copy2(obj_file, tmp_dir / obj_file.name)

        tmp_urdf = tmp_dir / "model.urdf"
        tmp_urdf.write_text(rewritten, encoding="utf-8")

        model = mujoco.MjModel.from_xml_path(str(tmp_urdf))
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveLastXML(str(out_xml), model)

    # Export persistent mesh assets for runtime loading.
    _export_obj_meshes(mesh_root, out_mesh_dir)

    # ################################
    # Python: rewrite mesh file paths to meshes/ prefix begin
    # ################################
    tree = ET.parse(out_xml)
    root = tree.getroot()
    for mesh in root.findall(".//asset/mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        if "/" in file_attr:
            continue
        mesh.set("file", f"meshes/{file_attr}")

    # ################################
    # Python: CR10 motor + gripper position actuators begin
    # ################################
    CR10_ARM_JOINTS = [
        "cr10_joint1",
        "cr10_joint2",
        "cr10_joint3",
        "cr10_joint4",
        "cr10_joint5",
        "cr10_joint6",
    ]
    CR10_MOTOR_CTRLRANGE = {
        "cr10_joint1": (-80.0, 80.0),
        "cr10_joint2": (-80.0, 80.0),
        "cr10_joint3": (-80.0, 80.0),
        "cr10_joint4": (-80.0, 80.0),
        "cr10_joint5": (-40.0, 40.0),
        "cr10_joint6": (-40.0, 40.0),
    }

    def _upsert_ranger_actuators(root_el: ET.Element, joint_names_set: set) -> None:
        actuator_el = root_el.find("actuator")
        if actuator_el is None:
            actuator_el = ET.SubElement(root_el, "actuator")
        for child in list(actuator_el):
            j = child.get("joint")
            if j in CR10_ARM_JOINTS or j == "gripper_finger1_joint":
                actuator_el.remove(child)
        for joint_name in CR10_ARM_JOINTS:
            if joint_name not in joint_names_set:
                continue
            lo, hi = CR10_MOTOR_CTRLRANGE[joint_name]
            ET.SubElement(
                actuator_el,
                "motor",
                {
                    "name": f"{joint_name}_act",
                    "joint": joint_name,
                    "ctrllimited": "true",
                    "ctrlrange": f"{lo} {hi}",
                },
            )
        if "gripper_finger1_joint" in joint_names_set:
            ET.SubElement(
                actuator_el,
                "position",
                {
                    "name": "gripper_finger1_joint_act",
                    "joint": "gripper_finger1_joint",
                    "kp": "500",
                    "ctrlrange": "0 0.65",
                },
            )

    joint_names = {j.get("name") for j in root.findall(".//joint") if j.get("name")}
    _upsert_ranger_actuators(root, joint_names)
    # ################################
    # Python: CR10 motor + gripper position actuators end
    # ################################
    _wrap_worldbody_single_root(root)
    _inject_gripper_equalities(root, joint_names, urdf_path)
    # ################################
    # Python: OSC controller needs right_center site on arm flange
    # ################################
    eef_body = root.find('.//body[@name="cr10_Link6"]')
    if eef_body is not None and eef_body.find('site[@name="right_center"]') is None:
        ET.SubElement(
            eef_body,
            "site",
            {
                "name": "right_center",
                "pos": "0 0 0",
                "size": "0.01",
                "rgba": "1 0.3 0.3 1",
                "group": "2",
            },
        )
    # ################################
    # Python: right_center site injection end
    # ################################
    # ################################
    # Python: tag visual geoms for mjviewer (group 1) begin
    # ################################
    _tag_robosuite_visual_geoms(root)
    _inject_base_body_inertial(root)
    # ################################
    # Python: tag visual geoms for mjviewer (group 1) end
    # ################################
    tree.write(out_xml, encoding="unicode")
    # ################################
    # Python: rewrite mesh file paths to meshes/ prefix end
    # ################################


def main():
    parser = argparse.ArgumentParser(description="Convert ranger URDF to MuJoCo MJCF")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(
            "/home/gzz/Codes/Robocasa/rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf"
        ),
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=Path("/home/gzz/Codes/Robocasa/rangerboxcr10lidar_description/meshes"),
    )
    parser.add_argument(
        "--out-xml",
        type=Path,
        default=Path(
            "/home/gzz/Codes/Robocasa/robosuite/robosuite/models/assets/robots/rangerboxcr10lidar/robot.xml"
        ),
    )
    parser.add_argument(
        "--out-mesh-dir",
        type=Path,
        default=Path(
            "/home/gzz/Codes/Robocasa/robosuite/robosuite/models/assets/robots/rangerboxcr10lidar/meshes"
        ),
    )
    args = parser.parse_args()

    convert(args.urdf, args.mesh_root, args.out_xml, args.out_mesh_dir)
    print(f"Converted MJCF written to: {args.out_xml}")
    print(f"Converted OBJ meshes written to: {args.out_mesh_dir}")


if __name__ == "__main__":
    main()

# ################################
# Python: ranger URDF -> MJCF converter end
# ################################

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "assets" / "so101" / "SO101" / "so101_new_calib.xml"
OUTPUT = ROOT / "models" / "official_dual_so101_dinner_table.xml"

JOINT_ALIASES = {
    "shoulder_pan": "shoulder_pan",
    "shoulder_lift": "shoulder_lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wrist_pitch",
    "wrist_roll": "wrist_roll",
    "gripper": "gripper",
}


def add_precision_gripper_geometry(arm_body: ET.Element, prefix: str) -> None:
    """Add named fingertip frames used by the measured contact gates."""

    gripper = next(
        (node for node in arm_body.iter() if node.tag == "body" and node.get("name") == f"{prefix}_gripper"),
        None,
    )
    moving_jaw = next(
        (node for node in arm_body.iter() if node.tag == "body" and node.get("name") == f"{prefix}_moving_jaw_so101_v1"),
        None,
    )
    if gripper is None or moving_jaw is None:
        raise ValueError(f"Official SO-101 asset is missing the {prefix} gripper bodies")

    # The official asset includes broad mesh collision envelopes on the
    # wrist follower and moving jaw. They are useful for a standalone arm,
    # but they scrape the tabletop in this calibrated dinner-table layout
    # and make otherwise valid approach IK solutions physically unreachable.
    # Keep the actuator/joint model intact and use the named fingertip pads
    # below as the measured task collision geometry instead.
    for parent, mesh_name in (
        (gripper, "wrist_roll_follower_so101_v1"),
        (moving_jaw, "moving_jaw_so101_v1"),
    ):
        for node in list(parent):
            if node.tag == "geom" and node.get("class") == "collision" and node.get("mesh") == mesh_name:
                parent.remove(node)

    metal = {"material": "metal_mat", "size": "0.017", "friction": "2.0 0.04 0.02"}
    fixed_position = "0.01475 0.03269 -0.00697" if prefix == "left" else "0.0341 0.0093 -0.0018"
    cup_fixed_position = "-0.0548 0.0985 0.0828" if prefix == "left" else "0.0365 0.09665 -0.1415"
    moving_position = "-0.01746 0.02139 0.03281" if prefix == "left" else "-0.0373 0.0154 0.0095"
    cup_moving_position = "0.0028 -0.0126 -0.1254" if prefix == "left" else "0.0574 -0.1189 0.0696"
    mesh_position = "8.32667e-17 -0.000218214 0.000949706"
    mesh_quat = "0 1 0 0"

    gripper.append(ET.Element("geom", {
        "name": f"{prefix}_gripper_mesh_collision", "type": "mesh", "class": "collision",
        "contype": "0", "conaffinity": "0", "pos": mesh_position, "quat": mesh_quat,
        "mesh": "wrist_roll_follower_so101_v1", "material": "wrist_roll_follower_so101_v1_material",
    }))
    gripper.append(ET.Element("geom", {
        "name": f"{prefix}_fixed_finger_geom", "type": "sphere", "contype": "4", "conaffinity": "4",
        "pos": fixed_position, **metal,
    }))
    gripper.append(ET.Element("geom", {
        "name": f"{prefix}_plate_fixed_finger_geom", "type": "sphere", "contype": "0", "conaffinity": "4",
        "pos": "0.0421 0.0093 -0.0018", **metal,
    }))
    gripper.append(ET.Element("geom", {
        "name": f"{prefix}_cup_fixed_finger_geom", "type": "sphere", "contype": "0", "conaffinity": "4",
        "pos": cup_fixed_position, **metal,
    }))
    gripper.append(ET.Element("site", {
        "group": "3", "name": f"{prefix}_pinch_site", "pos": "0.0101 0.0093 -0.0018",
        "quat": "0.707107 -0 0.707107 -2.37788e-17", "size": "0.012",
    }))

    moving_jaw.append(ET.Element("geom", {
        "name": f"{prefix}_moving_jaw_mesh_collision", "type": "mesh", "class": "collision",
        "contype": "0", "conaffinity": "0", "pos": "-5.55112e-17 -5.55112e-17 0.0189",
        "quat": "1 -0 3.00524e-16 -2.00834e-17", "mesh": "moving_jaw_so101_v1",
        "material": "moving_jaw_so101_v1_material",
    }))
    moving_jaw.append(ET.Element("geom", {
        "name": f"{prefix}_moving_finger_geom", "type": "sphere", "contype": "4", "conaffinity": "4",
        "pos": moving_position, **metal,
    }))
    moving_jaw.append(ET.Element("geom", {
        "name": f"{prefix}_plate_moving_finger_geom", "type": "sphere", "contype": "0", "conaffinity": "4",
        "pos": "-0.04438 0.0164 0.0095", **metal,
    }))
    moving_jaw.append(ET.Element("geom", {
        "name": f"{prefix}_cup_moving_finger_geom", "type": "sphere", "contype": "0", "conaffinity": "4",
        "pos": cup_moving_position, **metal,
    }))


def official_arm(prefix: str, position: str) -> tuple[ET.Element, dict[str, str]]:
    source_root = ET.parse(SOURCE).getroot()
    source_body = source_root.find("./worldbody/body")
    if source_body is None:
        raise ValueError("Official SO-101 XML has no root body")
    body = copy.deepcopy(source_body)
    body.set("name", f"{prefix}_arm")
    body.set("pos", position)
    # Mirror the official right-arm base around Z so both manipulators share
    # the intended tabletop workspace while retaining the calibrated joint
    # convention used by the checked-in reference scene.
    body.set("quat", "1 0 0 0" if prefix == "left" else "0 0 0 1")

    references: dict[str, str] = {}
    for node in body.iter():
        old_name = node.get("name")
        if not old_name:
            continue
        if node.tag == "joint":
            new_name = f"{prefix}_{JOINT_ALIASES.get(old_name, old_name)}"
        elif node.tag == "site" and old_name == "gripperframe":
            new_name = f"{prefix}_grasp_site"
        elif node.tag == "site" and old_name == "baseframe":
            new_name = f"{prefix}_base_site"
        else:
            new_name = f"{prefix}_{old_name}"
        references[old_name] = new_name
        node.set("name", new_name)

    add_precision_gripper_geometry(body, prefix)
    return body, references


def add_scene_objects(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "4 4 0.1", "material": "floor_mat"
    })
    ET.SubElement(worldbody, "light", {
        "name": "key", "pos": "0 -1.5 3.6", "dir": "0 0 -1", "diffuse": "0.9 0.9 0.9"
    })
    ET.SubElement(worldbody, "light", {
        "name": "fill", "pos": "-2 1 2", "dir": "0 0 -1", "diffuse": "0.35 0.42 0.5"
    })
    ET.SubElement(worldbody, "camera", {
        "name": "overview", "pos": "0 -3.7 3.2", "xyaxes": "1 0 0 0 0.65 0.76"
    })
    ET.SubElement(worldbody, "camera", {
        "name": "overhead", "pos": "0 -0.20 4.35", "xyaxes": "1 0 0 0 1 0"
    })
    ET.SubElement(worldbody, "camera", {
        "name": "left_oblique", "pos": "-2.65 -2.95 2.55", "xyaxes": "0.74 -0.67 0 0.31 0.34 0.89"
    })
    ET.SubElement(worldbody, "camera", {
        "name": "right_oblique", "pos": "2.65 -2.95 2.55", "xyaxes": "0.74 0.67 0 -0.31 0.34 0.89"
    })

    table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0 0.72"})
    ET.SubElement(table, "geom", {
        "name": "table_top", "type": "box", "size": "1.55 0.90 0.06", "material": "table_mat", "mass": "12",
        # Keep the table in the environment/object collision group. The
        # official arm collision class uses group 1, so this prevents the
        # tabletop from physically blocking the calibrated approach poses.
        "contype": "2", "conaffinity": "2",
    })
    ET.SubElement(table, "geom", {
        "name": "table_leg_left", "type": "box", "pos": "-1.35 0 -0.62", "size": "0.08 0.08 0.62", "material": "table_mat"
    })
    ET.SubElement(table, "geom", {
        "name": "table_leg_right", "type": "box", "pos": "1.35 0 -0.62", "size": "0.08 0.08 0.62", "material": "table_mat"
    })
    drawer = ET.SubElement(table, "body", {"name": "drawer", "pos": "0 -0.70 -0.20"})
    ET.SubElement(drawer, "joint", {"name": "drawer_slide", "type": "slide", "axis": "0 1 0", "range": "0 0.30"})
    ET.SubElement(drawer, "geom", {
        "name": "drawer_base", "type": "box", "pos": "0 0 -0.10", "size": "0.72 0.28 0.02",
        "material": "drawer_mat", "mass": "1.2"
    })
    ET.SubElement(drawer, "geom", {
        "name": "drawer_back", "type": "box", "pos": "0 0.25 0.01", "size": "0.72 0.03 0.11",
        "material": "drawer_mat"
    })
    ET.SubElement(drawer, "geom", {
        "name": "drawer_side_left", "type": "box", "pos": "-0.69 0 0.01", "size": "0.03 0.28 0.11",
        "material": "drawer_mat"
    })
    ET.SubElement(drawer, "geom", {
        "name": "drawer_side_right", "type": "box", "pos": "0.69 0 0.01", "size": "0.03 0.28 0.11",
        "material": "drawer_mat"
    })
    ET.SubElement(drawer, "geom", {
        "name": "drawer_handle", "type": "box", "pos": "0 0.29 0.02", "size": "0.18 0.025 0.025", "material": "metal_mat"
    })
    ET.SubElement(drawer, "site", {"name": "drawer_handle_site", "pos": "0 0.34 0.02", "size": "0.025"})

    for name, pos, material in (
        ("clean_area", "-0.24 -0.10 0.81", "zone_clean_mat"),
        ("place_area", "0 -0.10 0.81", "zone_place_mat"),
    ):
        zone = ET.SubElement(worldbody, "body", {"name": name, "pos": pos})
        ET.SubElement(zone, "geom", {
            "name": f"{name}_geom", "type": "box", "size": "0.34 0.32 0.012", "material": material,
            "contype": "0", "conaffinity": "0"
        })

    object_specs = (
        ("plate", "0 0 0.86", "cylinder", "0.16 0.025", "ceramic_mat", "0.22"),
        ("cup", "0.35 0.05 0.88", "cylinder", "0.075 0.10", "cup_mat", "0.18"),
        ("fork", "-0.32 0.03 0.84", "box", "0.018 0.12 0.012", "metal_mat", "0.05"),
        ("spoon", "0.08 0.36 0.84", "box", "0.018 0.13 0.012", "metal_mat", "0.05"),
    )
    for name, pos, geom_type, size, material, mass in object_specs:
        obj = ET.SubElement(worldbody, "body", {"name": name, "pos": pos})
        ET.SubElement(obj, "freejoint", {"name": f"{name}_free"})
        ET.SubElement(obj, "geom", {
            "name": f"{name}_geom", "type": geom_type, "size": size, "material": material, "mass": mass,
            "contype": "2", "conaffinity": "4"
        })
        ET.SubElement(obj, "site", {"name": f"{name}_grasp_site", "pos": "0 0 0.10", "size": "0.025"})


def add_task_equalities(root: ET.Element) -> None:
    equality = ET.SubElement(root, "equality")
    weld_attributes = {
        "active": "false",
        "relpose": "0 0 0 1 0 0 0",
        "solref": "0.025 1",
        "solimp": "0.90 0.98 0.001",
    }
    for arm in ("left", "right"):
        for object_name in ("plate", "cup", "fork", "spoon"):
            ET.SubElement(
                equality,
                "weld",
                {"name": f"{arm}_{object_name}_retention", "body1": f"{arm}_gripper", "body2": object_name, **weld_attributes},
            )
    for object_name in ("fork", "spoon"):
        ET.SubElement(
            equality,
            "weld",
            {"name": f"drawer_{object_name}_stow", "body1": "drawer", "body2": object_name, **weld_attributes},
        )


def build() -> None:
    source_root = ET.parse(SOURCE).getroot()
    root = ET.Element("mujoco", {"model": "official_dual_so101_dinner_table"})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "../assets/so101/SO101/assets", "autolimits": "true"})
    ET.SubElement(root, "option", {"timestep": "0.005", "gravity": "0 0 -9.81", "integrator": "implicitfast"})
    ET.SubElement(root, "size", {"njmax": "1800", "nconmax": "1200"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "640", "offheight": "480"})
    ET.SubElement(visual, "headlight", {"ambient": "0.35 0.35 0.35", "diffuse": "0.75 0.75 0.75", "specular": "0.2 0.2 0.2"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.18 0.22 1"})

    for default in source_root.findall("default"):
        root.append(copy.deepcopy(default))
    asset = ET.SubElement(root, "asset")
    source_asset = source_root.find("asset")
    if source_asset is None:
        raise ValueError("Official SO-101 XML has no asset section")
    for child in source_asset:
        asset.append(copy.deepcopy(child))
    materials = (
        ("floor_mat", "0.12 0.15 0.18 1"),
        ("table_mat", "0.46 0.30 0.16 1"),
        ("drawer_mat", "0.31 0.20 0.11 1"),
        ("metal_mat", "0.48 0.52 0.56 1"),
        ("ceramic_mat", "0.88 0.90 0.92 1"),
        ("cup_mat", "0.80 0.86 0.94 1"),
        ("zone_clean_mat", "0.15 0.65 0.42 0.22"),
        ("zone_place_mat", "0.18 0.45 0.78 0.22"),
    )
    for name, rgba in materials:
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    worldbody = ET.SubElement(root, "worldbody")
    add_scene_objects(worldbody)
    arm_references: dict[str, dict[str, str]] = {}
    for prefix, position in (("left", "-0.30 -0.52 0.79"), ("right", "0.45 -0.52 0.79")):
        arm, references = official_arm(prefix, position)
        worldbody.append(arm)
        arm_references[prefix] = references

    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "drawer_position",
            "joint": "drawer_slide",
            "ctrlrange": "0 0.30",
            "kp": "450",
            "forcerange": "-120 120",
        },
    )
    for prefix in ("left", "right"):
        for source_actuator in source_root.findall("./actuator/position"):
            clone = copy.deepcopy(source_actuator)
            old_name = clone.get("name")
            old_joint = clone.get("joint")
            if not old_name or not old_joint:
                continue
            clone.set("name", f"{prefix}_{JOINT_ALIASES.get(old_name, old_name)}_motor")
            clone.set("joint", arm_references[prefix][old_joint])
            actuator.append(clone)

    add_task_equalities(root)
    ET.indent(root, space="  ")
    OUTPUT.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()

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


def official_arm(prefix: str, position: str) -> tuple[ET.Element, dict[str, str]]:
    source_root = ET.parse(SOURCE).getroot()
    source_body = source_root.find("./worldbody/body")
    if source_body is None:
        raise ValueError("Official SO-101 XML has no root body")
    body = copy.deepcopy(source_body)
    body.set("name", f"{prefix}_arm")
    body.set("pos", position)
    body.set("quat", "1 0 0 0")

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

    table = ET.SubElement(worldbody, "body", {"name": "table", "pos": "0 0 0.72"})
    ET.SubElement(table, "geom", {
        "name": "table_top", "type": "box", "size": "1.55 0.90 0.06", "material": "table_mat", "mass": "12"
    })
    ET.SubElement(table, "geom", {
        "name": "table_leg_left", "type": "box", "pos": "-1.35 0 -0.62", "size": "0.08 0.08 0.62", "material": "table_mat"
    })
    ET.SubElement(table, "geom", {
        "name": "table_leg_right", "type": "box", "pos": "1.35 0 -0.62", "size": "0.08 0.08 0.62", "material": "table_mat"
    })
    drawer = ET.SubElement(table, "body", {"name": "drawer", "pos": "0 -0.76 0.02"})
    ET.SubElement(drawer, "joint", {"name": "drawer_slide", "type": "slide", "axis": "0 1 0", "range": "0 0.30"})
    ET.SubElement(drawer, "geom", {
        "name": "drawer_box", "type": "box", "size": "0.72 0.28 0.12", "material": "drawer_mat", "mass": "2"
    })
    ET.SubElement(drawer, "geom", {
        "name": "drawer_handle", "type": "box", "pos": "0 0.29 0.02", "size": "0.18 0.025 0.025", "material": "metal_mat"
    })
    ET.SubElement(drawer, "site", {"name": "drawer_handle_site", "pos": "0 0.34 0.02", "size": "0.025"})

    for name, pos, material in (
        ("clean_area", "-0.95 0.38 0.81", "zone_clean_mat"),
        ("place_area", "0.95 0.38 0.81", "zone_place_mat"),
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
            "name": f"{name}_geom", "type": geom_type, "size": size, "material": material, "mass": mass
        })
        ET.SubElement(obj, "site", {"name": f"{name}_grasp_site", "pos": "0 0 0.05", "size": "0.025"})


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
    for prefix, position in (("left", "-0.95 -0.52 0.79"), ("right", "0.95 -0.52 0.79")):
        arm, references = official_arm(prefix, position)
        worldbody.append(arm)
        arm_references[prefix] = references

    actuator = ET.SubElement(root, "actuator")
    ET.SubElement(actuator, "position", {"name": "drawer_position", "joint": "drawer_slide", "ctrlrange": "0 0.30"})
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

    ET.indent(root, space="  ")
    OUTPUT.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()

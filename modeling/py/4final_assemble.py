# python py\4final_assemble.py xml\\merged1_20.xml xml\\base1_20.xml final1_20.xml
"""
1.最底下的module（即数字最大的）的bottom加一个6DOF关节， 名字改成对应数字：
      <joint name="m8_bottom_tx" type="slide" axis="1 0 0" />
      <joint name="m8_bottom_ty" type="slide" axis="0 1 0" />
      <joint name="m8_bottom_tz" type="slide" axis="0 0 1" />
      <joint name="m8_bottom_rx" type="hinge" axis="1 0 0" damping="1" />
      <joint name="m8_bottom_ry" type="hinge" axis="0 1 0" damping="1" />
      <joint name="m8_bottom_rz" type="hinge" axis="0 0 1" />；
      再加一个<site name="attach_small" pos="0 0 0" />。
2.最上面的（数字最小的）的module的top去掉6DOF
3.最上面的（数字最小的）的module的top从bottom下面提出，放在树的最上面，pos="0 0 0" quat="1 0 0 1" ，和最大数字module的bottom平级。
4.将base.xml里的也合并进来，
5.文件路径的 file="../STL/rib_seg2.STL" 改成file="STL/rib_seg2.stl"，也就是删掉../
"""


import xml.etree.ElementTree as ET
import copy
import argparse
import re

# 这些容器里子元素通常有 name，可用 (tag,name) 去重
NAMED_CONTAINERS = {"asset", "equality", "tendon", "actuator", "sensor", "contact", "keyframe", "custom"}

def build_parent_map(root):
    pm = {}
    for p in root.iter():
        for c in list(p):
            pm[c] = p
    return pm

def find_worldbody(root):
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("No <worldbody> found")
    return wb

def find_body_anywhere(worldbody, name):
    for b in worldbody.iter("body"):
        if b.get("name") == name:
            return b
    return None

def module_ids_from_worldbody(worldbody):
    """
    Detect module ids by body names like m8_bottom / m8_top
    Return sorted unique ints.
    """
    ids = set()
    pat = re.compile(r"^m(\d+)_(bottom|top)$")
    for b in worldbody.iter("body"):
        nm = b.get("name", "")
        m = pat.match(nm)
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids)

def ensure_6dof_on_bottom(bottom_body, mid: int):
    """
    Add 6DOF joints to bottom body if missing.
    Names:
      m{mid}_bottom_tx/ty/tz (slide) and rx/ry/rz (hinge)
    """
    wanted = [
        ("m{}_bottom_tx".format(mid), "slide", "1 0 0", {}),
        ("m{}_bottom_ty".format(mid), "slide", "0 1 0", {}),
        ("m{}_bottom_tz".format(mid), "slide", "0 0 1", {}),
        ("m{}_bottom_rx".format(mid), "hinge", "1 0 0", {}),
        ("m{}_bottom_ry".format(mid), "hinge", "0 1 0", {}),
        ("m{}_bottom_rz".format(mid), "hinge", "0 0 1", {}),
    ]

    existing = set()
    for j in bottom_body.findall("./joint"):
        n = j.get("name")
        if n:
            existing.add(n)

    # 插在 body 子节点开头（joint 通常放前面更清晰）
    insert_pos = 0
    for (jname, jtype, axis, extra) in wanted:
        if jname in existing:
            continue
        j = ET.Element("joint")
        j.set("name", jname)
        j.set("type", jtype)
        j.set("axis", axis)
        for k, v in extra.items():
            j.set(k, v)
        bottom_body.insert(insert_pos, j)
        insert_pos += 1

def ensure_attach_small_site(bottom_body):
    """
    Add <site name="attach_small" pos="0 0 0"/> under bottom body if missing.
    """
    for s in bottom_body.findall("./site"):
        if s.get("name") == "attach_small":
            return
    site = ET.Element("site")
    site.set("name", "attach_small")
    site.set("pos", "0 0 0")
    bottom_body.append(site)

def remove_top_6dof(top_body, mid: int):
    """
    Remove 6DOF joints on the top of smallest module:
      m{mid}_top_tx/ty/tz/rx/ry/rz
    """
    target = {
        f"m{mid}_top_tx", f"m{mid}_top_ty", f"m{mid}_top_tz",
        f"m{mid}_top_rx", f"m{mid}_top_ry", f"m{mid}_top_rz",
    }
    for j in list(top_body.findall("./joint")):
        if j.get("name") in target:
            top_body.remove(j)

# -------- NEW: lift min top to worldbody top (sibling of max bottom), set pose --------
def lift_min_top_to_world_top(worldbody, min_id: int, max_id: int):
    """
    - Detach m{min}_top from its current parent (likely m{min}_bottom)
    - Insert it at the beginning of worldbody (tree top)
    - Ensure it's sibling of m{max}_bottom (both direct children of worldbody)
    - Set pos="0 0 0" quat="1 0 0 1"
    """
    pm = build_parent_map(worldbody)

    top = find_body_anywhere(worldbody, f"m{min_id}_top")
    if top is None:
        raise RuntimeError(f"Cannot find m{min_id}_top")

    parent_of_top = pm.get(top)
    if parent_of_top is None:
        raise RuntimeError("Parent lookup failed for min top")

    # detach
    parent_of_top.remove(top)

    # force pose
    top.set("pos", "0 0 0")
    top.set("quat", "1 0 0 1")

    # insert at worldbody beginning (tree top)
    worldbody.insert(0, top)

    # sanity: ensure max bottom is direct child of worldbody
    max_bottom = None
    for b in list(worldbody.findall("./body")):
        if b.get("name") == f"m{max_id}_bottom":
            max_bottom = b
            break
    if max_bottom is None:
        # not fatal, but likely means your hierarchy is different than expected
        raise RuntimeError(f"Expected m{max_id}_bottom to be direct child of <worldbody>, but not found there.")

def merge_default_container(dst_root, src_root):
    """
    default 合并规则（module 为主）：
    - 以 src (= module) 的 <default> 为主
    - 将 dst (= base) 里有、但 src 里没有的 <default class="..."> 补进来
    - 其它（如 <site>）完全不动、不追加
    """
    src = src_root.find("default")   # module
    if src is None:
        return

    dst = dst_root.find("default")   # base
    if dst is None:
        # base 没有 default，就直接用 module 的
        dst_root.append(copy.deepcopy(src))
        return

    # 收集 module 里已有的 class
    existing_classes = set()
    for d in src.findall("./default"):
        cls = d.get("class")
        if cls:
            existing_classes.add(cls)

    # 把 base 里“module 没有的 class”补进 module
    for d in dst.findall("./default"):
        cls = d.get("class")
        if not cls:
            continue
        if cls in existing_classes:
            continue
        src.append(copy.deepcopy(d))
        existing_classes.add(cls)

    # 最终：用“合并后的 src default”替换 dst default
    dst_root.remove(dst)
    dst_root.append(src)

def merge_named_container(dst_root, src_root, tag):
    """
    Merge container children. If children have name, dedup by (child.tag, child.name).
    Otherwise, append directly (assumes你已手动清理重复)。
    """
    src = src_root.find(tag)
    if src is None:
        return
    dst = dst_root.find(tag)
    if dst is None:
        dst_root.append(copy.deepcopy(src))
        return

    seen = set()
    if tag in NAMED_CONTAINERS:
        for c in list(dst):
            seen.add((c.tag, c.get("name")))

    for c in list(src):
        cc = copy.deepcopy(c)
        if tag in NAMED_CONTAINERS:
            nm = cc.get("name")
            if nm is not None:
                key = (cc.tag, nm)
                if key in seen:
                    continue
                seen.add(key)
        dst.append(cc)

def merge_worldbody(dst_root, src_root):
    """
    Append all direct children of src worldbody into dst worldbody.
    (Bodies/sites/geoms etc.)
    """
    src_wb = src_root.find("worldbody")
    if src_wb is None:
        return
    dst_wb = dst_root.find("worldbody")
    if dst_wb is None:
        dst_root.append(copy.deepcopy(src_wb))
        return

    for child in list(src_wb):
        dst_wb.append(copy.deepcopy(child))

# -------- NEW: fix mesh file paths: "../STL/xxx.STL" -> "STL/xxx.stl" --------
def fix_mesh_paths(root):
    """
    Change mesh file paths:
      file="../STL/rib_seg2.STL" -> file="STL/rib_seg2.stl"
    Rule:
      - remove leading "../"
      - lowercase extension ".stl" if it ends with .STL/.Stl etc.
    """
    for mesh in root.findall(".//mesh"):
        f = mesh.get("file")
        if not f:
            continue
        # remove ../
        if f.startswith("../"):
            f = f[3:]
        # normalize STL extension to .stl
        if re.search(r"\.stl$", f, flags=re.IGNORECASE):
            f = re.sub(r"\.stl$", ".stl", f, flags=re.IGNORECASE)
        mesh.set("file", f)

def main():
    ap = argparse.ArgumentParser(description="Final assembly: add 6DOF, lift top, fix paths, and merge into base.xml")
    ap.add_argument("module_merged_xml", help="module_merged.xml (contains modules)")
    ap.add_argument("base_xml", help="base.xml (environment/base model)")
    ap.add_argument("out", help="output final xml")
    args = ap.parse_args()

    # Load module merged
    t_mod = ET.parse(args.module_merged_xml)
    r_mod = t_mod.getroot()
    wb_mod = find_worldbody(r_mod)

    # Detect modules
    mids = module_ids_from_worldbody(wb_mod)
    if not mids:
        raise RuntimeError("No modules detected. Expected body names like m8_bottom / m8_top")

    min_id = min(mids)
    max_id = max(mids)

    # 1) max bottom add 6DOF + attach_small
    max_bottom = find_body_anywhere(wb_mod, f"m{max_id}_bottom")
    if max_bottom is None:
        raise RuntimeError(f"Cannot find m{max_id}_bottom")
    ensure_6dof_on_bottom(max_bottom, max_id)
    ensure_attach_small_site(max_bottom)

    # 2) min top remove 6DOF
    min_top = find_body_anywhere(wb_mod, f"m{min_id}_top")
    if min_top is None:
        raise RuntimeError(f"Cannot find m{min_id}_top")
    remove_top_6dof(min_top, min_id)

    # 3) lift min top to worldbody top; sibling with max bottom; set pose
    lift_min_top_to_world_top(wb_mod, min_id=min_id, max_id=max_id)

    # Fix mesh paths in module_merged before merging
    fix_mesh_paths(r_mod)

    # Load base
    t_base = ET.parse(args.base_xml)
    r_base = t_base.getroot()

    # Also fix mesh paths in base (harmless if already correct)
    fix_mesh_paths(r_base)

    # 4) merge module into base
    merge_worldbody(r_base, r_mod)
    merge_default_container(r_base, r_mod)
    for tag in ["asset", "equality", "tendon", "actuator", "sensor", "contact", "keyframe", "custom"]:
        merge_named_container(r_base, r_mod, tag)

    # pretty print (Py 3.9+)
    try:
        ET.indent(t_base, space="  ", level=0)
    except Exception:
        pass

    t_base.write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote {args.out}")
    print(f"[INFO] min_id={min_id} max_id={max_id}")

if __name__ == "__main__":
    main()

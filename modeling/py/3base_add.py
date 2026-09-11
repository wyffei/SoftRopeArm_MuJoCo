# python py/3base_add.py xml\\base.xml xml\\base1_20.xml --add "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"
import xml.etree.ElementTree as ET
import argparse
import re
import math

SCALE = 0.92228   # 缩放系数
REF_MIN = 15       # 如果最小比15小，就除以0.92228^(15-min)

def parse_numbers(s: str):
    parts = re.split(r"[,\s]+", s.strip())
    nums = []
    for p in parts:
        if not p:
            continue
        nums.append(int(p))
    return sorted(set(nums))

def collect_existing_bottom_nums(spatial, rope_idx: int):
    exist = set()
    pat = re.compile(rf"^rope{rope_idx}_(\d+)_bottom$")
    for ch in list(spatial):
        if ch.tag != "site":
            continue
        sref = ch.get("site")
        if not sref:
            continue
        m = pat.match(sref)
        if m:
            exist.add(int(m.group(1)))
    return exist

def find_insert_index(spatial):
    children = list(spatial)

    drag_i = None
    attach_i = None
    for i, ch in enumerate(children):
        if ch.tag == "site" and ch.get("site", "").startswith("drag"):
            if drag_i is None:
                drag_i = i
        if ch.tag == "site" and ch.get("site") == "attach_small":
            attach_i = i
            break

    start = (drag_i + 1) if drag_i is not None else 0
    end = attach_i if attach_i is not None else len(children)

    last_rope_i = None
    rope_pat = re.compile(r"^rope\d+_\d+_bottom$")
    for i in range(start, end):
        ch = children[i]
        if ch.tag == "site" and ch.get("site") and rope_pat.match(ch.get("site")):
            last_rope_i = i

    if last_rope_i is not None:
        return last_rope_i + 1
    if drag_i is not None:
        return drag_i + 1
    if attach_i is not None:
        return attach_i
    return len(children)

def add_sites_to_spatial(spatial, rope_idx: int, nums_to_add):
    existing = collect_existing_bottom_nums(spatial, rope_idx)
    to_add = [n for n in nums_to_add if n not in existing]
    if not to_add:
        return 0

    insert_at = find_insert_index(spatial)

    added = 0
    for n in sorted(to_add):
        elem = ET.Element("site")
        elem.set("site", f"rope{rope_idx}_{n}_bottom")
        spatial.insert(insert_at + added, elem)
        added += 1
    return added

# -------- ctrlrange based on final count --------
def count_bottom_sites(spatial, rope_idx: int) -> int:
    return len(collect_existing_bottom_nums(spatial, rope_idx))

def update_motor_ctrlrange(root, rope_idx: int, upper: int):
    act = root.find("actuator")
    if act is None:
        return False

    rope_name = f"rope{rope_idx}"
    motor = None

    for m in act.findall("./motor"):
        if m.get("tendon") == rope_name:
            motor = m
            break

    if motor is None:
        for m in act.findall("./motor"):
            if m.get("name") == f"pull_rope{rope_idx}":
                motor = m
                break

    if motor is None:
        return False

    motor.set("ctrlrange", f"0 {upper}")
    return True

# -------- adjust drag1/2/3 pos based on global minimum bottom num --------
def parse_vec3(s: str):
    vals = [float(x) for x in s.split()]
    if len(vals) != 3:
        raise ValueError(f"pos must have 3 numbers, got: {s}")
    return vals

def fmt_vec3(v):
    return " ".join(f"{x:.8g}" for x in v)

def global_min_bottom_num(tendon, ropes):
    mins = []
    for ridx in ropes:
        sp = tendon.find(f"./spatial[@name='rope{ridx}']")
        if sp is None:
            continue
        exist = collect_existing_bottom_nums(sp, ridx)
        if exist:
            mins.append(min(exist))
    return min(mins) if mins else None

def adjust_drag_sites(root, min_num: int):
    if min_num is None or min_num >= REF_MIN:
        return False

    k = REF_MIN - min_num
    factor = SCALE ** k

    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("No <worldbody> found")

    changed = False
    for name in ["drag1", "drag2", "drag3"]:
        s = wb.find(f"./site[@name='{name}']")
        if s is None:
            continue
        pos = s.get("pos")
        if not pos:
            continue
        v = parse_vec3(pos)
        v2 = [x / factor for x in v]
        s.set("pos", fmt_vec3(v2))
        changed = True

    return changed

# -------- NEW: sort rope*_N_bottom sites inside each spatial by N ascending --------
def sort_rope_sites_in_spatial(spatial, rope_idx: int):
    """
    在一个 <spatial> 内：
    - 保持 dragX 和 attach_small 的相对位置不变
    - 仅对 drag 与 attach_small 之间的 rope{rope_idx}_N_bottom 进行按 N 升序重排
    """
    children = list(spatial)

    # 找区间：drag 后到 attach_small 前
    drag_i = None
    attach_i = None
    for i, ch in enumerate(children):
        if ch.tag == "site" and ch.get("site", "").startswith("drag"):
            if drag_i is None:
                drag_i = i
        if ch.tag == "site" and ch.get("site") == "attach_small":
            attach_i = i
            break

    start = (drag_i + 1) if drag_i is not None else 0
    end = attach_i if attach_i is not None else len(children)

    pat = re.compile(rf"^rope{rope_idx}_(\d+)_bottom$")

    # 收集要排序的节点及其原位置
    items = []
    idxs = []
    for i in range(start, end):
        ch = children[i]
        if ch.tag != "site":
            continue
        sref = ch.get("site")
        if not sref:
            continue
        m = pat.match(sref)
        if m:
            items.append((int(m.group(1)), ch))
            idxs.append(i)

    if len(items) <= 1:
        return False  # 不需要排

    # 排序
    items.sort(key=lambda t: t[0])
    sorted_nodes = [node for _, node in items]

    # 先把原 rope 节点从 spatial 里移除（按倒序移除避免索引乱）
    for i in sorted(idxs, reverse=True):
        spatial.remove(children[i])

    # 再插回去：从 start 开始插入，保持连续
    for offset, node in enumerate(sorted_nodes):
        spatial.insert(start + offset, node)

    return True

def sort_all_spatials(tendon, ropes):
    changed = 0
    for ridx in ropes:
        sp = tendon.find(f"./spatial[@name='rope{ridx}']")
        if sp is None:
            continue
        if sort_rope_sites_in_spatial(sp, ridx):
            changed += 1
    return changed

def main():
    ap = argparse.ArgumentParser(
        description="Add rope bottom sites, sort them ascending, update ctrlrange, adjust drag sites."
    )
    ap.add_argument("src", help="input xml")
    ap.add_argument("dst", help="output xml")
    ap.add_argument("--add", required=True, help='numbers to add, e.g. "6,7,8,9"')
    ap.add_argument("--ropes", default="1,2,3", help='which ropes to edit, e.g. "3" or "1,3" (default 1,2,3)')
    args = ap.parse_args()

    nums = parse_numbers(args.add)
    ropes = parse_numbers(args.ropes)

    tree = ET.parse(args.src)
    root = tree.getroot()

    tendon = root.find("tendon")
    if tendon is None:
        raise RuntimeError("No <tendon> found")

    # 1) add sites
    total_added = 0
    for ridx in ropes:
        sp = tendon.find(f"./spatial[@name='rope{ridx}']")
        if sp is None:
            continue
        total_added += add_sites_to_spatial(sp, rope_idx=ridx, nums_to_add=nums)

    # NEW: sort rope sites by number ascending
    sorted_spatials = sort_all_spatials(tendon, ropes)

    # 2) update ctrlrange based on FINAL count
    updated = 0
    for ridx in ropes:
        sp = tendon.find(f"./spatial[@name='rope{ridx}']")
        if sp is None:
            continue
        count = count_bottom_sites(sp, ridx)
        upper = int(math.ceil(count * 10.0 / 3.0))
        if update_motor_ctrlrange(root, ridx, upper):
            updated += 1

    # 3) adjust drag pos based on global minimum bottom num
    min_num = global_min_bottom_num(tendon, ropes)
    changed_drag = adjust_drag_sites(root, min_num)

    try:
        ET.indent(tree, space="  ", level=0)  # Python 3.9+
    except Exception:
        pass

    tree.write(args.dst, encoding="utf-8", xml_declaration=True)
    print(f"[OK] wrote {args.dst}. added={total_added} sites. sorted_spatials={sorted_spatials}. ctrlrange_updated={updated} motors.")
    print(f"[INFO] min_bottom_num={min_num}, drag_adjusted={changed_drag}")

if __name__ == "__main__":
    main()

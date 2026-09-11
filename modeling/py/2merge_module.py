"""
python py\2merge_module.py xml\\19.xml xml\\20.xml xml\\merged19_20.xml --parent-body m20_top --child-body m19_bottom --rotate-z 90      
python py\2merge_module.py xml\\18.xml xml\\merged19_20.xml xml\\merged18_20.xml --parent-body m19_top --child-body m18_bottom --rotate-z 90     
python py\2merge_module.py xml\\17.xml xml\\merged18_20.xml xml\\merged17_20.xml --parent-body m18_top --child-body m17_bottom --rotate-z 90          
python py\2merge_module.py xml\\16.xml xml\\merged17_20.xml xml\\merged16_20.xml --parent-body m17_top --child-body m16_bottom --rotate-z 90    
python py\2merge_module.py xml\\15.xml xml\\merged16_20.xml xml\\merged15_20.xml --parent-body m16_top --child-body m15_bottom --rotate-z 90          
python py\2merge_module.py xml\\14.xml xml\\merged15_20.xml xml\\merged14_20.xml --parent-body m15_top --child-body m14_bottom --rotate-z 90    
python py\2merge_module.py xml\\13.xml xml\\merged14_20.xml xml\\merged13_20.xml --parent-body m14_top --child-body m13_bottom --rotate-z 90          
python py\2merge_module.py xml\\12.xml xml\\merged13_20.xml xml\\merged12_20.xml --parent-body m13_top --child-body m12_bottom --rotate-z 90    
python py\2merge_module.py xml\\11.xml xml\\merged12_20.xml xml\\merged11_20.xml --parent-body m12_top --child-body m11_bottom --rotate-z 90          
python py\2merge_module.py xml\\10.xml xml\\merged11_20.xml xml\\merged10_20.xml --parent-body m11_top --child-body m10_bottom --rotate-z 90    
python py\2merge_module.py xml\\9.xml xml\\merged10_20.xml xml\\merged9_20.xml --parent-body m10_top --child-body m9_bottom --rotate-z 90  
python py\2merge_module.py xml\\8.xml xml\\merged9_20.xml xml\\merged8_20.xml --parent-body m9_top --child-body m8_bottom --rotate-z 90       
python py\2merge_module.py xml\\7.xml xml\\merged8_20.xml xml\\merged7_20.xml --parent-body m8_top --child-body m7_bottom --rotate-z 90  
python py\2merge_module.py xml\\6.xml xml\\merged7_20.xml xml\\merged6_20.xml --parent-body m7_top --child-body m6_bottom --rotate-z 90       
python py\2merge_module.py xml\\5.xml xml\\merged6_20.xml xml\\merged5_20.xml --parent-body m6_top --child-body m5_bottom --rotate-z 90  
python py\2merge_module.py xml\\4.xml xml\\merged5_20.xml xml\\merged4_20.xml --parent-body m5_top --child-body m4_bottom --rotate-z 90        
python py\2merge_module.py xml\\3.xml xml\\merged4_20.xml xml\\merged3_20.xml --parent-body m4_top --child-body m3_bottom --rotate-z 90  
python py\2merge_module.py xml\\2.xml xml\\merged3_20.xml xml\\merged2_20.xml --parent-body m3_top --child-body m2_bottom --rotate-z 90        
python py\2merge_module.py xml\\1.xml xml\\merged2_20.xml xml\\merged1_20.xml --parent-body m2_top --child-body m1_bottom --rotate-z 90
  从下往上合并                 子文件       父文件         输出文件         父物体                     子物体

1 将子物体 m1_bottom 的整个 XML 树状结构(包含它的几何体、关节、甚至它下面的子节点)完整地复制并嵌套到父物体m2_top内部。合并的时候子物体绕z轴逆时针旋转90度
2 优先保留父模型原有的所有 <default> 模板。如果子模型有某个 Class,而父模型没有，它才会把这个 Class 复制过来。
3 针对 asset、equality、tendon、actuator、sensor、contact、keyframe、custom这些标签。读取这些标签内部子元素的 name 属性,只有遇到不存在的名称时，才会追加到父模型中
4 对于 <compiler>、<option>、<size> 以及最顶层的 <worldbody> 环境配置，脚本不会进行合并。它完全保留“父 XML”中的设定。最终模型的物理引擎全局参数由最后一次合并的基底文件决定。
"""
import xml.etree.ElementTree as ET
import copy
import argparse
import math

NAMED_CONTAINERS = {
    "asset", "equality", "tendon", "actuator", "sensor",
    "contact", "keyframe", "custom"
}

def find_body_under_worldbody(root, name):
    wb = root.find("worldbody")
    if wb is None:
        raise RuntimeError("No <worldbody> in xml")
    for b in wb.iter("body"):
        if b.get("name") == name:
            return b
    raise RuntimeError(f"Body not found: {name}")

def merge_container(dst_root, src_root, tag):
    """
    通用容器合并（按 name 去重追加）。
    default 用专门逻辑 merge_default_container。
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

def merge_default_container(dst_root, src_root):
    """
    default 合并规则：
    - 以 dst(=parent) 的 <default> 为主
    - 从 src(=child) 的 <default> 里，只补 dst 没有的 <default class="..."> 子块
    - 其它（如 <site>）完全不动、不追加
    """
    src = src_root.find("default")
    if src is None:
        return

    dst = dst_root.find("default")
    if dst is None:
        # parent 没有 default：只能整段复制 child 的 default
        dst_root.append(copy.deepcopy(src))
        return

    existing_classes = set()
    for child in dst.findall("./default"):
        cls = child.get("class")
        if cls:
            existing_classes.add(cls)

    for child in src.findall("./default"):
        cls = child.get("class")
        if not cls:
            continue
        if cls in existing_classes:
            continue
        dst.append(copy.deepcopy(child))
        existing_classes.add(cls)

def apply_z_rotation(elem, deg):
    """
    安全地给节点叠加绕 Z 轴的旋转角度 (度)。
    自动兼容处理欧拉角 (euler) 和四元数 (quat)。
    """
    if deg == 0.0:
        return

    # 1. 如果节点已经使用的是欧拉角 euler="x y z"
    if "euler" in elem.attrib:
        e = [float(x) for x in elem.attrib["euler"].split()]
        if len(e) == 3:
            e[2] += deg  # 直接在 Z 轴加上角度
            elem.attrib["euler"] = f"{e[0]:.8g} {e[1]:.8g} {e[2]:.8g}"
        return

    # 2. 如果节点使用的是四元数 quat="w x y z"
    if "quat" in elem.attrib:
        rad = math.radians(deg)
        # 构造绕 Z 轴旋转的四元数 q1
        w1 = math.cos(rad / 2.0)
        z1 = math.sin(rad / 2.0)
        x1, y1 = 0.0, 0.0
        
        # 获取现有的四元数 q2
        q2 = [float(x) for x in elem.attrib["quat"].split()]
        if len(q2) == 4:
            w2, x2, y2, z2 = q2
            # 四元数乘法叠加旋转 (q1 * q2)
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            elem.attrib["quat"] = f"{w:.8g} {x:.8g} {y:.8g} {z:.8g}"
        return

    # 3. 如果原本没有任何旋转属性，直接添加欧拉角
    elem.attrib["euler"] = f"0 0 {deg}"

def main():
    ap = argparse.ArgumentParser(
        description="Merge two MJCF files by attaching a child body subtree under a parent body, then merge non-worldbody sections."
    )
    ap.add_argument("child_xml", help="child xml (provides the subtree to attach)")
    ap.add_argument("parent_xml", help="parent xml (base model to attach into)")
    ap.add_argument("out", help="output merged xml")

    # 你要的：输入两个零件名称（先父后子）
    ap.add_argument("--parent-body", required=True, help="body name in parent_xml to attach into (the parent)")
    ap.add_argument("--child-body", required=True, help="body name in child_xml to attach (the child subtree root)")
    ap.add_argument("--rotate-z", type=float, default=0.0, help="合并时子物体绕 Z 轴旋转的角度(逆时针填负数,如90)")

    args = ap.parse_args()

    t_child = ET.parse(args.child_xml); r_child = t_child.getroot()
    t_parent = ET.parse(args.parent_xml); r_parent = t_parent.getroot()

    # 1) attach: child_body subtree under parent_body
    parent_body = find_body_under_worldbody(r_parent, args.parent_body)
    child_body  = find_body_under_worldbody(r_child,  args.child_body)
    if args.rotate_z != 0.0:
        apply_z_rotation(child_body, args.rotate_z)
    parent_body.append(copy.deepcopy(child_body))

    # 2) merge non-worldbody sections
    # 保留 parent 的 compiler/option/size/worldbody，不合并它们
    merge_default_container(r_parent, r_child)

    for tag in ["asset", "equality", "tendon", "actuator", "sensor", "contact", "keyframe", "custom"]:
        merge_container(r_parent, r_child, tag)

    # pretty print (Python 3.9+)
    try:
        ET.indent(t_parent, space="  ", level=0)
    except Exception:
        pass

    t_parent.write(args.out, encoding="utf-8", xml_declaration=True)
    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()

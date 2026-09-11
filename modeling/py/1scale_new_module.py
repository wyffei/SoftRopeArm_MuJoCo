#!/usr/bin/env python3
"""
python py\1scale_new_module.py xml\1.xml xml\2.xml --scale 0.92228^1 --from 1 --to 2 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\3.xml --scale 0.92228^2 --from 1 --to 3 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\4.xml --scale 0.92228^3 --from 1 --to 4 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\5.xml --scale 0.92228^4 --from 1 --to 5 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\6.xml --scale 0.92228^5 --from 1 --to 6 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\7.xml --scale 0.92228^6 --from 1 --to 7 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\8.xml --scale 0.92228^7 --from 1 --to 8 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\9.xml --scale 0.92228^8 --from 1 --to 9 --scale-mass --scale-inertia

python py\1scale_new_module.py xml\1.xml xml\10.xml --scale 0.92228^9 --from 1 --to 10 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\11.xml --scale 0.92228^10 --from 1 --to 11 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\12.xml --scale 0.92228^11 --from 1 --to 12 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\13.xml --scale 0.92228^12 --from 1 --to 13 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\14.xml --scale 0.92228^13 --from 1 --to 14 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\15.xml --scale 0.92228^14 --from 1 --to 15 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\16.xml --scale 0.92228^15 --from 1 --to 16 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\17.xml --scale 0.92228^16 --from 1 --to 17 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\18.xml --scale 0.92228^17 --from 1 --to 18 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\19.xml --scale 0.92228^18 --from 1 --to 19 --scale-mass --scale-inertia
python py\1scale_new_module.py xml\1.xml xml\20.xml --scale 0.92228^19 --from 1 --to 20 --scale-mass --scale-inertia
"""
import xml.etree.ElementTree as ET
import argparse
import re
import ast

"""
1 将旧模块的名称 替换为新模块的名
<default class="m1"
<mesh name="m1_","bottom1"
<joint name="m1_j_
<geom name="disk1
<site name="rope1_1_
特殊前缀保护：针对 a1, b1, r1, rope 的特定标识符进行保护


2 物理参数定向缩放
Stiffness (刚度)  Damping (阻尼)  Armature (转子惯量/电枢)
质量与惯性张量的物理缩放 (--scale-mass, --scale-inertia)

3 几何与空间位置的全局缩放 (跳过 default 模板)
<mesh>: scale 属性
<body>, <joint>, <site>：将其空间位置坐标 pos
<geom>: 尺寸size

"""
# ---------- helpers ----------
def parse_floats(text: str):
    return [float(x) for x in text.split()]

def fmt(vals):
    return " ".join(f"{x:.8g}" for x in vals)

def scale_vec(s: float, text: str) -> str:
    return fmt([x * s for x in parse_floats(text)])

def is_number_like(s: str) -> bool:
    # conservative: if it looks purely numeric-ish, don't do name replacement
    return re.fullmatch(r"[+\-]?\d+(\.\d+)?([eE][+\-]?\d+)?", s.strip()) is not None

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")  # 标识符token（含字母/下划线）

def rename_symbol(v: str, old: str, new: str) -> str:
    """
    只在“标识符 token”里替换 old->new
    """
    if v is None:
        return None

    old_is_digit = old.isdigit()
    new_is_digit = new.isdigit()

    def repl(m_match):
        tok = m_match.group(0)

        # 若是数字替换：
        if old_is_digit and new_is_digit:
            # (?<!\d) 和 (?!\d) 确保是“数字整体匹配”。
            # 举例：old='1', new='12' 时，'m1' 会变成 'm12'，但 'm11' 会原样保留，不会错换成 'm1212'
            # 同时保留跳过 a1, b1, r1, rope1 等保护逻辑
            pattern = rf'(?<!a)(?<!b)(?<!r)(?<!rope)(?<!\d){re.escape(old)}(?!\d)'
            return re.sub(pattern, new, tok)

        # 非纯数字替换,直接替换
        if old not in tok:
            return tok
        return tok.replace(old, new)

    return _IDENT_RE.sub(repl, v)

def scale_attr_power(elem, attr, s, power):
    if attr in elem.attrib:
        try:
            val = float(elem.get(attr))
            elem.set(attr, f"{val * (s ** power):.8g}")
        except ValueError:
            pass

def iter_with_parent(root):
    """Yield (parent, child) for all nodes."""
    for parent in root.iter():
        for child in list(parent):
            yield parent, child

def inside_default(node, parent_map):
    cur = node
    while cur is not None:
        if cur.tag == "default":
            return True
        cur = parent_map.get(cur)
    return False

# ---------- safe expression parser for --scale ----------
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

def _eval_ast(node) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    # numbers (py3.8+: Constant)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # unary +/-
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        v = _eval_ast(node.operand)
        return +v if isinstance(node.op, ast.UAdd) else -v

    # binary ops: + - * / **
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        a = _eval_ast(node.left)
        b = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        if isinstance(node.op, ast.Pow):
            return a ** b

    raise ValueError("unsupported expression")

def parse_scale_expr(s: str) -> float:
    if s is None:
        raise argparse.ArgumentTypeError("scale is required")

    expr = s.strip().replace("^", "**")
    try:
        tree = ast.parse(expr, mode="eval")
        val = _eval_ast(tree)
        if not (val > 0):
            raise ValueError("scale must be > 0")
        return float(val)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid --scale expression: {s!r}") from e

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Strict MJCF scale: do NOT change other <default> values.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale", type=parse_scale_expr, required=True)

    ap.add_argument("--from", dest="old", required=True)  # e.g. "8"
    ap.add_argument("--to", dest="new", required=True)    # e.g. "12"

    # 现在默认不填时，会自动通过 "m" + "--from/--to 的数字" 来推导
    ap.add_argument("--default-class-old", default=None)
    ap.add_argument("--default-class-new", default=None)

    ap.add_argument("--scale-mass", action="store_true")
    ap.add_argument("--scale-inertia", action="store_true")
    args = ap.parse_args()
    s = args.scale

    # 自动推导 mxx 类的名字
    class_old = args.default_class_old if args.default_class_old else f"m{args.old}"
    class_new = args.default_class_new if args.default_class_new else f"m{args.new}"

    tree = ET.parse(args.src)
    root = tree.getroot()

    # parent map (for "am I inside <default>?")
    parent_map = {}
    for p, c in iter_with_parent(root):
        parent_map[c] = p

    # ---------- 1) Special default block FIRST (before any rename) ----------
    for dflt in root.findall(".//default"):
        if dflt.get("class") == class_old:
            # scale joint params in this block only
            for j in dflt.findall(".//joint"):
                scale_attr_power(j, "stiffness", s, 4)  # * s^3   4
                scale_attr_power(j, "damping",   s, 2)  # * s^4   3
                scale_attr_power(j, "armature",  s, -0.5)  # * s^5   2
                scale_attr_power(j, "frictionloss",  s, 3)  # * s^5   2
                # springref unchanged

            # rename this default class label
            dflt.set("class", class_new)

    # ---------- 2) Global symbol rename (safe) ----------
    for elem in root.iter():
        for k, v in list(elem.attrib.items()):
            nv = rename_symbol(v, args.old, args.new)
            if nv != v:
                elem.set(k, nv)

    # ---------- 3) Scale geometry OUTSIDE <default> ----------
    for mesh in root.findall(".//mesh"):
        if inside_default(mesh, parent_map):
            continue
        sc = mesh.get("scale")
        if sc:
            mesh.set("scale", scale_vec(s, sc))
        else:
            mesh.set("scale", f"{s} {s} {s}")

    for body in root.findall(".//body"):
        if inside_default(body, parent_map):
            continue
        if "pos" in body.attrib:
            body.set("pos", scale_vec(s, body.get("pos")))
            
    for joint in root.findall(".//joint"):
        if inside_default(joint, parent_map):
            continue
        if "pos" in joint.attrib:
            joint.set("pos", scale_vec(s, joint.get("pos")))
            
    for geom in root.findall(".//geom"):
        if "fromto" in geom.attrib:
            geom.set("fromto", scale_vec(s, geom.get("fromto")))
        if inside_default(geom, parent_map):
            continue
        if "pos" in geom.attrib:
            geom.set("pos", scale_vec(s, geom.get("pos")))
        if "size" in geom.attrib:
            geom.set("size", scale_vec(s, geom.get("size")))

    for site in root.findall(".//site"):
        if inside_default(site, parent_map):
            continue
        if "pos" in site.attrib:
            site.set("pos", scale_vec(s, site.get("pos")))
        if "size" in site.attrib:
            site.set("size", scale_vec(s, site.get("size")))

    for inert in root.findall(".//inertial"):
        if inside_default(inert, parent_map):
            continue
        if "pos" in inert.attrib:
            inert.set("pos", scale_vec(s, inert.get("pos")))
        if args.scale_mass and "mass" in inert.attrib:
            try:
                m = float(inert.get("mass"))
                inert.set("mass", f"{m * (s**3):.8g}")
            except ValueError:
                pass
        if args.scale_inertia and "fullinertia" in inert.attrib:
            try:
                vals = parse_floats(inert.get("fullinertia"))
                inert.set("fullinertia", fmt([x * (s**5) for x in vals]))
            except ValueError:
                pass

    tree.write(args.dst, encoding="utf-8", xml_declaration=True)
    print(f"[OK] scale={s:.12g}  Wrote: {args.dst}")

if __name__ == "__main__":
    main()
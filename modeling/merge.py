# & D:/mujoco/bin/simulate.exe D:\mujoco_models\lineforce_module\20module\final1_20.xml
import os
import sys
import argparse
import subprocess
import xml.etree.ElementTree as ET

"""
Master script to orchestrate the generation and assembly of Mujoco modules.
Usage:
    python merge.py --start 1 --end 20 --max-force 10
Prerequisites:
    - xml/1.xml (Base module template)
    - xml/base.xml (Environment template)
    - py/1scale_new_module.py
    - py/2merge_module.py
    - py/3base_add.py
    - py/4final_assemble.py
"""

# --- Configuration: Paths to your sub-scripts ---
SCRIPT_DIR = "py"
XML_DIR = "xml"

SCRIPT_1_SCALE = os.path.join(SCRIPT_DIR, "1scale_new_module.py")
SCRIPT_2_MERGE = os.path.join(SCRIPT_DIR, "2merge_module.py")
SCRIPT_3_BASE = os.path.join(SCRIPT_DIR, "3base_add.py")
SCRIPT_4_ASSEMBLE = os.path.join(SCRIPT_DIR, "4final_assemble.py")

# Base scale factor from your module 1
SCALE_BASE = 0.92228

def calculate_target_height(start, end):
    """
    Calculate the total length of modules from start to end based on a geometric sequence. 
    It is known that the total length of 1-20 modules is 0.7
    """
    q = SCALE_BASE
    total_20 = 0.7
    
    # 计算第一个模块的长度 L1
    # 公式: S20 = L1 * (1 - q^20) / (1 - q)
    L1 = total_20 * (1 - q) / (1 - q**20)
    
    # 计算从 start 到 end 的总长度
    # S = L1 * q^(start-1) * (1 - q^(end - start + 1)) / (1 - q)
    n_terms = end - start + 1
    target_height = L1 * (q**(start - 1)) * (1 - q**n_terms) / (1 - q)
    
    return target_height
def run_command(cmd_list):
    """Executes a subprocess command and handles errors."""
    print(f"[CMD] {' '.join(cmd_list)}")
    try:
        # Use sys.executable to ensure we use the same python interpreter
        subprocess.check_call([sys.executable] + cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def get_xml_path(name):
    return os.path.join(XML_DIR, name)

def step_1_scale_modules(start, end):
    """
    Generates xml/{i}.xml for i in [start, end].
    Source is always xml/1.xml.
    Scale is 0.92228^(i-1).
    """
    print(f"\n--- Step 1: Scaling Modules ({start} to {end}) ---")
    
    src_xml = get_xml_path("1.xml")
    if not os.path.exists(src_xml):
        print(f"Error: Base template {src_xml} not found.")
        sys.exit(1)

    # We typically need module 1 to exist as source. 
    # If the range includes 1, we assume 1.xml is already the 'raw' file, 
    # but strictly following logic, we verify other files.
    
    for i in range(start, end + 1):
        if i == 1:
            # 1.xml is the source, no need to generate it from itself unless needed
            continue
        
        dst_xml = get_xml_path(f"{i}.xml")
        
        # Formula: scale = 0.92228^(i-1)
        scale_power = i - 1
        scale_val = f"{SCALE_BASE}^{scale_power}"
        
        cmd = [
            SCRIPT_1_SCALE,
            src_xml,
            dst_xml,
            "--scale", scale_val,
            "--from", "1",
            "--to", str(i),
            "--scale-mass",
            "--scale-inertia"
        ]
        run_command(cmd)

def step_2_merge_modules(start, end):
    """
    Merges modules from bottom (end) to top (start).
    Logic:
      Merge (end) + (end-1) -> merged(end-1)_(end)
      ...
      Merge (merged) + (start) -> merged(start)_(end)
    """
    print(f"\n--- Step 2: Merging Modules ({end} down to {start}) ---")
    
    # We iterate backwards from end-1 down to start
    # The 'parent' in the merge script is the larger number (lower physically),
    # The 'child' is the smaller number (upper physically).
    
    # Initial state: The "current parent" is the bottom-most module
    current_parent_xml = get_xml_path(f"{end}.xml")
    
    # We name the accumulated file based on the range it covers
    # E.g. merging 20 and 19 creates merged19_20
    
    for i in range(end - 1, start - 1, -1):
        child_xml = get_xml_path(f"{i}.xml")
        output_xml = get_xml_path(f"merged{i}_{end}.xml")
        
        parent_body_name = f"m{i+1}_top"
        child_body_name = f"m{i}_bottom"
        
        cmd = [
            SCRIPT_2_MERGE,
            child_xml,          # child xml
            current_parent_xml, # parent xml
            output_xml,         # output
            "--parent-body", parent_body_name,
            "--child-body", child_body_name,
            "--rotate-z", "90"
        ]
        run_command(cmd)
        
        # Update current parent to be the newly merged file for the next iteration
        current_parent_xml = output_xml
        
    return current_parent_xml

def step_3_prepare_base(start, end):
    """
    Prepares the base.xml by adding sites for the specific range.
    """
    print(f"\n--- Step 3: Preparing Base XML ---")
    
    src_base = get_xml_path("base.xml")
    dst_base = get_xml_path(f"base{start}_{end}.xml")
    
    if not os.path.exists(src_base):
        print(f"Error: {src_base} not found.")
        sys.exit(1)

    # Create comma-separated list of numbers
    nums = ",".join(str(x) for x in range(start, end + 1))
    
    cmd = [
        SCRIPT_3_BASE,
        src_base,
        dst_base,
        "--add", nums
    ]
    run_command(cmd)
    
    return dst_base

def step_4_final_assemble(merged_xml, base_xml, start, end):
    """
    Combines the merged module stack with the base environment.
    """
    print(f"\n--- Step 4: Final Assembly ---")
    
    final_xml = f"final{start}_{end}.xml" # Output in current dir or xml dir? User example put it in current.
    # Let's put it in the root folder as requested by user command structure usually
    
    cmd = [
        SCRIPT_4_ASSEMBLE,
        merged_xml,
        base_xml,
        final_xml
    ]
    run_command(cmd)
    
    return final_xml

def step_5_post_process(final_xml_path, start_mod_id, target_height, max_force):
    """
    Applies the specific adjustments requested:
    1. m{start}_top: pos="0 0 {target_height}", quat="0 0 0 1"
    2. drag sites: Update Z to {target_height} (keeping their X/Y relative or absolute?)
       User said: <site name="drag1" pos="0.028997516 0 0.7" />
       It seems we should set the Z component of drag sites to target_height.
    """
    print(f"\n--- Step 5: Post-Processing (Height: {target_height}) ---")
    
    tree = ET.parse(final_xml_path)
    root = tree.getroot()
    wb = root.find("worldbody")
    
    # 1. Modify Top Body Position
    top_body_name = f"m{start_mod_id}_top"
    found_top = False
    for body in wb.iter("body"):
        if body.get("name") == top_body_name:
            print(f"  -> Updating {top_body_name}: pos='0 0 {target_height}' quat='0 0 0 1'")
            body.set("pos", f"0 0 {target_height}")
            body.set("quat", "0 0 0 1")
            found_top = True
            break
            
    if not found_top:
        print(f"Warning: Could not find body {top_body_name} to adjust height.")

    # 2. Modify Drag Sites Z-position
    # We look for drag1, drag2, drag3 and update their Z to target_height
    for site in wb.iter("site"):
        name = site.get("name", "")
        if name.startswith("drag") or name == "m19_top": # Safety catch if user meant specific names
            current_pos = site.get("pos")
            if current_pos:
                vals = [float(x) for x in current_pos.split()]
                if len(vals) == 3:
                    new_pos = f"{vals[0]:.8g} {vals[1]:.8g} {target_height}"
                    print(f"  -> Updating site {name}: {current_pos} -> {new_pos}")
                    site.set("pos", new_pos)

    # 3.  Modify motor max value
    for motor in root.iter("motor"):
        name = motor.get("name", "unknown_motor")
        current_range = motor.get("ctrlrange", "")
        motor.set("ctrlrange", f"0 {max_force}")
        print(f"  -> Updating motor {name}: ctrlrange '{current_range}' -> '0 {max_force}'")

    tree.write(final_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"[OK] Final file written to: {final_xml_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate and assemble Mujoco modules")
    parser.add_argument("--start", type=int, required=True, help="Start module number (top)")
    parser.add_argument("--end", type=int, required=True, help="End module number (bottom)")
    parser.add_argument("--height", type=str, default="0.7", help="Z-height for the top module (e.g. 0.7)")
    parser.add_argument("--max-force", type=int, default=67, help="Motor max value (e.g. 67)")
    
    args = parser.parse_args()
    start = args.start
    end = args.end
    max_force = args.max_force

    # Validate
    if start >= end:
        print("Error: Start number must be less than End number.")
        sys.exit(1)
    target_height = calculate_target_height(start, end)

    # 1. Scale
    step_1_scale_modules(start, end)
    
    # 2. Merge
    merged_xml = step_2_merge_modules(start, end)
    
    # 3. Prepare Base
    base_prepared_xml = step_3_prepare_base(start, end)
    
    # 4. Final Assemble
    final_file = step_4_final_assemble(merged_xml, base_prepared_xml, start, end)
    
    # 5. Post Process (Fix Height & Quat)
    step_5_post_process(final_file, start, target_height, max_force)

if __name__ == "__main__":
    main()
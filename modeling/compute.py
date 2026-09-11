
import mujoco
import mujoco.viewer
import numpy as np
import time

# ================= 辅助函数：获取整个 Body 树的所有 ID =================
def get_tree_ids(mj_model, root_name):
    root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    if root_id == -1:
        raise ValueError(f"XML 中找不到名为 '{root_name}' 的 body！")
        
    tree_ids = [root_id]
    # 遍历后续的所有 body，如果它的父节点在当前树中，则把它也加进来
    for i in range(root_id + 1, mj_model.nbody):
        if mj_model.body_parentid[i] in tree_ids:
            tree_ids.append(i)
    return tree_ids

def add_force_arrow(viewer, start, force_world, scale=0.02, radius=0.003,
                    rgba=(1.0, 0.2, 0.2, 0.9), max_len=0.2):
    mag = np.linalg.norm(force_world)
    if mag < 1e-6:
        return False

    start = np.asarray(start, dtype=np.float64).reshape(3,)
    vec = scale * np.asarray(force_world, dtype=np.float64).reshape(3,)
    vec_norm = np.linalg.norm(vec)
    if vec_norm > max_len:
        vec = vec / vec_norm * max_len
    end = start + vec

    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return False

    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_ARROW,
        radius,
        start,
        end,
    )
    scn.geoms[scn.ngeom].rgba[:] = rgba
    scn.ngeom += 1
    return True

# ================= 1. 加载模型与数据 =================
model_path = "test\scene4.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

tube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tube")
actuator_rope1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pull_rope1")
actuator_rope2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pull_rope2")
actuator_rope3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pull_rope3")
actuator_rope4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pull_rope_add3")
actuator_rope5_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pull_rope_add2")

if -1 in [tube_id]:
    raise ValueError("找不到 tube，请检查 XML！")
if -1 in [actuator_rope1_id, actuator_rope2_id, actuator_rope3_id, actuator_rope4_id]:
    raise ValueError("找不到 pull_rope actuator，请检查 XML！")

# ================= 2. 获取 m20 和 m1 的所有子模块 ID =================
m20_tree_ids = get_tree_ids(model, "m20_bottom")
m1_tree_ids = get_tree_ids(model, "m1_top")

# 将两棵树的 ID 合并成一个集合 (set)，查找速度 O(1)，性能极佳！
tracked_body_ids = set(m20_tree_ids + m1_tree_ids)

print(f"成功加载！m20_bottom 树有 {len(m20_tree_ids)} 个模块，m1_top 树有 {len(m1_tree_ids)} 个模块。")
print(f"总计监控 {len(tracked_body_ids)} 个子模块与 tube 的接触力。")

# ================= 3. 启动渲染与控制循环 =================
with mujoco.viewer.launch_passive(model, data) as viewer:
    
    dt = model.opt.timestep 
    ramp_duration = 1  
    max_force = 6.0      

    while viewer.is_running():
        step_start = time.time()

        # ---------------- A. 施加正弦/余弦平滑增加的控制力 ----------------
        if data.time < 1: 
            current_force1 = 4 * data.time
        else:
            current_force1= 4
        data.ctrl[actuator_rope1_id] = current_force1    
        
        if data.time < 1:
            current_force4 = 0
        elif data.time < 2:
            current_force4 = 10 * (data.time - 1.0)
        else:
            current_force4= 10
        data.ctrl[actuator_rope4_id] = current_force4    
        
        if data.time < 1:
            current_force3 = 0
        elif data.time < 2:
            current_force3 = 7.5 * (data.time - 1.0)
        else:
            current_force3= 7.5
        data.ctrl[actuator_rope5_id] = current_force3    

        # ---------------- B. 物理步进 ----------------
        mujoco.mj_step(model, data)

       # ---------------- C. 提取接触力 ----------------
        forces = {bid: np.zeros(3) for bid in tracked_body_ids}
        contact_detected = False

        # 用来存要画的箭头：[(pos, force), ...]
        arrows_to_draw = []

        for i in range(data.ncon):
            contact = data.contact[i]

            # 只画真正参与求解的接触
            if contact.exclude != 0 or contact.efc_address < 0:
                continue

            b1_id = model.geom_bodyid[contact.geom1]
            b2_id = model.geom_bodyid[contact.geom2]

            is_b1_tracked = b1_id in tracked_body_ids
            is_b2_tracked = b2_id in tracked_body_ids
            is_b1_tube = b1_id == tube_id
            is_b2_tube = b2_id == tube_id

            if (is_b1_tracked and is_b2_tube) or (is_b2_tracked and is_b1_tube):
                contact_detected = True

                # 关心“模块受到的力”
                target_body_id = b1_id if is_b1_tracked else b2_id
                sign = -1.0 if target_body_id == b1_id else 1.0

                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, c_array)
                local_f = c_array[:3]

                c_frame = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
                world_f = sign * (c_frame.T @ local_f)

                forces[target_body_id] += world_f

                # 记录这个接触点上的箭头
                contact_pos = np.array(contact.pos, dtype=np.float64, copy=True)
                arrows_to_draw.append((contact_pos, np.array(world_f, dtype=np.float64, copy=True)))

        # ---------------- D. 打印信息 ----------------
        if int(data.time / dt) % int(0.1 / dt) == 0:
            if contact_detected:
                # === 计算所有模块的总和力 ===
                total_force = np.sum(list(forces.values()), axis=0)
                total_force_mag = np.linalg.norm(total_force)
                print(f"--- 时间: {data.time:.2f}s ---")
                print(f"★ [总和力] 所有抓取模块受力: F={total_force.round(3)} N (总大小: {total_force_mag:.3f} N)")
                for bid, force in forces.items():
                    if np.linalg.norm(force) > 1e-3:
                        b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
                        print(f"  [{b_name}] 受力: F={force.round(3)} N")
        
        # ---------------- E. 在 viewer 中画“机械臂-物体”接触力 ----------------
        with viewer.lock():
            # 不用 MuJoCo 自带 contact force，避免全场景都显示
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

            # 清空上一帧自定义箭头
            viewer.user_scn.ngeom = 0

            # 逐接触点画箭头
            for pos, f in arrows_to_draw:
                add_force_arrow(
                    viewer,
                    start=pos,
                    force_world=f,
                    scale=0.1,      # 1 N -> 0.02 m，可自己调
                    radius=0.003,
                    rgba=(1.0, 0.2, 0.2, 0.9),
                )                
      
        viewer.sync()

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

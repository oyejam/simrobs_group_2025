import mujoco
import mujoco.viewer
import time
import os
import numpy as np

def update_tendon_sites(model, data):
    """Обновляет site'ы для внутренних касательных между блоками"""
    
    try:
        # Получаем позиции ТЕЛ (не геометрий!)
        block1_pos = data.body("body_block1").xpos[:2]  # [x, y]
        block2_pos = data.body("body_block2").xpos[:2]  # [x, y]
        
        block1_radius = model.geom("block1").size[0]
        block2_radius = model.geom("block2").size[0]
        
        # Вычисляем внутренние касательные
        tangents = calculate_internal_tangents(block1_pos, block1_radius, 
                                             block2_pos, block2_radius)
        
        if tangents is None:
            print("No tangents found")
            return
        
        # Обновляем ГЛОБАЛЬНЫЕ позиции site'ов через site_xpos
        z_pos = 0.0
        
        # Точки касания на первом блоке
        t1_block1_pos = tangents[0][0]  # Глобальная позиция
        t2_block1_pos = tangents[1][0]  # Глобальная позиция
        
        # Точки касания на втором блоке  
        t1_block2_pos = tangents[0][1]  # Глобальная позиция
        t2_block2_pos = tangents[1][1]  # Глобальная позиция
        
        # Преобразуем глобальные позиции в относительные относительно родительских тел
        block1_xmat = data.body("body_block1").xmat.reshape(3, 3)
        block2_xmat = data.body("body_block2").xmat.reshape(3, 3)
        
        # Для body_block1
        rel_t1_block1 = block1_xmat.T @ (np.array([t1_block1_pos[0], t1_block1_pos[1], z_pos]) - data.body("body_block1").xpos)
        rel_t2_block1 = block1_xmat.T @ (np.array([t2_block1_pos[0], t2_block1_pos[1], z_pos]) - data.body("body_block1").xpos)
        
        # Для body_block2
        rel_t1_block2 = block2_xmat.T @ (np.array([t1_block2_pos[0], t1_block2_pos[1], z_pos]) - data.body("body_block2").xpos)
        rel_t2_block2 = block2_xmat.T @ (np.array([t2_block2_pos[0], t2_block2_pos[1], z_pos]) - data.body("body_block2").xpos)
        
        # Устанавливаем позиции через суставы (используем только x,y компоненты)
        # Индексы суставов должны соответствовать порядку в модели
        data.qpos[model.joint("t1_block1_joint").qposadr[0]:model.joint("t1_block1_joint").qposadr[0]+2] = rel_t1_block1[:2]
        data.qpos[model.joint("t2_block1_joint").qposadr[0]:model.joint("t2_block1_joint").qposadr[0]+2] = rel_t2_block1[:2]
        data.qpos[model.joint("t1_block2_joint").qposadr[0]:model.joint("t1_block2_joint").qposadr[0]+2] = rel_t1_block2[:2]
        data.qpos[model.joint("t2_block2_joint").qposadr[0]:model.joint("t2_block2_joint").qposadr[0]+2] = rel_t2_block2[:2]

    except Exception as e:
        print(f"Error in update_tendon_sites: {e}")
        import traceback
        traceback.print_exc()


def calculate_internal_tangents(center1, r1, center2, r2):
    """Вычисляет точки ВНУТРЕННИХ касательных между двумя окружностями"""
    
    # Вектор между центрами
    d_vec = center2 - center1
    d = np.linalg.norm(d_vec)
    
    print(f"Centers: {center1} -> {center2}, distance: {d:.4f}")
    print(f"Radii: r1={r1:.4f}, r2={r2:.4f}")
       
    # Угол между центрами
    theta = np.arctan2(d_vec[1], d_vec[0])
    
    # Угол для внутренних касательных
    # Для внутренних: sin(alpha) = (r1 + r2) / d
    alpha = np.arcsin((r1 + r2) / d)
    
    # Две внутренние касательные
    # Они симметричны относительно линии центров
    angles1 = [theta + alpha, theta - alpha]
    angles2 = [theta + alpha + np.pi, theta - alpha + np.pi]
    
    # Точки касания на первой окружности
    points1 = []
    for angle in angles1:
        x = center1[0] + r1 * np.cos(angle)
        y = center1[1] + r1 * np.sin(angle)
        points1.append(np.array([x, y]))
    
    # Точки касания на второй окружности  
    points2 = []
    for angle in angles2:
        x = center2[0] + r2 * np.cos(angle)
        y = center2[1] + r2 * np.sin(angle)
        points2.append(np.array([x, y]))
    
    print(f"Internal tangents found:")
    print(f"  Tangent 1: {points1[0]} -> {points2[0]}")
    print(f"  Tangent 2: {points1[1]} -> {points2[1]}")
    
    return [(points1[0], points2[0]), (points1[1], points2[1])]


def callback(model, data):
    data.ctrl[0] = 0.05 * np.sin(2 * np.pi * data.time / 2.0)
    update_tendon_sites(model, data)


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model.xml")

model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

mujoco.set_mjcb_control(callback)

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
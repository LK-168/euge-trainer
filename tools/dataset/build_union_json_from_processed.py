#!/usr/bin/env python3
"""
构建 SDXL Union JSON 清单 (硬编码配置版)

功能：
1. 扫描 PROCESSED_DIR 下的子文件夹 (例如 000001/)。
2. (可选) 如果配置了 CONTROLS_ROOT，会创建符号链接树：controls_root/<type>/<id>.png -> processed_data/<id>/<type>.png。
3. 生成 JSON 文件，包含 entries 和 modes 信息。

使用方法：
修改 --- CONFIGURATION --- 区域的路径，然后直接 python build_union_json.py
"""

import json
import os
import sys
from pathlib import Path
import random
# 1. handle_parquet.py 生成的 processed_data 绝对路径
PROCESSED_DIR = "/root/data-local/z_qwen/dataset/anicontrol-20k/processed_data"

# 2. 输出 JSON 文件的保存路径
OUT_JSON = "/root/data-local/z_qwen/dataset/anicontrol-20k/union_manifest_2cd_multi.json"

# 3. 控件图符号链接根目录
#    - 如果需要创建符号链接树 (用于分开存放不同类型的 condition)，请填写路径。
#    - 如果不需要创建 (例如你的 Dataset loader 支持直接读同级目录)，请设置为 None
CONTROLS_ROOT = "/root/data-local/z_qwen/dataset/anicontrol-20k/controls_root_2cd" 
# CONTROLS_ROOT = None 

# 4. 控制类型列表 (顺序很重要，对应模型通道的顺序)
CONTROL_TYPES = [
    # "openpose",
    "depth_midas",
    "canny",
    # "lineart_anime",
    # "lineart_realistic",
    # "manga_line",
    # "scribble_hed",
    # "scribble_pidinet",
    # "dwpose",
]

ENABLE_MULTI_CONDITION = True  # 是否启用多条件复合生成
MULTI_CONDITION_PROB = 0.5    # 多条件生成的概率
MAX_MULTI_CONDITIONS = 3       # 最多同时使用多少种条件

MAX_IMAGE_TRAINING_TYPES = 4   # 每张图最多训练多少种模式 (含单条件和多条件)

SEED = 42


def discover_samples(processed_dir: Path):
    """遍历 processed_dir，返回 [(image_key, sample_dir)] 列表。"""
    samples = []
    if not processed_dir.exists():
        print(f"❌ 错误: 目录不存在 - {processed_dir}")
        sys.exit(1)
        
    print(f"正在扫描目录: {processed_dir} ...")
    for child in processed_dir.iterdir():
        if child.is_dir():
            # 假设目录名即为 ID (如 000001)
            samples.append((child.name, child))
    
    # 按名称排序，保证顺序一致
    return sorted(samples)


def ensure_symlink_controls_tree(controls_root: Path, samples, control_types):
    """
    在 controls_root 下创建符号链接。
    返回: {image_key: [存在的控制类型列表]}
    """
    existing_map = {}
    controls_root.mkdir(parents=True, exist_ok=True)
    
    # 预先为每种控制类型创建文件夹
    for ctype in control_types:
        (controls_root / ctype).mkdir(parents=True, exist_ok=True)

    print(f"正在构建符号链接树到: {controls_root} ...")
    
    for image_key, sample_dir in samples:
        present = []
        for ctype in control_types:
            src = sample_dir / f"{ctype}.png"
            dst = controls_root / ctype / f"{image_key}.png"
            
            if src.exists():
                present.append(ctype)
                try:
                    # 检查目标是否存在
                    if dst.exists():
                        if dst.is_symlink():
                            # 如果是软链接，删除重建（防止指向错误）
                            dst.unlink()
                        else:
                            # 如果是普通文件，跳过不覆盖
                            continue
                    
                    # 创建软链接
                    os.symlink(src, dst)
                except Exception as e:
                    print(f"⚠️ 创建链接失败 {dst}: {e}")
        
        existing_map[image_key] = present
    
    return existing_map


def scan_processed_dir_only(samples, control_types):
    """
    不创建软链接，仅扫描 processed_data 目录下的文件存在情况。
    返回: {image_key: [存在的控制类型列表]}
    """
    existing_map = {}
    for image_key, sample_dir in samples:
        present = []
        for ctype in control_types:
            if (sample_dir / f"{ctype}.png").exists():
                present.append(ctype)
        existing_map[image_key] = present
    return existing_map


def read_caption(sample_dir: Path) -> str:
    cap_fp = sample_dir / "caption.txt"
    if cap_fp.exists():
        try:
            return cap_fp.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def build_entries(processed_dir: Path, samples, control_types, 
                  controls_present_map, 
                  multi_condition=True, 
                  multi_condition_prob=0.5, 
                  max_multi_conditions=3,
                  max_image_training_types=4):
    """
    构建 JSON entries 列表 (增强版：支持复合条件生成)。
    """
    entries = []
    
    for image_key, sample_dir in samples:
        main_image = sample_dir / "image.png"
        
        if not main_image.exists():
            continue 

        present = controls_present_map.get(image_key, [])
        present = [c for c in present if c in control_types]
        
        if not present:
            continue

        caption_text = read_caption(sample_dir)
        modes = []

        # --- 策略 A: 总是包含单条件样本 (基础能力) ---
        for c in present:
            modes.append([c])
        
        if multi_condition and len(present) >= 2:
            if random.random() < multi_condition_prob:
                current_limit = min(max_multi_conditions, len(present))
                
                # 确保下限是 2，上限至少是 2
                if current_limit < 2:
                    current_limit = 2
                
                num_to_pick = random.randint(2, current_limit)
                
                combo = random.sample(present, num_to_pick)

                combo.sort()
                
                if combo not in modes:
                    modes.append(combo)
        
        # 限制每张图的训练模式数量，防止过多
        if len(modes) > max_image_training_types:
            modes = random.sample(modes, max_image_training_types)

        entries.append({
            "file": f"{image_key}/image.png", 
            "modes": modes,
            "caption": caption_text,
        })
        
    return entries

def main():
    # 路径转换
    random.seed(SEED)
    processed_dir_path = Path(PROCESSED_DIR).resolve()
    out_json_path = Path(OUT_JSON).resolve()
    
    controls_root_path = None
    if CONTROLS_ROOT:
        controls_root_path = Path(CONTROLS_ROOT).resolve()

    # 1. 发现样本
    samples = discover_samples(processed_dir_path)
    if not samples:
        print(f"❌ 未在 {processed_dir_path} 发现任何子目录样本。")
        sys.exit(1)

    print(f"✅ 发现 {len(samples)} 个样本文件夹。")

    # 2. 统计控制图 (并可选创建软链接)
    if controls_root_path:
        controls_present_map = ensure_symlink_controls_tree(controls_root_path, samples, CONTROL_TYPES)
    else:
        print("CONTROLS_ROOT 未设置，跳过符号链接创建，直接扫描源目录。")
        controls_present_map = scan_processed_dir_only(samples, CONTROL_TYPES)

    # 3. 构建 Entries
    entries = build_entries(processed_dir_path, samples, CONTROL_TYPES, 
                            controls_present_map, multi_condition=ENABLE_MULTI_CONDITION, 
                            multi_condition_prob=MULTI_CONDITION_PROB, max_multi_conditions=MAX_MULTI_CONDITIONS,
                            max_image_training_types=MAX_IMAGE_TRAINING_TYPES)

    # 4. 组装 Manifest
    manifest = {
        "images_root": str(processed_dir_path),
        "controls_root": str(controls_root_path) if controls_root_path else "",
        "control_type_order": CONTROL_TYPES,
        "entries": entries,
    }

    # 5. 保存文件
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"✅ JSON 生成完毕: {out_json_path}")
    print(f"📊 统计:")
    print(f"   - 总 Entries: {len(entries)}")
    print(f"   - Images Root: {manifest['images_root']}")
    print(f"   - Controls Root: {manifest['controls_root'] or 'None'}")
    print(f"   - Control Types: {manifest['control_type_order']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
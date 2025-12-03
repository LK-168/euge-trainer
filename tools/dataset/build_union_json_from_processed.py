#!/usr/bin/env python3
"""
从 handle_parquet.py 生成的 processed_data 目录，构建 SDXL Union JSON 清单，
并可选创建 controls_root/<ctype>/<stem>.png 的符号链接树，以适配 SDXLUnionJSONDataset。

目录假设：
processed_data/
  <image_key>/
    image.png                # 主图像（来自列 'image'）
    canny.png                # 控制图（来自列名）
    depth_midas.png
    dwpose.png
    lineart_anime.png
    lineart_realistic.png
    manga_line.png
    scribble_hed.png
    scribble_pidinet.png
    caption.txt              # 可选

生成的 JSON 结构示例：
{
  "images_root": "/path/to/processed_data",
  "controls_root": "/path/to/controls_root",  # 可选（若创建链接树）
  "control_type_order": ["openpose","depth_midas","canny",...],
  "entries": [
    {"file": "000001/image.png", "modes": [["canny"],["depth_midas","openpose"]]},
    ...
  ]
}

用法：
python tools/dataset/build_union_json_from_processed.py \
  --processed-dir /data/anicontrol-20k/processed_data \
  --out-json /data/anicontrol-20k/union_manifest.json \
  --controls-root /data/anicontrol-20k/controls_root \
  --control-order openpose depth_midas canny lineart_anime lineart_realistic manga_line scribble_hed scribble_pidinet dwpose

注意：
- 若提供 --controls-root，将为每个样本创建符号链接 controls_root/<ctype>/<stem>.png 指向 processed_data/<image_key>/<ctype>.png
- stem 使用目录名 <image_key>。
- 若不提供 --controls-root，JSON 中该字段为空，数据集将尝试在线生成或使用零填充。
"""

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_CONTROL_TYPES = [
    # 你可以按需调整默认顺序；通常建议与训练配置的 num_control_type 对齐
    # "openpose",
    # "depth_midas",
    # "canny",
    # "lineart_anime",
    # "lineart_realistic",
    # "manga_line",
    # "scribble_hed",
    # "scribble_pidinet",
    # "dwpose",
    # "depth_midas",
    "canny",
]


def discover_samples(processed_dir: Path):
    """遍历 processed_dir，返回 [(image_key, sample_dir)] 列表。"""
    samples = []
    for child in processed_dir.iterdir():
        if child.is_dir():
            samples.append((child.name, child))
    return sorted(samples)


def ensure_symlink_controls_tree(controls_root: Path, samples, control_types):
    """在 controls_root 下为每个样本创建符号链接树 <ctype>/<stem>.png 指向 processed_data/<image_key>/<ctype>.png。
    返回存在的控制图类型映射 {image_key: [ctype, ...]}。
    """
    existing = {}
    controls_root.mkdir(parents=True, exist_ok=True)
    for ctype in control_types:
        (controls_root / ctype).mkdir(parents=True, exist_ok=True)

    for image_key, sample_dir in samples:
        present = []
        for ctype in control_types:
            src = sample_dir / f"{ctype}.png"
            dst = controls_root / ctype / f"{image_key}.png"
            if src.exists():
                try:
                    # 若目标已存在且是正确链接，跳过；若存在普通文件，保留现状
                    if dst.exists():
                        if dst.is_symlink():
                            # 允许覆盖坏链接
                            try:
                                dst.unlink()
                            except Exception:
                                pass
                        else:
                            # 普通文件存在则不覆盖
                            pass
                    if not dst.exists():
                        os.symlink(src, dst)
                    present.append(ctype)
                except Exception:
                    # 不因单个符号链接问题中断
                    pass
        existing[image_key] = present
    return existing


def read_caption(sample_dir: Path) -> str:
    cap_fp = sample_dir / "caption.txt"
    if cap_fp.exists():
        try:
            return cap_fp.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def build_entries(processed_dir: Path, samples, control_types, controls_present_map):
    """构建 JSON entries 列表。每个样本默认为：
    - modes: 将每个存在的控制类型作为一个单独条目 [ctype]
    - 若希望组合多控制，可在此逻辑中自定义（例如全部组合为一条），此处保守按单控制展开。
    """
    entries = []
    for image_key, sample_dir in samples:
        main_image = sample_dir / "image.png"
        if not main_image.exists():
            # 若主图不存在，跳过该样本
            continue
        present = controls_present_map.get(image_key, [])
        # 仅保留定义在 control_types 中的控制
        present = [c for c in present if c in control_types]
        caption_text = read_caption(sample_dir)
        if not present:
            # 没有控制图也可以加入，modes 为空；训练时将作为零控制图处理
            modes = []
        else:
            # 默认展开为多个单控制样本
            modes = [[c] for c in present]

        entries.append({
            "file": f"{image_key}/image.png",
            "modes": modes,
            "caption": caption_text,
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description="构建 SDXL Union JSON 清单，支持控件图符号链接树")
    parser.add_argument("--processed-dir", required=True, help="handle_parquet 输出的 processed_data 根目录")
    parser.add_argument("--out-json", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--controls-root", default=None, help="可选：为控件图创建符号链接树的根目录")
    parser.add_argument("--control-order", nargs="*", default=None, help="control_type_order 顺序，默认内置")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    out_json = Path(args.out_json).resolve()
    controls_root = Path(args.controls_root).resolve() if args.controls_root else None
    control_types = args.control_order if args.control_order else DEFAULT_CONTROL_TYPES

    if not processed_dir.exists():
        print(f"❌ processed-dir 不存在: {processed_dir}", file=sys.stderr)
        sys.exit(1)

    samples = discover_samples(processed_dir)
    if not samples:
        print(f"❌ 在 {processed_dir} 下未发现样本目录", file=sys.stderr)
        sys.exit(1)

    # 统计已存在的控制图；若提供 controls_root，则创建符号链接并统计，否则直接从样本目录统计
    if controls_root:
        controls_present_map = ensure_symlink_controls_tree(controls_root, samples, control_types)
    else:
        controls_present_map = {}
        for image_key, sample_dir in samples:
            present = []
            for ctype in control_types:
                if (sample_dir / f"{ctype}.png").exists():
                    present.append(ctype)
            controls_present_map[image_key] = present

    entries = build_entries(processed_dir, samples, control_types, controls_present_map)

    manifest = {
        "images_root": str(processed_dir),
        "controls_root": str(controls_root) if controls_root else "",
        "control_type_order": control_types,
        "entries": entries,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("✅ 已生成 Union JSON:", out_json)
    print("  images_root:", manifest["images_root"])
    print("  controls_root:", manifest["controls_root"] or "<未设置>")
    print("  control_type_order:", ", ".join(control_types))
    print("  entries 数量:", len(entries))


if __name__ == "__main__":
    main()

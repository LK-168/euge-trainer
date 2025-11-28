import os
import datasets
from PIL import Image
import sys
from tqdm import tqdm

# --- 您需要配置的区域 ---

# 1. 数据集源目录 (包含所有 .parquet 文件的文件夹)
SOURCE_DATA_DIR = "/data2/xujr/anicontrol-20k/data/"

# 2. Hugging Face 缓存目录 (请确保与您环境设置一致)
CACHE_DIR = "/data2/xujr/hf_cache"

# 3. 提取出的图片要保存到的目标目录 (脚本会自动创建)
OUTPUT_DIR = "/data2/xujr/anicontrol-20k/processed_data"

# 4. 您希望提取多少条记录进行检查？(设置为 None 则提取全部)
#    建议先设置一个小数目（比如 10 或 20）来快速测试！
NUM_ITEMS_TO_EXTRACT = None

# --- 主程序 ---

def main():
    print("="*80)
    print("开始从 Parquet 文件中提取图像 (结构化输出)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"文件将被保存到: {os.path.abspath(OUTPUT_DIR)}")

    try:
        print(f"正在从 '{SOURCE_DATA_DIR}' 加载数据集...")
        full_dataset = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(SOURCE_DATA_DIR, 'train-*.parquet'),
            cache_dir=CACHE_DIR
        )
        dataset_split = full_dataset['train']
    except Exception as e:
        print(f"\n❌ 加载数据集时发生致命错误:", file=sys.stderr)
        print(f"   具体错误: {e}", file=sys.stderr)
        sys.exit(1)

    total_size = len(dataset_split)
    items_to_process = total_size if NUM_ITEMS_TO_EXTRACT is None else min(NUM_ITEMS_TO_EXTRACT, total_size)
    
    print(f"✅ 数据集加载成功，总计 {total_size} 条记录。")
    print(f"将提取并整理前 {items_to_process} 条记录。")
    print("="*80)
    
    features = dataset_split.features
    image_columns = [name for name, feature in features.items() if isinstance(feature, datasets.Image)]
    string_columns = [name for name, feature in features.items() if feature.dtype == 'string']

    # 使用 tqdm 创建一个可视化的进度条
    for i in tqdm(range(items_to_process), desc="正在提取样本"):
        item = dataset_split[i]
        
        # 使用 image_key 或 序号 创建样本子目录名
        item_key = item.get('image_key', f'{i:06d}') # 使用6位补零序号
        sample_dir = os.path.join(OUTPUT_DIR, item_key)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存所有图像文件
        for col_name in image_columns:
            image_obj = item[col_name]
            if isinstance(image_obj, Image.Image):
                output_path = os.path.join(sample_dir, f"{col_name}.png")
                try:
                    image_obj.save(output_path)
                except Exception:
                    pass # 忽略保存失败的单个图片

        # 保存所有文本文件 (特别是 caption)
        if 'caption' in string_columns:
            caption_text = item['caption']
            if caption_text:
                caption_path = os.path.join(sample_dir, "caption.txt")
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(caption_text)

    print("\n" + "="*80)
    print("✅ 结构化图像提取完成！")

if __name__ == "__main__":
    main()
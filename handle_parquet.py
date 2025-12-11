import os
import datasets
from PIL import Image
import sys
from tqdm import tqdm
import multiprocessing

# --- 参数设置 ---
# L_K不会整parquet文件，只能先处理成图片再训练了

# 1. 数据集源目录
SOURCE_DATA_DIR = "/root/data-local/z_qwen/dataset/anicontrol-20k/data"

# 2. Hugging Face 缓存目录
CACHE_DIR = "/root/data-local/z_qwen/hf_cache"

# 3. 提取出的图片要保存到的目标目录
OUTPUT_DIR = "/root/data-local/z_qwen/dataset/anicontrol-20k/processed_data_all"

# 4. 希望保留的条件图类型列表
#    - 设置为 None: 提取所有包含图像的样本，不过滤。
#    - 设置为列表 (如 ['canny', 'depth_midas']): 只提取包含列表中至少一种条件图的样本
# KEEP_CONDITIONS = ['canny', 'depth_midas']
KEEP_CONDITIONS = None

# 5. 除了条件图外，还必须保留的图像列名
ALWAYS_KEEP_COLUMNS = ['image','caption'] 

# 6. 用于处理数据的 Worker 数量
# NUM_WORKERS = max(1, os.cpu_count() // 2)
NUM_WORKERS = 2

NUM_ITEMS_TO_PROCESS = None
# --- 全局变量（供子进程访问） ---
dataset_split = None
keep_conditions_set = None
always_keep_set = None
image_columns = None
output_dir_global = None

def process_and_save_item(i):
    try:
        item = dataset_split[i]
        
        # 判断此样本是否应该被处理
        item_is_valid = False                                                                                                                               
        present_images = {col_name for col_name in image_columns if isinstance(item.get(col_name), Image.Image)}

        if keep_conditions_set is None:
            if present_images:
                item_is_valid = True
        elif present_images.intersection(keep_conditions_set):
            item_is_valid = True
        
        if not item_is_valid:
            return "skipped"

        # 2. 保存
        item_key = item.get('image_key') or f'{i:06d}'
        sample_dir = os.path.join(output_dir_global, item_key)
        os.makedirs(sample_dir, exist_ok=True)
        

        # 遍历当前样本中所有存在的图片，只保存符合条件的
        for col_name in present_images:
            # 判断当前图片列名 (col_name) 是否需要保存:
            if keep_conditions_set is None or col_name in always_keep_set or col_name in keep_conditions_set:
                image_obj = item[col_name]
                output_path = os.path.join(sample_dir, f"{col_name}.png")
                if image_obj.mode != 'RGB':
                    image_obj = image_obj.convert('RGB')
                image_obj.save(output_path, 'PNG')

        caption_text = item.get('caption')
        if caption_text and isinstance(caption_text, str):
            caption_path = os.path.join(sample_dir, "caption.txt")
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption_text)
        
        return "saved"
    except Exception:
        return "failed"

def main():
    print("="*80)
    print("开始从 Parquet 文件中提取图像")
    
    if KEEP_CONDITIONS:
        print(f"过滤模式: 将只保留包含以下至少一种条件图的样本: {KEEP_CONDITIONS}")
        print(f"           对于有效样本，将只保存图像: {ALWAYS_KEEP_COLUMNS + KEEP_CONDITIONS}")
    else:
        print("将提取所有包含图像的样本。")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"文件将被保存到: {os.path.abspath(OUTPUT_DIR)}")
    print(f"将使用 {NUM_WORKERS} 个工作进程进行处理。")

    try:
        print(f"正在从 '{SOURCE_DATA_DIR}' 加载数据集...")
        full_dataset = datasets.load_dataset(
            "parquet", data_files=os.path.join(SOURCE_DATA_DIR, 'train-*.parquet'), cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"\n❌ 加载数据集时发生错误: {e}", file=sys.stderr)
        sys.exit(1)

    # 将配置加载到全局变量，以便子进程可以访问
    global dataset_split, keep_conditions_set, image_columns, output_dir_global, always_keep_set
    dataset_split = full_dataset['train']
    output_dir_global = OUTPUT_DIR
    if KEEP_CONDITIONS is not None:
        keep_conditions_set = set(KEEP_CONDITIONS)
    
    always_keep_set = set(ALWAYS_KEEP_COLUMNS)
    
    features = dataset_split.features
    image_columns = [name for name, feature in features.items() if isinstance(feature, datasets.Image)]
    
    total_size = len(dataset_split)
    items_to_process = total_size if NUM_ITEMS_TO_PROCESS is None else min(NUM_ITEMS_TO_PROCESS, total_size)
    
    print(f"✅ 数据集加载成功，总计 {total_size} 条记录。")
    print(f"将检查并处理前 {items_to_process} 条记录。")
    print("="*80)
    
    saved_count = 0
    skipped_count = 0
    failed_count = 0

    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        results_iterator = pool.imap_unordered(process_and_save_item, range(items_to_process))
        
        for result in tqdm(results_iterator, total=items_to_process, desc="正在处理样本"):
            if result == "saved":
                saved_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:
                failed_count += 1

    print("\n" + "="*80)
    print("✅ 结构化图像提取完成！")
    print(f"📊 处理结果统计:")
    print(f"   - 成功保存的样本数: {saved_count}")
    print(f"   - 因不满足条件而跳过的样本数: {skipped_count}")
    if failed_count > 0:
        print(f"   - 处理失败的样本数: {failed_count}")

if __name__ == "__main__":
    main()
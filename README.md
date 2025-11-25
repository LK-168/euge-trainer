# Euge Trainer

一个魔改 [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) 的 SD 微调的项目。

# 特性

在 kohya-ss 的基础上，对训练脚本进行了优化。支持 HuggingFace 数据集，支持多 GPU 训练，支持元数据文件，支持缓存潜变量。

# 使用方法

## 安装

```bash
git clone https://github.com/Eugeoter/euge-trainer.git
cd euge-trainer
git checkout dev
pip install -r requirements.txt
```

## 快速开始

### 准备数据

一个简单的数据集由两部分组成：图片文件和标注文件，标注文件放在用于图片文件同名的 txt 文本文件中，例如：

```
data/
|-- 0001.jpg # 图片文件
|-- 0001.txt # 标注文件，例如，"a dog playing with a ball"
|-- 0002.jpg
|-- 0002.txt
|-- ...
```

### 配置参数

在 [configs/config_train_sdxl.py](configs/config_train_sdxl.py) 内配置参数（或自行新建）后，执行 `accelerate launch train.py --trainer sdxl --config configs/config_train_sdxl.py` 即可开始训练。

其中，`--config` 参数为配置文件路径，`configs/config_train_sdxl.py` 为默认配置文件，您需要根据自己的需求修改配置文件，或指定您自己修改的配置文件路径。

完整的参数介绍请参考 [docs/config.md](docs/config.md)。

```

```

## ControlNet Union 多条件训练

Union ControlNet 支持多控制类型。示例启动：
```bash
accelerate launch train.py --trainer sdxl_controlnet_union --config configs/example_config_train_sdxl_controlnet_union.py
```

### 使用 JSON 清单控制训练样本与模式
示例 JSON：
```json
{
	"images_root": "/data/images",
	"controls_root": "/data/controls",
	"control_type_order": ["openpose","depth_midas","canny","lineart","normal","segment"],
	"entries": [
		{"file": "0001.jpg", "modes": [["canny"],["depth_midas"],["openpose"]]},
		{"file": "0002.jpg", "modes": [["canny","depth_midas"],["openpose"]]},
		{"file": "0003.jpg", "modes": [["openpose"]]}
	]
}
```
说明：
- 不在 entries 中的图片不训练。
- `modes` 中每个子列表是一条展开样本；有多项则代表复合条件同时激活。
- 同一原图在一个 epoch 中出现次数 = 其 `modes` 子列表数量。

路径约定：`controls_root/<control_type>/<stem>.png` 存放预生成条件图；缺失且 `generate_missing_controls=True` 时在线生成（需安装 `controlnet_aux`）

配置添加：
```python
config.union_json_path = 'data/union_manifest.json'
config.generate_missing_controls = False
config.num_control_type = 6
```
设置后自动切换为 `SDXLUnionJSONDataset`。

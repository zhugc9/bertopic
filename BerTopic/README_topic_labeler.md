# BERTopic主题标签生成器

## 简介
`topic_labeler.py` 是一个简洁的Python脚本，用于自动为BERTopic项目生成的主题生成中文议题名称。

## 功能特点
- 🎯 **简洁高效**: 遵循SOTA和KISS原则，代码简洁易维护
- 🤖 **AI驱动**: 使用大语言模型API自动生成高质量的中文标签
- 📊 **进度显示**: 使用tqdm显示处理进度
- 🔄 **错误处理**: 内置API调用错误处理和重试机制

## 使用方法

### 1. 配置参数
打开 `topic_labeler.py` 文件，在顶部的"用户配置区域"修改以下参数：

```python
# API配置
API_KEY = "your_api_key_here"  # 替换为您的API密钥

# 提示词配置（可选）
PROMPT_TEMPLATE = """基于关键词生成中文议题名称，要求10-15个汉字，输出JSON格式：
关键词：{keywords}
输出：{{"topic_label": "议题名称"}}"""

# 处理配置（可选）
MAX_RETRIES = 3  # API调用最大重试次数
REQUEST_DELAY = 0.5  # 请求间隔（秒）
TIMEOUT = 60  # 请求超时时间（秒）
```

### 2. 运行脚本
```bash
python topic_labeler.py
```

### 3. 查看结果
脚本会生成 `results/topics_summary_labeled.csv` 文件，包含原始数据和AI生成的中文标签。

## 输入文件
- `results/topics_summary.csv` - 由main.py生成的原始主题摘要文件

## 输出文件
- `results/topics_summary_labeled.csv` - 包含AI生成标签的完整结果文件

## 依赖包
```bash
pip install pandas tqdm requests pyyaml
```

## 注意事项
- 需要先运行 `main.py` 生成 `topics_summary.csv` 文件
- 请在脚本顶部的配置区域修改API密钥和其他参数
- 脚本会自动处理API调用限制，添加适当延迟和重试机制
- 提示词可以根据需要自定义，支持JSON格式输出

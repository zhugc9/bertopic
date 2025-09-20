# BERTopic 博士论文主题分析工具

## 🎯 快速开始（3步完成）

### 第1步：安装环境
```bash
双击运行："启动分析.bat"
```

### 第2步：准备数据
将Excel文件放入 `data/` 文件夹：
- `媒体_最终分析数据库.xlsx`
- `社交媒体_最终分析数据库.xlsx`

### 第3步：运行分析
```bash
双击运行："run_web_ui.bat"
```

---

## 📊 数据文件要求

### Excel文件格式
| 列名 | 说明 | 示例 |
|------|------|------|
| **Incident** | 文本内容（必须） | "中俄两国签署战略协议..." |
| Source | 数据来源 | "人民日报", "新华社" |
| 日期 | 时间信息 | 2024-01-15 |
| speaker | 发言者 | "外交部发言人" |

### 框架分析列（可选）
- `Frame_ProblemDefinition_Present`
- `Frame_CausalInterpretation_Present`  
- `Frame_MoralEvaluation_Present`
- `Frame_TreatmentRecommendation_Present`

---

## ⚙️ 关键参数设置

### 基础参数
打开 `config.yaml` 修改：

```yaml
# 主题数量控制
bertopic_params:
  min_topic_size: 15        # 调整这个数字
  nr_topics: "auto"         # 或设置具体数字如20
```

### 参数调优指南

| 数据量 | min_topic_size | 效果 |
|--------|----------------|------|
| <500条 | 5-10 | 更多细分主题 |
| 500-2000条 | 10-20 | 平衡的主题数量 |
| >2000条 | 20-50 | 主要主题聚焦 |

### 高级参数

```yaml
# 时间分析
analysis:
  time_analysis:
    bins: 10              # 时间段数量：5-20
    
# 语言模型
bertopic_params:
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

## 📊 使用流程

### 📁 数据准备
支持的数据格式：
```
Excel文件要求：
├── 必需列：文本内容列（默认名称："Incident"）
├── 可选列：来源信息（"Source"）
├── 可选列：时间信息（"日期"）
└── 可选列：框架分析列（"Frame_*_Present"格式）
```

### 🎯 参数配置
```yaml
# 核心参数调优指南
bertopic_params:
  min_topic_size: 15          # 小数据集用5-10，大数据集用20-50
  nr_topics: "auto"           # 可设置具体数字如20
  expert_keyword_extraction:
    enable_pos_patterns: true # 启用专家级关键词提取
    use_custom_stopwords: true # 使用政治新闻停用词库
```

### 📈 输出结果

#### 基础分析文件
- `topics_summary_enhanced.csv` - 增强主题摘要（包含语言构成）
- `cross_lingual_composition.csv` - 跨语言分析报告
- `dynamic_evolution_analysis.csv` - 动态演化数据
- `comprehensive_analysis_report.txt` - 综合分析总结

#### 学术级图表（PNG + PDF）
- `academic_topic_distribution.png/pdf` - 二维主题分布图
- `academic_topic_sizes.png/pdf` - 主题规模南丁格尔图  
- `academic_topic_evolution.png/pdf` - 主题时间演化图
- `cross_lingual_analysis.png/pdf` - 跨语言成分分析图

#### 交互式可视化
- `topic_visualization.html` - 主题关系交互图
- `topic_by_source.html` - 来源对比图表
- `topics_over_time.html` - 时间演化交互图
- `topic_frame_heatmap.html` - 框架关联热力图

## 🔧 高级功能详解

### 🎯 模块一：专家级关键词提取
- **基于词性标注的短语模式提取**：自动识别"新时代全面战略协作伙伴关系"等完整术语
- **自定义领域停用词表**：过滤政治新闻中的无意义词汇
- **多语言智能识别**：支持中英俄三语混合文本处理

```yaml
# 配置示例
expert_keyword_extraction:
  enable_pos_patterns: true           # 启用词性标注模式
  pos_patterns:                       # 语言特定的语法规则
    zh: '<n.*|a.*>*<n.*>+'           # 中文: (名词|形容词)*名词+
    en: '<JJ.*>*<NN.*>+'             # 英文: 形容词*名词+
    ru: '<A.*>*<N.*>+'               # 俄文: 形容词*名词+
  custom_stopwords_path: "stopwords/politics_stopwords.txt"
  use_custom_stopwords: true          # 启用自定义停用词
```

### 📊 模块二：出版级学术图表生成
- **高分辨率静态图表**：符合学术论文发表标准
- **带注释的二维主题分布图**：清晰展示主题关系
- **南丁格尔玫瑰图**：直观显示主题规模分布
- **PDF格式输出**：支持学术出版需求

### 🕐 模块三：动态主题演化分析
- **主题诞生与消亡追踪**：识别议题的生命周期
- **演化模式检测**：发现上升、下降、稳定和波动趋势
- **时间段自适应分析**：智能划分分析时间窗口

### 🌍 模块四：跨语言主题成分分析
- **语言构成统计**：精确计算每个主题的中俄英文档比例
- **主题类型分类**：自动识别"中俄共通议题"、"中方特色议题"等
- **可视化对比图表**：直观展示语言分布特征

### 💻 模块五：交互式Web应用界面
- **拖拽式文件上传**：无需命令行操作
- **实时参数调节**：滑块和选择框调整分析参数
- **一键运行分析**：自动化整个分析流程
- **结果在线预览**：直接查看图表和数据

## 📊 完整输出文件说明

### 增强的主题摘要文件
- `topics_summary_enhanced.csv`：包含原始关键词、增强关键词、语言构成等信息

### 学术级图表（PNG + PDF）
- `academic_topic_distribution.png/pdf`：二维主题分布图
- `academic_topic_sizes.png/pdf`：主题规模南丁格尔图
- `academic_topic_evolution.png/pdf`：主题时间演化图
- `cross_lingual_analysis.png/pdf`：跨语言分析图

### 专业分析报告
- `dynamic_evolution_analysis.csv`：动态演化详细数据
- `cross_lingual_composition.csv`：跨语言构成报告
- `comprehensive_analysis_report.txt`：综合分析总结

## 🛠️ 故障排除

### 常见问题解决

| 问题 | 解决方案 |
|------|----------|
| 🔍 **主题数量不理想** | 调整 `min_topic_size`：5-10(细粒度), 20-50(粗粒度) |
| 📄 **找不到数据文件** | 检查文件名和路径，确保中文字符正确 |
| 💾 **内存不足** | 增大 `min_topic_size` 或分批处理数据 |
| 🌐 **语言模型缺失** | 运行 `python -m spacy download zh_core_web_sm` |
| 🖥️ **Web界面启动失败** | 检查端口占用，手动运行 `streamlit run web_ui.py` |
| 📊 **图表显示异常** | 确保安装了完整的可视化依赖包 |

### 性能优化建议

```yaml
# 大数据集优化配置
bertopic_params:
  min_topic_size: 30              # 提高阈值
  calculate_probabilities: false  # 禁用概率计算提升速度
  umap_params:
    n_neighbors: 30               # 增加邻居数
system:
  use_gpu: true                   # 启用GPU加速（如果可用）
```

## 🔧 高级使用技巧

### 1. 针对中俄关系研究的参数优化
```yaml
bertopic_params:
  min_topic_size: 20                  # 适合大规模新闻数据
  expert_keyword_extraction:
    pos_patterns:
      zh: '<n.*|a.*>*<n.*>+'         # 捕获复合政治术语
      ru: '<A.*>*<N.*>+'             # 俄文形容词+名词模式
```

### 2. 学术论文图表定制
- 所有图表默认300 DPI高分辨率
- 支持PNG（用于在线发布）和PDF（用于印刷出版）
- 配色方案符合学术期刊标准

### 3. 大规模数据处理建议
```yaml
# 对于>10K文档的大数据集
bertopic_params:
  min_topic_size: 30                  # 提高最小主题大小
  umap_params:
    n_neighbors: 30                   # 增加邻居数
analysis:
  time_analysis:
    bins: 20                          # 增加时间分箱数
```

## 💡 最佳实践建议

### 1. 数据准备
- 确保文本长度>10字符
- 统一编码格式为UTF-8
- 时间字段格式标准化

### 2. 参数调优
- 小数据集(<1K)：`min_topic_size=5-10`
- 中等数据集(1K-10K)：`min_topic_size=10-20`
- 大数据集(>10K)：`min_topic_size=20-50`

### 3. 结果解读
- 关注"中俄共通议题"识别结果
- 分析主题演化的时间节点
- 对比增强关键词与原始关键词的差异

## 📚 技术文档

- 📖 **[喂给AI的开发者说明.txt](喂给ai的开发者说明.txt)** - 详细技术架构和API文档
- ⚙️ **[config.yaml](config.yaml)** - 完整配置参数说明
- 🔧 **[validate_config.py](validate_config.py)** - 配置验证工具
- 📋 本README - 包含完整的五大增强模块使用指南

## 🏗️ 系统架构

```
bertopic/
├── 🎯 核心分析引擎
│   ├── main.py                     # 主控制器
│   ├── topic_analyzer/             # 分析模块包
│   │   ├── data_loader.py         # 数据ETL处理器
│   │   ├── model.py               # BERTopic模型封装
│   │   ├── expert_keywords.py     # 专家级关键词提取
│   │   ├── academic_charts.py     # 学术图表生成器
│   │   ├── dynamic_evolution.py   # 动态演化分析
│   │   └── cross_lingual.py       # 跨语言分析
│   └── config.yaml                # 超参数配置文件
├── 💻 用户界面
│   ├── web_ui.py                  # Streamlit Web应用
│   ├── quick_start.py             # 命令行启动器
│   └── topic_labeler.py           # AI标签生成器
├── 🛠️ 工具脚本
│   ├── 启动分析.bat               # Windows一键启动
│   ├── run_web_ui.bat             # Web界面启动
│   └── validate_config.py         # 配置验证工具
└── 📊 资源文件
    ├── stopwords/                 # 自定义停用词库
    ├── data/                      # 输入数据目录
    └── results/                   # 输出结果目录
```

## 🤝 贡献指南

欢迎贡献代码和改进建议！请遵循以下原则：

- **SOTA原则**：采用最新的NLP技术和最佳实践
- **KISS原则**：保持代码简洁、模块化、易于维护
- **学术标准**：确保输出符合学术论文发表要求

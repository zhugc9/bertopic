# BERTopic多语种调参优化指南

## 📊 问题诊断与解决方案

### 🔴 Outlier比例过高 (>15%)
**症状**: 大量文档被标记为-1 (异常值)
**原因**: 聚类参数过于严格
**解决方案**:
```
优先级1: min_topic_size ↓ (减小20-30%)
优先级2: min_cluster_size ↓ (与min_topic_size保持一致)  
优先级3: min_samples ↓ (减至min_cluster_size的1/3-1/4)
优先级4: n_neighbors ↑ (增加10-20%)
```
**具体数值**:
- 中文: min_topic_size = 数据量×0.008
- 英文: min_topic_size = 数据量×0.01  
- 俄文: min_topic_size = 数据量×0.006
- 多语言混合: min_topic_size = 数据量×0.008

### 🟡 主题数量过少 (<数据量/100)
**症状**: 主题太宽泛，语义混杂
**原因**: 聚类过于粗糙
**解决方案**:
```
优先级1: min_topic_size ↓ (减小30-40%)
优先级2: n_components ↑ (增加1-2维)
优先级3: min_samples ↓ (降低门槛)
```

### 🟠 主题数量过多 (>数据量/50)
**症状**: 主题过于碎片化，语义重复
**原因**: 聚类过于细分
**解决方案**:
```
优先级1: min_topic_size ↑ (增加20-30%)
优先级2: n_neighbors ↓ (减少连接性)
优先级3: min_samples ↑ (提高门槛)
```

### 🟢 主题质量差 (关键词不相关/截断)
**症状**: 主题关键词语义不连贯
**原因**: 降维或嵌入问题
**解决方案**:
```
优先级1: 检查embedding_model (使用多语言模型)
优先级2: n_components ↑ (保留更多语义信息)
优先级3: n_neighbors ↑ (增强语义连接)
优先级4: 调整n_gram_range
```

## 🌍 语言特定优化策略

### 中文 (zh)
```json
{
  "基础参数": {
    "min_topic_size": "数据量 × 0.008",
    "n_gram_range": [1, 2],
    "n_neighbors": "基础值 × 0.9",
    "min_samples": "min_cluster_size / 4"
  },
  "特点": "语义密度高，词汇变化相对简单",
  "注意": "可以用稍大的主题，避免过度细分"
}
```

### 英文 (en)  
```json
{
  "基础参数": {
    "min_topic_size": "数据量 × 0.01", 
    "n_gram_range": [1, 3],
    "n_neighbors": "基础值",
    "min_samples": "min_cluster_size / 3"
  },
  "特点": "支持短语，语义相对规整",
  "注意": "可以使用3-gram捕获短语"
}
```

### 俄文 (ru)
```json
{
  "基础参数": {
    "min_topic_size": "数据量 × 0.006",
    "n_gram_range": [1, 2], 
    "n_neighbors": "基础值 × 1.2",
    "min_samples": "min_cluster_size / 5"
  },
  "特点": "形态变化丰富，词汇多样性高",
  "注意": "需要更小的主题和更多邻居捕获语义差异"
}
```

### 阿拉伯文 (ar)
```json
{
  "基础参数": {
    "min_topic_size": "数据量 × 0.007",
    "n_gram_range": [1, 2],
    "n_neighbors": "基础值 × 1.1", 
    "min_samples": "min_cluster_size / 4"
  },
  "特点": "词根系统复杂，语义变化多",
  "注意": "类似俄文，需要更细粒度的聚类"
}
```

### 日文 (ja)
```json
{
  "基础参数": {
    "min_topic_size": "数据量 × 0.009",
    "n_gram_range": [1, 2],
    "n_neighbors": "基础值 × 0.95",
    "min_samples": "min_cluster_size / 3"
  },
  "特点": "汉字+假名混合，语义密度中等",
  "注意": "介于中英文之间的处理方式"
}
```

## 📏 数据规模调参策略

### 小规模 (<5,000文档)
```json
{
  "min_topic_size": "max(8, 数据量×0.015)",
  "n_neighbors": 15,
  "n_components": 5,
  "min_samples": "min_cluster_size / 3",
  "策略": "保守参数，避免过度拟合"
}
```

### 中等规模 (5,000-50,000文档)  
```json
{
  "min_topic_size": "max(15, 数据量×0.008)",
  "n_neighbors": "15-25",
  "n_components": "5-7", 
  "min_samples": "min_cluster_size / 4",
  "策略": "平衡精度和召回"
}
```

### 大规模 (50,000-500,000文档)
```json
{
  "min_topic_size": "max(30, 数据量×0.002)",
  "n_neighbors": "25-40",
  "n_components": "5-8",
  "min_samples": "min_cluster_size / 5", 
  "策略": "注重计算效率和稳定性"
}
```

### 超大规模 (>500,000文档)
```json
{
  "min_topic_size": "max(50, 数据量×0.0005)",
  "n_neighbors": "40-60", 
  "n_components": "7-10",
  "min_samples": "min_cluster_size / 6",
  "策略": "分层采样+高效参数"
}
```

## 🛠 实用调参工作流

### 第一轮: 快速诊断
1. 查看Outlier比例
2. 统计主题数量分布  
3. 检查最小/最大主题大小
4. 评估关键词质量

### 第二轮: 针对性调整
```python
if Outlier > 15%:
    min_topic_size *= 0.7
    min_samples *= 0.8
elif 主题数 < 数据量/100:
    min_topic_size *= 0.6
    n_components += 1
elif 主题数 > 数据量/50:
    min_topic_size *= 1.3
    n_neighbors -= 5
```

### 第三轮: 精细优化
1. 调整embedding模型
2. 优化n_gram_range
3. 调节UMAP参数
4. 测试不同随机种子

## 📊 参数影响矩阵

| 参数             | 对Outlier  | 对主题数 | 对质量    | 对速度 
|------------      |-----------|----------|--------- |---------
| min_topic_size   | ⬇️⬇️⬇️   | ⬆️⬆️⬆️ | ⬆️⬆️    | ⬆️    
| min_samples      | ⬇️⬇️     | ⬆️⬆️    | ⬆️       | ⬆️    
| n_neighbors      | ⬇️       | ⬆️       | ⬆️⬆️    | ⬇️⬇️ 
| n_components     | ⬇️       | ⬆️       | ⬆️⬆️    | ⬇️ 
| min_cluster_size | ⬇️⬇️⬇️  | ⬆️⬆️⬆️  | ⬆️       | ⬆️ 

图例: ⬆️增加 ⬇️减少 数量表示影响强度

## 🚨 常见错误避免

### ❌ 错误做法
- 只调一个参数就期望大幅改善
- min_cluster_size与min_topic_size差异过大
- 忽略语言特性盲目套用英文参数
- 大数据集用小数据集参数

### ✅ 正确做法  
- 系统性调参，观察多个指标
- 保持相关参数的协调性
- 根据语言特点调整策略
- 数据规模驱动的参数选择

## 🎯 目标指标

### 理想状态
- **Outlier率**: < 10%
- **主题数**: 数据量/80 ± 20%
- **最小主题大小**: > min_topic_size×0.8
- **主题覆盖率**: > 90%
- **关键词相关性**: 主观评估良好

### 可接受范围
- **Outlier率**: 10-15%
- **主题数**: 数据量/120 - 数据量/60
- **最小主题大小**: > min_topic_size×0.6
- **主题覆盖率**: > 85%

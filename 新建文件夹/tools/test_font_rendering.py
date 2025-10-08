"""
字体渲染测试脚本
用于验证PDF图表中中文、俄文、英文是否正常显示
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# 设置输出目录
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 设置多语言字体
plt.rcParams.update({
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial', 'DejaVu Sans', 'sans-serif'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,  # TrueType字体嵌入
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'figure.dpi': 100
})

print("=" * 60)
print("📊 字体渲染测试")
print("=" * 60)
print(f"当前字体设置: {plt.rcParams['font.sans-serif']}")
print(f"PDF字体类型: {plt.rcParams['pdf.fonttype']}")
print()

# 测试1: 简单文字图
print("🧪 测试1: 生成多语言文字测试图...")
fig, ax = plt.subplots(figsize=(10, 6))

# 测试文本（中文、俄文、英文）
test_texts = [
    ("中文测试：主题分析", 0.9, 'red'),
    ("Russian: Анализ тем", 0.7, 'blue'),
    ("English: Topic Analysis", 0.5, 'green'),
    ("混合文本 Mixed Текст", 0.3, 'purple')
]

for text, y_pos, color in test_texts:
    ax.text(0.5, y_pos, text, fontsize=16, ha='center', va='center', color=color)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('多语言字体渲染测试 (Font Rendering Test)', fontsize=18, fontweight='bold')
ax.axis('off')

# 保存PDF和PNG
pdf_path = os.path.join(output_dir, 'test_font_rendering.pdf')
png_path = os.path.join(output_dir, 'test_font_rendering.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, bbox_inches='tight')
plt.close(fig)

print(f"  ✅ PDF保存到: {pdf_path}")
print(f"  ✅ PNG保存到: {png_path}")
print()

# 测试2: 带数据的图表
print("🧪 测试2: 生成多语言数据图表...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左侧：柱状图（中文标签）
categories_zh = ['国际关系', '经济政策', '文化交流', '军事安全']
values_zh = [45, 32, 28, 19]
ax1.bar(categories_zh, values_zh, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_ylabel('文档数量', fontsize=12)
ax1.set_title('中文主题分布', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', labelrotation=15)

# 右侧：柱状图（俄文标签）
categories_ru = ['Политика', 'Экономика', 'Культура', 'Оборона']
values_ru = [52, 38, 25, 22]
ax2.bar(categories_ru, values_ru, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax2.set_ylabel('Количество документов', fontsize=12)
ax2.set_title('Распределение тем (Russian)', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', labelrotation=15)

fig.suptitle('多语言图表测试 Multilingual Chart Test Многоязычный', fontsize=16, fontweight='bold')
plt.tight_layout()

# 保存PDF和PNG
pdf_path2 = os.path.join(output_dir, 'test_multilang_chart.pdf')
png_path2 = os.path.join(output_dir, 'test_multilang_chart.png')
fig.savefig(pdf_path2, bbox_inches='tight')
fig.savefig(png_path2, bbox_inches='tight')
plt.close(fig)

print(f"  ✅ PDF保存到: {pdf_path2}")
print(f"  ✅ PNG保存到: {png_path2}")
print()

# 测试3: 查看系统可用字体
print("🔍 测试3: 检查系统可用字体...")
available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
target_fonts = ['Microsoft YaHei', 'SimHei', 'Arial']

print(f"  系统共有 {len(available_fonts)} 种字体")
for font in target_fonts:
    matches = [f for f in available_fonts if font.lower() in f.lower()]
    if matches:
        print(f"  ✅ 找到字体: {font} (匹配: {matches[0]})")
    else:
        print(f"  ❌ 未找到字体: {font}")
print()

print("=" * 60)
print("✅ 测试完成！")
print("=" * 60)
print()
print("📋 检查结果：")
print(f"  1. 打开 {pdf_path}")
print(f"  2. 查看中文、俄文、英文是否都正常显示")
print(f"  3. 如果看到 □ 方块，说明字体未正确嵌入")
print(f"  4. PNG文件是备用参考（通常正常）")
print()

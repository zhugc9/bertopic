#!/usr/bin/env python3
"""
测试智能分析流程
"""

import pandas as pd
import sys
sys.path.insert(0, '.')

def test_analysis():
    print('=== 测试智能分析流程 ===')
    
    # 1. 加载数据
    try:
        df = pd.read_excel('temp/media_data.xlsx')
        documents = df['Unit_Text'].dropna().astype(str)
        documents = documents[documents.str.len() > 10].tolist()
        print(f'✓ 加载文档: {len(documents)} 条')
    except Exception as e:
        print(f'❌ 数据加载失败: {e}')
        return False
    
    # 2. 测试IntelligentTuner
    try:
        from topic_analyzer.intelligent_tuner import IntelligentTuner
        print('✓ IntelligentTuner导入成功')
        
        tuner = IntelligentTuner()
        print('✓ IntelligentTuner初始化成功')
        
        # 采样测试
        sample_docs = documents[:50]  # 小样本测试
        print(f'✓ 采样文档: {len(sample_docs)} 条')
        
        # 执行分析
        results = tuner.auto_tune(sample_docs)
        print(f'✓ auto_tune执行成功')
        print(f'✓ 返回结果类型: {type(results)}')
        
        if isinstance(results, dict):
            print(f'✓ 结果键: {list(results.keys())}')
            
            if 'data_features' in results:
                features = results['data_features']
                print(f'✓ data_features类型: {type(features)}')
                print(f'✓ 文档数量: {features.get("total_docs", "未知")}')
                print(f'✓ 主要语言: {features.get("dominant_language", "未知")}')
            
            if 'optimized_parameters' in results:
                params = results['optimized_parameters']
                print(f'✓ optimized_parameters类型: {type(params)}')
                if isinstance(params, dict):
                    print(f'✓ 参数键: {list(params.keys())}')
        
        print('🎉 智能分析测试通过!')
        return True
        
    except Exception as e:
        print(f'❌ 智能分析失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_analysis()

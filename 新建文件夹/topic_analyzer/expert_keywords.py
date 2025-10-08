"""
专家级关键词提取模块
====================
基于词性标注的短语模式和自定义停用词表的高级关键词提取
"""

import re
import spacy
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect
# 无异常类，手动捕获 RuntimeError

logger = logging.getLogger(__name__)


class ExpertKeywordExtractor:
    """专家级关键词提取器 - SOTA & KISS实现"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化专家级关键词提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.expert_config = config['bertopic_params'].get('expert_keyword_extraction', {})
        
        # 初始化语言模型
        self.nlp_models = {}
        self._load_language_models()
        
        # 加载自定义停用词
        self.custom_stopwords = set()
        self._load_custom_stopwords()
        
    def _load_language_models(self):
        """加载spaCy语言模型"""
        models_to_load = {
            'zh': 'zh_core_web_sm',
            'en': 'en_core_web_sm', 
            'ru': 'ru_core_news_sm'
        }
        
        for lang, model_name in models_to_load.items():
            try:
                self.nlp_models[lang] = spacy.load(model_name)
                logger.info(f"  ✓ 加载{lang}语言模型: {model_name}")
            except OSError:
                logger.warning(f"  ⚠ 未找到{lang}语言模型: {model_name}")
                logger.warning(f"    请运行: python -m spacy download {model_name}")
    
    def _load_custom_stopwords(self):
        """加载自定义停用词表"""
        if not self.expert_config.get('use_custom_stopwords', False):
            return
            
        stopwords_path = Path(self.expert_config.get('custom_stopwords_path', ''))
        if not stopwords_path.exists():
            logger.warning(f"  ⚠ 停用词文件不存在: {stopwords_path}")
            return
            
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.custom_stopwords.add(line.lower())
            logger.info(f"  ✓ 加载自定义停用词: {len(self.custom_stopwords)} 个")
        except Exception as e:
            logger.error(f"  ✗ 加载停用词失败: {e}")
    
    def detect_language(self, text: str) -> str:
        """检测文本语言"""
        try:
            lang = detect(text)
            # 映射langdetect的语言代码到我们的配置
            lang_mapping = {'zh-cn': 'zh', 'zh': 'zh', 'en': 'en', 'ru': 'ru'}
            return lang_mapping.get(lang, 'en')  # 默认英文
        except RuntimeError:
            return 'en'  # 检测失败时默认英文
    
    def extract_pos_phrases(self, text: str, language: str = None) -> List[str]:
        """
        基于词性标注模式提取短语
        
        Args:
            text: 输入文本
            language: 语言代码，如果为None则自动检测
            
        Returns:
            提取的短语列表
        """
        if not self.expert_config.get('enable_pos_patterns', False):
            return []
            
        # 自动语言检测
        if language is None and self.expert_config.get('pos_language_detection', True):
            language = self.detect_language(text)
        
        # 获取对应语言的模型
        if language not in self.nlp_models:
            return []
            
        nlp = self.nlp_models[language]
        
        # 获取词性模式
        pos_patterns = self.expert_config.get('pos_patterns', {})
        pattern = pos_patterns.get(language, '')
        
        if not pattern:
            return []
        
        try:
            # 处理文本
            doc = nlp(text)
            
            # 构建词性序列
            pos_sequence = []
            tokens = []
            
            for token in doc:
                if not token.is_space and not token.is_punct:
                    pos_sequence.append(f"<{token.pos_}>")
                    # 使用词形还原（lemma），而不是原始文本
                    tokens.append(token.lemma_.lower())
            
            # 使用正则表达式匹配模式
            pos_string = " ".join(pos_sequence)
            token_string = " ".join(tokens)
            
            # 找到匹配的模式
            matches = re.finditer(pattern, pos_string)
            phrases = []
            
            for match in matches:
                start_pos = len(pos_string[:match.start()].split())
                end_pos = start_pos + len(match.group().split())
                
                if start_pos < len(tokens) and end_pos <= len(tokens):
                    phrase = " ".join(tokens[start_pos:end_pos])
                    if len(phrase.strip()) > 2:  # 过滤太短的短语
                        phrases.append(phrase.strip())
            
            return phrases
            
        except Exception as e:
            logger.warning(f"  ⚠ 词性标注提取失败: {e}")
            return []
    
    def create_enhanced_vectorizer(self) -> CountVectorizer:
        """
        创建增强的向量化器，集成PoS模式和自定义停用词
        
        Returns:
            增强的CountVectorizer
        """
        # 基础参数（使用min_df=1避免小文档集冲突，通过max_features和停用词保证质量）
        vectorizer_params = {
            'ngram_range': tuple(self.config['bertopic_params']['n_gram_range']),
            'min_df': 1,  # 设为1避免BERTopic内部子集处理时的冲突
            'max_df': 0.95,
            'max_features': self.expert_config.get('max_features', 5000),  # 限制词汇表大小保证质量
            'lowercase': True
        }
        
        # 合并自定义停用词
        if self.custom_stopwords:
            vectorizer_params['stop_words'] = list(self.custom_stopwords)
            logger.info(f"  ✓ 使用自定义停用词: {len(self.custom_stopwords)} 个")
        else:
            vectorizer_params['stop_words'] = None
        
        # 如果启用PoS模式，使用自定义的token_pattern
        if self.expert_config.get('enable_pos_patterns', False):
            vectorizer_params['token_pattern'] = None  # 使用自定义tokenizer
            vectorizer_params['tokenizer'] = self._pos_aware_tokenizer
            logger.info("  ✓ 启用基于PoS的token化")
        
        return CountVectorizer(**vectorizer_params)
    
    def _pos_aware_tokenizer(self, text: str) -> List[str]:
        """
        基于PoS模式的自定义tokenizer（带词形还原）
        
        Args:
            text: 输入文本
            
        Returns:
            token列表（已词形还原）
        """
        # 首先提取PoS短语（已经是lemma形式）
        pos_phrases = self.extract_pos_phrases(text)
        
        # fallback：对整个文本进行词形还原
        language = self.detect_language(text)
        standard_tokens = []
        
        if language in self.nlp_models:
            nlp = self.nlp_models[language]
            doc = nlp(text)
            for token in doc:
                if (not token.is_stop and 
                    not token.is_punct and 
                    not token.is_space and
                    len(token.lemma_) > 1):
                    standard_tokens.append(token.lemma_.lower())
        else:
            # 没有模型时的简单fallback
            import re
            standard_tokens = re.findall(r'\b\w+\b', text.lower())
        
        # 合并PoS短语和标准tokens
        all_tokens = pos_phrases + standard_tokens
        
        # 去重并过滤
        unique_tokens = []
        seen = set()
        
        for token in all_tokens:
            token_clean = token.lower().strip()
            if (token_clean not in seen and 
                len(token_clean) > 1 and 
                token_clean not in self.custom_stopwords):
                unique_tokens.append(token_clean)
                seen.add(token_clean)
        
        return unique_tokens
    
    def enhance_topic_representation(self, 
                                   topic_model, 
                                   documents: List[str]) -> Dict[int, List[Tuple[str, float]]]:
        """
        增强主题表示，集成PoS短语
        
        Args:
            topic_model: 训练好的BERTopic模型
            documents: 文档列表
            
        Returns:
            增强的主题表示字典
        """
        enhanced_topics = {}
        
        # 获取原始主题
        topic_info = topic_model.get_topic_info()
        
        for topic_id in topic_info['Topic']:
            if topic_id == -1:  # 跳过离群点
                continue
                
            # 获取该主题的文档
            topic_docs = [doc for i, doc in enumerate(documents) 
                         if topic_model.topics_[i] == topic_id]
            
            # 为每个文档提取PoS短语
            all_phrases = []
            for doc in topic_docs[:100]:  # 限制处理文档数量以提高效率
                phrases = self.extract_pos_phrases(doc)
                all_phrases.extend(phrases)
            
            # 统计短语频率
            phrase_freq = {}
            for phrase in all_phrases:
                phrase_clean = phrase.lower().strip()
                if len(phrase_clean) > 2:
                    phrase_freq[phrase_clean] = phrase_freq.get(phrase_clean, 0) + 1
            
            # 选择高频短语
            top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # 获取原始主题词
            original_topic = topic_model.get_topic(topic_id)
            
            # 合并原始主题词和PoS短语
            enhanced_representation = []
            
            # 添加高频PoS短语（权重基于频率）
            max_freq = max([freq for _, freq in top_phrases]) if top_phrases else 1
            for phrase, freq in top_phrases:
                score = freq / max_freq * 0.8  # 调整权重
                enhanced_representation.append((phrase, score))
            
            # 添加原始主题词
            enhanced_representation.extend(original_topic[:5])
            
            # 按分数排序并去重
            seen_words = set()
            final_representation = []
            for word, score in sorted(enhanced_representation, key=lambda x: x[1], reverse=True):
                if word.lower() not in seen_words:
                    final_representation.append((word, score))
                    seen_words.add(word.lower())
                    if len(final_representation) >= 10:
                        break
            
            enhanced_topics[topic_id] = final_representation
        
        logger.info(f"  ✓ 增强了 {len(enhanced_topics)} 个主题的表示")
        return enhanced_topics


def install_language_models():
    """安装必要的spaCy语言模型"""
    import subprocess
    import sys
    
    models = [
        'zh_core_web_sm',
        'en_core_web_sm', 
        'ru_core_news_sm'
    ]
    
    for model in models:
        try:
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', model])
            print(f"✅ 成功安装: {model}")
        except subprocess.CalledProcessError:
            print(f"❌ 安装失败: {model}")


if __name__ == "__main__":
    # 安装语言模型
    install_language_models()

"""
多语言文本预处理模块
====================
实现高级多语言tokenizer和自定义CountVectorizer - SOTA & KISS实现
"""

import re
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from langdetect import detect, LangDetectException
import warnings
warnings.filterwarnings('ignore')

# 条件导入依赖库
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import pymorphy2
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class MultilingualTokenizer:
    """多语言智能分词器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化多语言分词器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化语言处理器
        self.language_processors = {}
        self._init_language_processors()
        
        # 加载自定义停用词
        self.custom_stopwords = set()
        self._load_custom_stopwords()
        
        # 语言检测缓存
        self.language_cache = {}
        
    def _init_language_processors(self):
        """初始化各语言的处理器"""
        self.logger.info("初始化多语言处理器...")
        
        # 中文处理器
        if JIEBA_AVAILABLE:
            self.language_processors['zh'] = {
                'type': 'jieba',
                'processor': jieba
            }
            self.logger.info("  ✓ 中文处理器 (jieba) 已加载")
        else:
            self.logger.warning("  ⚠ jieba未安装，中文处理将使用简单分词")
        
        # 英文处理器
        if SPACY_AVAILABLE:
            try:
                nlp_en = spacy.load('en_core_web_sm')
                self.language_processors['en'] = {
                    'type': 'spacy',
                    'processor': nlp_en
                }
                self.logger.info("  ✓ 英文处理器 (spaCy) 已加载")
            except OSError:
                self.logger.warning("  ⚠ 英文spaCy模型未安装，将尝试NLTK")
                if NLTK_AVAILABLE:
                    try:
                        lemmatizer = WordNetLemmatizer()
                        self.language_processors['en'] = {
                            'type': 'nltk',
                            'processor': lemmatizer
                        }
                        self.logger.info("  ✓ 英文处理器 (NLTK) 已加载")
                    except Exception as e:
                        self.logger.warning(f"  ⚠ NLTK初始化失败: {e}")
        
        # 俄文处理器
        if PYMORPHY_AVAILABLE:
            try:
                morph_ru = pymorphy2.MorphAnalyzer()
                self.language_processors['ru'] = {
                    'type': 'pymorphy',
                    'processor': morph_ru
                }
                self.logger.info("  ✓ 俄文处理器 (pymorphy2) 已加载")
            except Exception as e:
                self.logger.warning(f"  ⚠ pymorphy2初始化失败: {e}")
        elif SPACY_AVAILABLE:
            try:
                nlp_ru = spacy.load('ru_core_news_sm')
                self.language_processors['ru'] = {
                    'type': 'spacy',
                    'processor': nlp_ru
                }
                self.logger.info("  ✓ 俄文处理器 (spaCy) 已加载")
            except OSError:
                self.logger.warning("  ⚠ 俄文spaCy模型未安装")
        
        if not self.language_processors:
            self.logger.warning("⚠ 未加载任何高级语言处理器，将使用基础分词")
    
    def _load_custom_stopwords(self):
        """加载自定义停用词"""
        expert_config = self.config.get('bertopic_params', {}).get('expert_keyword_extraction', {})
        
        if not expert_config.get('use_custom_stopwords', False):
            return
            
        stopwords_path = Path(expert_config.get('custom_stopwords_path', ''))
        if not stopwords_path.exists():
            self.logger.warning(f"  ⚠ 停用词文件不存在: {stopwords_path}")
            return
            
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.custom_stopwords.add(line.lower())
            self.logger.info(f"  ✓ 加载自定义停用词: {len(self.custom_stopwords)} 个")
        except Exception as e:
            self.logger.error(f"  ✗ 加载停用词失败: {e}")
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码 ('zh', 'en', 'ru', 'unknown')
        """
        # 缓存检查
        text_hash = hash(text[:200])  # 只用前200字符计算hash
        if text_hash in self.language_cache:
            return self.language_cache[text_hash]
        
        detected_lang = 'unknown'
        
        try:
            # 使用langdetect
            lang = detect(text[:500])  # 只检测前500字符提高速度
            
            # 映射到我们支持的语言
            lang_mapping = {
                'zh-cn': 'zh', 'zh': 'zh',
                'en': 'en',
                'ru': 'ru'
            }
            detected_lang = lang_mapping.get(lang, 'unknown')
            
        except LangDetectException:
            # langdetect失败时使用字符特征判断
            if re.search(r'[\u4e00-\u9fff]', text):
                detected_lang = 'zh'
            elif re.search(r'[а-яё]', text, re.IGNORECASE):
                detected_lang = 'ru'
            elif re.search(r'[a-zA-Z]', text):
                detected_lang = 'en'
        
        # 缓存结果
        self.language_cache[text_hash] = detected_lang
        return detected_lang
    
    def tokenize_chinese(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 中文文本
            
        Returns:
            分词结果
        """
        if 'zh' in self.language_processors:
            processor = self.language_processors['zh']['processor']
            tokens = list(processor.cut(text))
            # 过滤单字符和标点
            return [token for token in tokens if len(token) > 1 and re.match(r'[\u4e00-\u9fff]+', token)]
        else:
            # 简单中文分词fallback
            words = re.findall(r'[\u4e00-\u9fff]+', text)
            return [word for word in words if len(word) > 1]
    
    def tokenize_english(self, text: str) -> List[str]:
        """
        英文词形还原
        
        Args:
            text: 英文文本
            
        Returns:
            词形还原结果
        """
        if 'en' in self.language_processors:
            processor_info = self.language_processors['en']
            
            if processor_info['type'] == 'spacy':
                # 使用spaCy
                doc = processor_info['processor'](text)
                tokens = []
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        not token.is_space and
                        len(token.lemma_) > 2 and
                        token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']):
                        tokens.append(token.lemma_.lower())
                return tokens
                
            elif processor_info['type'] == 'nltk':
                # 使用NLTK
                lemmatizer = processor_info['processor']
                words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
                tokens = []
                for word in words:
                    if len(word) > 2:
                        lemma = lemmatizer.lemmatize(word)
                        tokens.append(lemma)
                return tokens
        
        # 简单英文分词fallback
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [word for word in words if len(word) > 2]
    
    def tokenize_russian(self, text: str) -> List[str]:
        """
        俄文词形还原
        
        Args:
            text: 俄文文本
            
        Returns:
            词形还原结果
        """
        if 'ru' in self.language_processors:
            processor_info = self.language_processors['ru']
            
            if processor_info['type'] == 'pymorphy':
                # 使用pymorphy2
                morph = processor_info['processor']
                words = re.findall(r'\b[а-яё]+\b', text.lower(), re.IGNORECASE)
                tokens = []
                for word in words:
                    if len(word) > 2:
                        parsed = morph.parse(word)[0]
                        normal_form = parsed.normal_form
                        # 过滤常见的停用词性
                        if parsed.tag.POS not in ['PREP', 'CONJ', 'PRCL', 'INTJ']:
                            tokens.append(normal_form)
                return tokens
                
            elif processor_info['type'] == 'spacy':
                # 使用spaCy
                doc = processor_info['processor'](text)
                tokens = []
                for token in doc:
                    if (not token.is_stop and 
                        not token.is_punct and 
                        not token.is_space and
                        len(token.lemma_) > 2):
                        tokens.append(token.lemma_.lower())
                return tokens
        
        # 简单俄文分词fallback
        words = re.findall(r'\b[а-яё]+\b', text.lower(), re.IGNORECASE)
        return [word for word in words if len(word) > 2]
    
    def tokenize_text(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        智能多语言分词
        
        Args:
            text: 输入文本
            language: 指定语言，如果为None则自动检测
            
        Returns:
            分词结果
        """
        if not text or not isinstance(text, str):
            return []
        
        # 自动语言检测
        if language is None:
            language = self.detect_language(text)
        
        # 根据语言选择分词方法
        if language == 'zh':
            tokens = self.tokenize_chinese(text)
        elif language == 'en':
            tokens = self.tokenize_english(text)
        elif language == 'ru':
            tokens = self.tokenize_russian(text)
        else:
            # 混合语言或未知语言的处理
            tokens = self._tokenize_mixed_language(text)
        
        # 过滤停用词和无效token
        filtered_tokens = []
        for token in tokens:
            token_clean = token.lower().strip()
            if (len(token_clean) > 1 and 
                token_clean not in self.custom_stopwords and
                not re.match(r'^[0-9]+$', token_clean)):  # 排除纯数字
                filtered_tokens.append(token_clean)
        
        return filtered_tokens
    
    def _tokenize_mixed_language(self, text: str) -> List[str]:
        """
        混合语言文本分词
        
        Args:
            text: 混合语言文本
            
        Returns:
            分词结果
        """
        tokens = []
        
        # 分别提取不同语言的部分
        chinese_parts = re.findall(r'[\u4e00-\u9fff]+', text)
        english_parts = re.findall(r'\b[a-zA-Z]+\b', text)
        russian_parts = re.findall(r'\b[а-яё]+\b', text, re.IGNORECASE)
        
        # 分别处理
        for part in chinese_parts:
            if len(part) > 1:
                tokens.extend(self.tokenize_chinese(part))
        
        for part in english_parts:
            if len(part) > 2:
                tokens.extend(self.tokenize_english(part))
        
        for part in russian_parts:
            if len(part) > 2:
                tokens.extend(self.tokenize_russian(part))
        
        return tokens


class EnhancedMultilingualVectorizer:
    """增强的多语言向量化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增强向量化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化多语言分词器
        self.tokenizer = MultilingualTokenizer(config)
        
    def create_vectorizer(self) -> CountVectorizer:
        """
        创建自定义CountVectorizer
        
        Returns:
            配置好的CountVectorizer
        """
        self.logger.info("创建增强的多语言向量化器...")
        
        # 基础参数
        bertopic_params = self.config.get('bertopic_params', {})
        vectorizer_params = {
            'ngram_range': tuple(bertopic_params.get('n_gram_range', [1, 2])),
            'min_df': 1,  # 设为1避免BERTopic内部子集处理时的冲突
            'max_df': 0.95,
            'max_features': bertopic_params.get('max_features', 5000),  # 限制词汇表大小保证质量
            'lowercase': True,  # 我们在tokenizer中处理大小写
            'token_pattern': None,  # 使用自定义tokenizer
            'tokenizer': self._custom_tokenizer,
            'stop_words': None  # 停用词在tokenizer中处理
        }
        
        self.logger.info(f"  ✓ N-gram范围: {vectorizer_params['ngram_range']}")
        self.logger.info("  ✓ 启用多语言智能分词")
        
        return CountVectorizer(**vectorizer_params)
    
    def _custom_tokenizer(self, text: str) -> List[str]:
        """
        自定义tokenizer函数
        
        Args:
            text: 输入文本
            
        Returns:
            token列表
        """
        return self.tokenizer.tokenize_text(text)
    
    def get_language_statistics(self, documents: List[str]) -> Dict[str, Any]:
        """
        获取文档集的语言统计信息
        
        Args:
            documents: 文档列表
            
        Returns:
            语言统计字典
        """
        language_counts = {}
        total_docs = len(documents)
        
        # 采样分析（大数据集只分析前1000个）
        sample_docs = documents[:1000] if total_docs > 1000 else documents
        
        for doc in sample_docs:
            if doc and isinstance(doc, str):
                lang = self.tokenizer.detect_language(doc)
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # 计算百分比
        language_stats = {}
        for lang, count in language_counts.items():
            percentage = (count / len(sample_docs)) * 100
            language_stats[lang] = {
                'count': count,
                'percentage': percentage
            }
        
        return {
            'language_distribution': language_stats,
            'total_analyzed': len(sample_docs),
            'total_documents': total_docs
        }


# 便捷函数
def create_multilingual_vectorizer(config: Dict[str, Any]) -> CountVectorizer:
    """
    创建多语言向量化器的便捷函数
    
    Args:
        config: 配置字典
        
    Returns:
        配置好的CountVectorizer
    """
    enhancer = EnhancedMultilingualVectorizer(config)
    return enhancer.create_vectorizer()


def analyze_document_languages(documents: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    分析文档语言分布的便捷函数
    
    Args:
        documents: 文档列表
        config: 配置字典
        
    Returns:
        语言分析结果
    """
    enhancer = EnhancedMultilingualVectorizer(config)
    return enhancer.get_language_statistics(documents)


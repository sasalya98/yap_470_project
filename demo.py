import pandas as pd
import numpy as np
import joblib
import re
import os
import warnings

# Gerekli kütüphaneleri ve nesneleri import etme
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder

# Öznitelik çıkarma için gerekli kütüphaneler
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from gensim.models import KeyedVectors
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt_tab')
# --- Genel Ayarlar ---
warnings.filterwarnings('ignore')

# ==============================================================================
# BÖLÜM 1: Gerekli Nesnelerin ve Fonksiyonların Tanımlanması
# ==============================================================================

# --- Dosya Yolları ---
GLOVE_MODEL_PATH = "glove.twitter.27B.200d.word2vec.txt"
TFIDF_VECTORIZER_PATH = './saved_objects/tfidf_vectorizer.joblib'
EXTRA_SENTIMENT_PIPE_PATH = './saved_objects/pipe_extra_sentiment.joblib'
MODELS_DIR_NORMAL = './models-yontem-1/'
MODELS_DIR_ENHANCED = './models-yontem-1/enhanced/'

# --- LabelEncoder Alternatifi: Sabit (Hardcoded) Eşleme ---
CLASS_MAP = {
    'Irrelevant': 0,
    'Negative': 1,
    'Neutral': 2,
    'Positive': 3
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


# --- Gerekli NLTK Kaynaklarını İndirme (Daha Sağlam Yöntem) ---
def download_nltk_resource(resource_name, resource_path):
    """Bir NLTK kaynağının mevcut olup olmadığını kontrol eder ve yoksa indirir."""
    try:
        nltk.data.find(resource_path)
        print(f"- NLTK kaynağı '{resource_name}' zaten mevcut.")
        return True
    except LookupError:
        print(f"NLTK kaynağı '{resource_name}' bulunamadı, indiriliyor...")
        # quiet=False, indirme sürecini ve olası hataları görmek için.
        # Bu, ağ veya izin sorunlarını tespit etmeye yardımcı olur.
        try:
            nltk.download(resource_name, quiet=False)
            print(f"'{resource_name}' kaynağı başarıyla indirildi.")
            return True
        except Exception as e:
            print(f"HATA: '{resource_name}' kaynağı indirilirken bir sorun oluştu: {e}")
            return False

# --- Ön İşleme ve Öznitelik Çıkarma Fonksiyonları ---
stemmer_obj = PorterStemmer()
stop_words_list = set(stopwords.words('english'))

def preprocess_text(text):
    """Metin verisini temizler ve normalize eder."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    processed_tokens = [stemmer_obj.stem(word) for word in tokens if word not in stop_words_list and len(word) > 2]
    return " ".join(processed_tokens)

def extract_manual_features(tweet, entity, embedding_model, sent_analyzer):
    """Verilen bir tweet ve varlık için manuel öznitelikleri çıkarır."""
    if not isinstance(tweet, str) or not tweet.strip():
        return {"ctx_neg":0, "ctx_neu":1, "ctx_pos":0, "ctx_compound":0, "emb_sim":0, "fuzzy_max":0}

    tokens = word_tokenize(tweet.lower())
    
    try:
        idx = tokens.index(entity.lower())
        window = tokens[max(0, idx-3): idx] + tokens[idx+1: idx+4]
    except ValueError:
        window = tokens
    ctx_text = " ".join(window)
    ctx_sent = sent_analyzer.polarity_scores(ctx_text)

    vecs = [embedding_model[w] for w in tokens if w in embedding_model]
    if vecs and entity in embedding_model:
        tweet_vec = np.mean(vecs, axis=0)
        ent_vec = embedding_model[entity]
        emb_sim = float(np.dot(tweet_vec, ent_vec) / (np.linalg.norm(tweet_vec) * np.linalg.norm(ent_vec) + 1e-8))
    else:
        emb_sim = 0.0

    fuzzy_max = max((fuzz.partial_ratio(entity, w) for w in tokens), default=0)

    return {
        "ctx_neg": ctx_sent["neg"], "ctx_neu": ctx_sent["neu"], "ctx_pos": ctx_sent["pos"],
        "ctx_compound": ctx_sent["compound"], "emb_sim": emb_sim, "fuzzy_max": fuzzy_max
    }

def create_feature_vector(raw_tweet, raw_entity, version, resources):
    """Tek bir ham girdiden modelin anlayacağı öznitelik vektörünü oluşturur."""
    processed_tweet = preprocess_text(raw_tweet)
    tfidf_features = resources['tfidf_vectorizer'].transform([processed_tweet])
    
    manual_features_dict = extract_manual_features(raw_tweet, raw_entity, resources['embedding_model'], resources['sentiment_analyzer'])
    manual_features_array = np.array(list(manual_features_dict.values())).reshape(1, -1)
    
    contains_entity_flag = 1 if raw_entity.lower() in raw_tweet.lower() else 0
    explicit_entity_feature = np.array([[contains_entity_flag]])
    
    normal_features = hstack([
        tfidf_features,
        csr_matrix(explicit_entity_feature),
        csr_matrix(manual_features_array)
    ], format='csr')
    
    if version == 'enhanced':
        proba_features = resources['extra_sentiment_pipe'].predict_proba([processed_tweet])
        return hstack([normal_features, csr_matrix(proba_features)], format='csr')
    else:
        return normal_features

def load_all_resources():
    """
    Demo için gerekli olan tüm büyük nesneleri bir kerede yükler.
    Bu fonksiyon betik çalıştığında sadece bir kez çağrılır.
    """
    print("\n--- Adım 1: Gerekli Nesneler Yükleniyor (Bu işlem biraz sürebilir)... ---")
    try:
        # --- Gerekli NLTK Kaynaklarını İndirme ---
        print(">>> NLTK kaynakları kontrol ediliyor/indiriliyor...")
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        print("NLTK kaynakları hazır.\n")

        # Büyük dosyaları yükle
        print("\n>>> Büyük modeller ve vektörleyiciler yükleniyor...")
        embedding_model = KeyedVectors.load_word2vec_format(GLOVE_MODEL_PATH, binary=False)
        print("- GloVe modeli yüklendi.")
        
        tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        print("- TF-IDF Vektörleyici yüklendi.")
        
        extra_sentiment_pipe = joblib.load(EXTRA_SENTIMENT_PIPE_PATH)
        print("- Ek Duygu Pipeline'ı yüklendi.")

        resources = {
            "embedding_model": embedding_model,
            "tfidf_vectorizer": tfidf_vectorizer,
            "sentiment_analyzer": SentimentIntensityAnalyzer(),
            "extra_sentiment_pipe": extra_sentiment_pipe
        }
        print("\nTüm yardımcı nesneler başarıyla belleğe yüklendi.")
        return resources
    except FileNotFoundError as e:
        print(f"\nHATA: Gerekli bir dosya bulunamadı! - {e}")
        return None
    except Exception as e:
        print(f"\nNESNELER YÜKLENİRKEN BEKLENMEDİK BİR HATA OLUŞTU: {e}")
        return None

# ==============================================================================
# BÖLÜM 2: Ana Demo Uygulaması
# ==============================================================================
if __name__ == "__main__":
    print(">>> İnteraktif Duygu Analizi Demosu Başlatılıyor...")
    
    # Tüm ağır kaynakları betiğin başında SADECE BİR KEZ yükle
    resources = load_all_resources()
    
    if resources is None:
        print("\nKaynaklar yüklenemediği için demo sonlandırılıyor.")
        exit()

    print("\n" + "="*60)
    print("DEMO BAŞLADI - Çıkmak için herhangi bir alana 'exit' yazın.")
    print("="*60)
    
    model_choices = {
        "1": "ExtraTrees", "2": "MLP", "3": "RandomForest", 
        "4": "XGBoost", "5": "LinearSVC", "6": "LogisticRegression"
    }
    model_filenames = {
        "ExtraTrees": {"normal": "best_extra_trees.pkl", "enhanced": "et_enhanced.pkl"},
        "MLP": {"normal": "best_mlp.pkl", "enhanced": "mlp_enhanced.pkl"},
        "RandomForest": {"normal": "best_rf.pkl", "enhanced": "rf_enhanced.pkl"},
        "XGBoost": {"normal": "best_xgb_sklearn.pkl", "enhanced": "xgb_enhanced_sklearn.pkl"},
        "LinearSVC": {"normal": "best_lsvc.pkl", "enhanced": "lsvc_enhanced.pkl"},
        "LogisticRegression": {"normal": "best_logreg.pkl", "enhanced": "lr_enhanced.pkl"}
    }
    
    while True:
        print("\nLütfen kullanmak istediğiniz modeli seçin:")
        for key, value in model_choices.items():
            print(f"  {key}: {value}")
        model_choice = input("Seçiminiz (1-6): ")
        if model_choice.lower() == 'exit': break
        if model_choice not in model_choices:
            print("Geçersiz seçim, lütfen tekrar deneyin.")
            continue
        
        selected_model_name = model_choices[model_choice]

        version_choice = input("Model versiyonunu seçin ('normal' veya 'enhanced'): ").lower()
        if version_choice.lower() == 'exit': break
        if version_choice not in ['normal', 'enhanced']:
            print("Geçersiz versiyon, lütfen 'normal' veya 'enhanced' yazın.")
            continue
            
        custom_tweet = input("Lütfen bir cümle (tweet) girin: ")
        if custom_tweet.lower() == 'exit': break
            
        custom_entity = input("Lütfen cümlenin içindeki varlığı (entity) girin: ")
        if custom_entity.lower() == 'exit': break

        try:
            filename = model_filenames[selected_model_name][version_choice]
            model_dir = MODELS_DIR_ENHANCED if version_choice == 'enhanced' else MODELS_DIR_NORMAL
            model_path = os.path.join(model_dir, filename)
            
            print("-" * 20)
            print(f"... '{selected_model_name} ({version_choice})' modeli yükleniyor ve tahmin yapılıyor ...")
            
            model = joblib.load(model_path)
            
            # Ağır kaynakları tekrar yüklemek yerine, bellekteki nesneleri kullan
            feature_vector = create_feature_vector(
                custom_tweet, custom_entity, version_choice, resources
            )
            
            prediction_numeric = model.predict(feature_vector)[0]
            prediction_text = INV_CLASS_MAP.get(prediction_numeric, "Bilinmeyen Etiket")
            
            print("\n" + "--- SONUÇ ---".center(30))
            print(f"Girilen Cümle: \"{custom_tweet}\"")
            print(f"Analiz Edilen Varlık: \"{custom_entity}\"")
            print(f"Modelin Tahmin Ettiği Duygu:  --->   {prediction_text.upper()}   <---")
            
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(feature_vector)[0]
                print("\nTahmin Olasılıkları:")
                classes_for_proba = model.classes_ if hasattr(model, 'classes_') else list(CLASS_MAP.keys())
                for i, class_name in enumerate(INV_CLASS_MAP.values()):
                     class_index = list(classes_for_proba).index(CLASS_MAP[class_name] if pd.api.types.is_numeric_dtype(classes_for_proba) else class_name)
                     print(f"  - {class_name}: {prediction_proba[class_index]:.2%}")
            else:
                print("\nNot: Bu model ('LinearSVC') olasılık tahmini sağlamamaktadır.")

            print("-" * 30)

        except FileNotFoundError:
            print(f"\nHATA: Model dosyası bulunamadı! Yol: {model_path}")
        except Exception as e:
            print(f"\nBEKLENMEDİK BİR HATA OLUŞTU: {e}")

    print("\nDemo sonlandırıldı. Görüşmek üzere!")

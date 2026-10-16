import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay

# Uyarıları gizle
warnings.filterwarnings('ignore', category=FutureWarning)

# 1. Veri setini oku
df = pd.read_csv('emails.csv')
print(f"Toplam e-posta sayısı: {len(df)}")

# 2. Özellik (X) ve Etiket (y) olarak ayır
X = df['text']
y = df['spam']

# 3. Veri setini %80 eğitim, %20 test olarak böl
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF Vektörleştirici ile metinleri sayısal formata çevir
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

print("Veri ön işleme ve vektörleştirme başarıyla tamamlandı!")

# 5. Tüm Modelleri Tanımla (SVM için modern CalibratedClassifierCV kullanıldı)
svm_base = SVC(random_state=42)
svm_calibrated = CalibratedClassifierCV(svm_base, method='sigmoid')

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": svm_calibrated,
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# 6. Modelleri Eğit ve Değerlendir
trained_models = {}
plt.figure(figsize=(10, 8))

for name, model in models.items():
    print(f"\n--- {name} Eğitiliyor ve Değerlendiriliyor ---")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Doğruluk (Accuracy): {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Eğrisi için çizime ekle
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=plt.gca(), name=name)

# 7. ROC Grafiğini Düzenle ve Göster
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')
plt.title('Spam Tespiti - 5 Modelin ROC Eğrileri Karşılaştırması')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
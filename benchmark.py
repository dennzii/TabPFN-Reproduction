import torch
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from train import Transformer_TabPFN
# ----------------------------------------------------
# 1. 32'Lİ ENSEMBLE DESTEKLİ TAHMİN FONKSİYONU
# ----------------------------------------------------
def predict(model, X_train, y_train, X_test, max_features=100, n_ensemble=32, return_probs=False):
    """
    TabPFN ile 32'li veri permütasyonu ensembling yöntemiyle tahmin yapar.
    n_ensemble=1  -> TabPFN_n.e. (Tek geçiş - 0.05s)
    n_ensemble=32 -> TabPFN (Makaledeki tam versiyon - ~0.5s)
    """
    device = next(model.parameters()).device
    model.eval()

    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

    X_train = X_train.to(device)
    y_train = y_train.to(device).view(-1, 1)
    X_test = X_test.to(device)

    N_train, k = X_train.shape
    N_test = X_test.shape[0]
    num_classes = len(torch.unique(y_train))

    # Z-Score Normalizasyonu (Appendix C.2.1)
    mean = X_train.mean(dim=0, keepdim=True)
    std = X_train.std(dim=0, keepdim=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    scale = max_features / k
    all_ensemble_probs = []

    with torch.no_grad():
        for ensemble_idx in range(n_ensemble):
            if ensemble_idx == 0:
                # İlk üye: Orijinal sıra (Permütasyonsuz)
                feat_perm = torch.arange(k, device=device)
                class_perm = torch.arange(num_classes, device=device)
            else:
                # Sütunları ve Sınıfları rastgele karıştır (Makale Sayfa 29)
                feat_perm = torch.randperm(k, device=device)
                class_perm = torch.randperm(num_classes, device=device)

            # 1. Özellikleri permüte et ve 100 boyuta padle:
            X_tr_perm = X_train_norm[:, feat_perm]
            X_te_perm = X_test_norm[:, feat_perm]

            X_tr_pad = torch.zeros((N_train, max_features), device=device)
            X_te_pad = torch.zeros((N_test, max_features), device=device)
            X_tr_pad[:, :k] = X_tr_perm * scale
            X_te_pad[:, :k] = X_te_perm * scale

            # 2. Sınıf etiketlerini permüte et:
            y_tr_mapped = class_perm[y_train.long().squeeze(-1)].unsqueeze(-1).float()

            # 3. Girdi Matrisini Birleştir: [X, y] ve [X, 0]
            train_tokens = torch.cat([X_tr_pad, y_tr_mapped], dim=1)
            test_tokens = torch.cat([X_te_pad, torch.zeros((N_test, 1), device=device)], dim=1)
            x_full = torch.cat([train_tokens, test_tokens], dim=0).unsqueeze(0)

            # 4. Modele sor:
            logits = model(x=x_full, n_train=N_train)
            active_logits = logits.squeeze(0)[:, :num_classes]
            probs_perm = torch.softmax(active_logits, dim=-1)

            # 5. Olasılıkları orijinal sınıf sırasına geri çevir:
            probs_orig = probs_perm[:, class_perm]
            all_ensemble_probs.append(probs_orig)

    # 32 geçişin ortalamasını al (Ensemble):
    final_probabilities = torch.stack(all_ensemble_probs).mean(dim=0)

    if return_probs:
        return final_probabilities.cpu().numpy()
    
    predictions = torch.argmax(final_probabilities, dim=-1)
    return predictions.cpu().numpy()


# ----------------------------------------------------
# 2. BENCHMARK DEĞERLENDİRME DÖNGÜSÜ
# ----------------------------------------------------
BENCHMARK_DATASET_IDS = [
    11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 
    50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1462, 
    1464, 1480, 1494, 1510, 6332, 23381, 40966, 40975, 40982, 40994
]

def evaluate_tabpfn_on_benchmark(model, n_ensemble=32):
    results = {'accuracy': {}, 'roc_auc_ovo': {}, 'log_loss': {}}
    
    print(f"\n--- 30 Veri Seti Kıyaslaması Başlatılıyor (n_ensemble={n_ensemble}) ---")
    print(f"{'Veri Seti':<28} | {'ROC AUC (OVO)':<14} | {'Accuracy':<10} | {'Log Loss':<10}")
    print("-" * 72)

    for data_id in BENCHMARK_DATASET_IDS:
        name = f"ID_{data_id}"
        try:
            data = fetch_openml(data_id=data_id, as_frame=True)
            X, y = data.data, data.target
            name = data.details['name']
            
            # Kategorik sütunları dönüştür:
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                X[cat_cols] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit_transform(X[cat_cols])
            
            X = SimpleImputer(strategy='median').fit_transform(X)
            y = LabelEncoder().fit_transform(y)
            
            if len(X) > 1000:
                X, _, y, _ = train_test_split(X, y, train_size=1000, stratify=y, random_state=42)
                
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
            num_classes = len(np.unique(y_train))
            
            # 32'li Ensemble Tahmin:
            probs = predict(model, X_train, y_train, X_test, n_ensemble=n_ensemble, return_probs=True)
            preds = probs.argmax(axis=1)
            
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs[:, 1]) if num_classes == 2 else roc_auc_score(y_test, probs, multi_class='ovo')
            loss = log_loss(y_test, probs, labels=list(range(num_classes)))
            
            results['accuracy'][name] = acc
            results['roc_auc_ovo'][name] = auc
            results['log_loss'][name] = loss
            
            print(f"{name[:26]:<28} | {auc:<14.4f} | %{acc*100:<9.2f} | {loss:<10.4f}")
            
        except Exception as e:
            print(f"ID {data_id} ({name}) atlandı: {e}")

    mean_auc = np.mean(list(results['roc_auc_ovo'].values()))
    mean_acc = np.mean(list(results['accuracy'].values()))
    mean_loss = np.mean(list(results['log_loss'].values()))
    
    print("=" * 72)
    print(f"{'GENEL ORTALAMA':<28} | {mean_auc:<14.4f} | %{mean_acc*100:<9.2f} | {mean_loss:<10.4f}")
    print("=" * 72)
    return results


# 1. Modeli oluştur ve ağırlıkları yükle:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer_TabPFN().to(device)

checkpoint = torch.load("/content/drive/MyDrive/tabpfn_step_18000.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print("Aendi!")

# 2. Resmi 30 veri seti testini başlat:
benchmark_results = evaluate_tabpfn_on_benchmark(model)
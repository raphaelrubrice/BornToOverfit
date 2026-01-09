# ═══════════════════════════════════════════════════════════════════════════════
# CELLULES À AJOUTER À TON NOTEBOOK POUR L'ENTRAÎNEMENT FINAL
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# CELLULE 1 : Configuration pour entraînement final
# ═══════════════════════════════════════════════════════════════════════════════

"""
## **Entraînement Final (Train + Validation)**

Une fois que tu as trouvé les meilleurs hyperparamètres, tu peux entraîner
le modèle final sur train + validation combinés pour maximiser les données
d'entraînement avant de prédire sur le test set.
"""

# Configuration
FINAL_LOSS = "mse"  # ou "infonce" ou "triplet" selon ce qui marche le mieux
FINAL_EPOCHS = 50   # Peut-être augmenter puisque pas d'early stopping
FINAL_LR = 5e-4
FINAL_MODEL = "model_final_full.pt"

import os
os.environ['FINAL_LOSS'] = FINAL_LOSS
os.environ['FINAL_EPOCHS'] = str(FINAL_EPOCHS)
os.environ['FINAL_LR'] = str(FINAL_LR)
os.environ['FINAL_MODEL'] = FINAL_MODEL

print(f"🔧 Configuration entraînement final:")
print(f"   Loss: {FINAL_LOSS}")
print(f"   Epochs: {FINAL_EPOCHS}")
print(f"   LR: {FINAL_LR}")
print(f"   Output: {FINAL_MODEL}")


# ═══════════════════════════════════════════════════════════════════════════════
# CELLULE 2 : Lancer l'entraînement final
# ═══════════════════════════════════════════════════════════════════════════════

%%bash
echo "🚀 Entraînement final sur TRAIN + VALIDATION combinés..."

python data_baseline/train_final_full_dataset.py \
  --data_dir data_baseline/data \
  --loss $FINAL_LOSS \
  --epochs $FINAL_EPOCHS \
  --lr $FINAL_LR \
  --out_ckpt results/$FINAL_MODEL

echo ""
echo "✅ Entraînement terminé !"


# ═══════════════════════════════════════════════════════════════════════════════
# CELLULE 3 : Prédiction sur le Test Set avec le modèle final
# ═══════════════════════════════════════════════════════════════════════════════

%%bash
echo "🔮 Prédiction sur le TEST SET avec le modèle final..."

# Utilise ton script de retrieval existant
python data_baseline/retrieval_answer_new.py \
  --code train_final_full_dataset \
  --model $FINAL_MODEL \
  --data_dir data_baseline/data \
  --results_dir results

echo ""
echo "✅ Prédictions sauvegardées !"


# ═══════════════════════════════════════════════════════════════════════════════
# CELLULE 4 : Copie vers Drive et préparation soumission Kaggle
# ═══════════════════════════════════════════════════════════════════════════════

import shutil
from pathlib import Path

# Chemins
RESULTS_DIR = Path("results")
DRIVE_PATH = Path("/content/drive/MyDrive/Kaggle_ALTEGRAD/submissions")
SUBMISSION_FILE = "data_baseline/data/test_retrieved_descriptions.csv"

# Créer le dossier si nécessaire
DRIVE_PATH.mkdir(parents=True, exist_ok=True)

# Copier le fichier de soumission
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
submission_name = f"submission_{FINAL_LOSS}_{timestamp}.csv"

shutil.copy(SUBMISSION_FILE, DRIVE_PATH / submission_name)
print(f"✅ Soumission copiée vers: {DRIVE_PATH / submission_name}")

# Copier le modèle aussi
model_backup = f"model_final_{FINAL_LOSS}_{timestamp}.pt"
shutil.copy(RESULTS_DIR / FINAL_MODEL, DRIVE_PATH / model_backup)
print(f"✅ Modèle copié vers: {DRIVE_PATH / model_backup}")

print("\n" + "="*50)
print("📤 Prêt pour soumission Kaggle !")
print(f"Fichier à soumettre : {submission_name}")
print("="*50)


# ═══════════════════════════════════════════════════════════════════════════════
# CELLULE BONUS : Comparaison des résultats avant/après train+val
# ═══════════════════════════════════════════════════════════════════════════════

"""
## Notes importantes

### Pourquoi entraîner sur train + validation ?
- Plus de données = meilleur modèle
- Pour Kaggle, on utilise TOUTES les données d'entraînement disponibles
- Le validation set n'est plus nécessaire pour l'early stopping une fois 
  qu'on connaît le bon nombre d'epochs

### Risques
- Overfitting possible si trop d'epochs
- Utilise le nombre d'epochs qui donnait les meilleurs résultats sur validation

### Méthode recommandée
1. D'abord, trouve le meilleur nombre d'epochs sur train seul (avec early stopping)
2. Ensuite, entraîne sur train+val avec ce nombre d'epochs fixe
3. Soumets sur Kaggle et compare le score
"""

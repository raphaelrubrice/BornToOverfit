# ============================================================
# BENCHMARK DUAL ENCODER - GOOGLE COLAB
# Copie-colle dans une cellule Colab
# ============================================================

import subprocess
from pathlib import Path

SCRIPT_PATH = 'train_dual_encoder.py'
LOG_DIR = Path('dual_encoder_logs')
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# CONFIGURATION RECOMMANDÉE (Master MVA)
# ============================================================

# === PHASE 1 : Baseline Dual Encoder ===
baseline_config = {
    'loss': 'infonce',
    'temperature': 0.07,
    'batch_size': 16,
    'epochs': 30,
    'lr_graph': 5e-4,
    'lr_text': 2e-5,
    'hidden_dim': 256,
    'num_layers': 4,
}

# === PHASE 2 : Temperature Sweep (InfoNCE) ===
temperature_sweep = [
    {'loss': 'infonce', 'temperature': 0.05, 'batch_size': 16},
    {'loss': 'infonce', 'temperature': 0.07, 'batch_size': 16},
    {'loss': 'infonce', 'temperature': 0.1, 'batch_size': 16},
]

# === PHASE 3 : LR Text Sweep ===
lr_text_sweep = [
    {'loss': 'infonce', 'temperature': 0.07, 'lr_text': 1e-5},
    {'loss': 'infonce', 'temperature': 0.07, 'lr_text': 2e-5},
    {'loss': 'infonce', 'temperature': 0.07, 'lr_text': 5e-5},
]

# === PHASE 4 : Triplet Loss Comparison ===
triplet_configs = [
    {'loss': 'triplet', 'margin': 0.1, 'batch_size': 16},
    {'loss': 'triplet', 'margin': 0.2, 'batch_size': 16},
    {'loss': 'triplet', 'margin': 0.3, 'batch_size': 16},
]

# ============================================================
# CHOIX DE LA PHASE À EXÉCUTER
# ============================================================
# Décommente UNE SEULE ligne ci-dessous

experiments = [baseline_config]           # Phase 1 : Baseline
# experiments = temperature_sweep         # Phase 2 : Temperature
# experiments = lr_text_sweep             # Phase 3 : LR Text
# experiments = triplet_configs           # Phase 4 : Triplet

# ============================================================
# FONCTION D'EXÉCUTION
# ============================================================
def run_experiment(config):
    """Lance une expérience avec logging."""
    
    # Construire la commande
    cmd = ['python', SCRIPT_PATH]
    
    for key, value in config.items():
        cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # Nom du log
    config_str = "_".join([f"{k}{v}" for k, v in config.items()])
    log_file = LOG_DIR / f"dual_{config_str}.log"
    
    print(f"\n{'='*70}")
    print(f"🚀 Config: {config}")
    print(f"📝 Log: {log_file}")
    print(f"{'='*70}\n")
    
    # Exécuter
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    success = process.returncode == 0
    print(f"\n{'✅' if success else '❌'} {'Succès' if success else 'Erreur'}\n")
    
    return success

# ============================================================
# LANCER LES EXPÉRIENCES
# ============================================================
results = []

for i, config in enumerate(experiments, 1):
    print(f"\n{'#'*70}")
    print(f"# EXPÉRIENCE {i}/{len(experiments)}")
    print(f"{'#'*70}")
    
    success = run_experiment(config)
    results.append({'config': config, 'success': success})

# ============================================================
# RÉSUMÉ
# ============================================================
print(f"\n{'='*70}")
print("RÉSUMÉ DU BENCHMARK")
print(f"{'='*70}")
print(f"Total: {len(experiments)} expériences")
print(f"Succès: {sum(r['success'] for r in results)}")
print(f"Échecs: {sum(not r['success'] for r in results)}")
print(f"\nLogs: {LOG_DIR}/")
print(f"{'='*70}\n")

# ============================================================
# ANALYSE RAPIDE DES RÉSULTATS
# ============================================================
print("Meilleurs MRR par expérience :")
print("-" * 70)

import re

for result in results:
    config = result['config']
    config_str = "_".join([f"{k}={v}" for k, v in list(config.items())[:3]])
    log_file = LOG_DIR / f"dual_{config_str}.log"
    
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extraire le meilleur MRR
        mrr_matches = re.findall(r"'MRR': ([\d.]+)", content)
        if mrr_matches:
            best_mrr = max(float(m) for m in mrr_matches)
            print(f"{config_str:50s} | MRR: {best_mrr:.4f}")
        else:
            print(f"{config_str:50s} | MRR: N/A")
    else:
        print(f"{config_str:50s} | Log non trouvé")

print("-" * 70)

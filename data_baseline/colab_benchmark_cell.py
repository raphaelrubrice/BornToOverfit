# ============================================================
# BENCHMARK CELL POUR GOOGLE COLAB
# Copie-colle ce code dans une cellule Colab
# ============================================================

import subprocess
from datetime import datetime
from pathlib import Path

# Configuration du benchmark
SCRIPT_PATH = 'data_baseline/train_gcn_v3_gps_PT_args.py'
LOG_DIR = Path('benchmark_logs')
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# PHASE 1 : Baseline (décommente si pas encore fait)
# ============================================================
# experiments = [
#     {'loss': 'mse'},
#     {'loss': 'infonce', 'temperature': 0.07},
#     {'loss': 'triplet', 'margin': 0.2},
# ]

# ============================================================
# PHASE 2A : InfoNCE Temperature Sweep
# ============================================================
experiments = [
    {'loss': 'infonce', 'temperature': 0.05},
    {'loss': 'infonce', 'temperature': 0.1},
    {'loss': 'infonce', 'temperature': 0.15},
]

# ============================================================
# PHASE 2B : Triplet Margin Sweep (décommente si Triplet gagne)
# ============================================================
# experiments = [
#     {'loss': 'triplet', 'margin': 0.1},
#     {'loss': 'triplet', 'margin': 0.3},
#     {'loss': 'triplet', 'margin': 0.5},
# ]

# ============================================================
# PHASE 3 : Batch Size (remplace par ta meilleure config)
# ============================================================
# experiments = [
#     {'loss': 'infonce', 'temperature': 0.07, 'batch_size': 16},
#     {'loss': 'infonce', 'temperature': 0.07, 'batch_size': 32},
#     {'loss': 'infonce', 'temperature': 0.07, 'batch_size': 64},
#     {'loss': 'infonce', 'temperature': 0.07, 'batch_size': 128},
# ]

# ============================================================
# EXÉCUTION
# ============================================================
def run_experiment(config):
    """Lance une expérience et sauvegarde les logs."""
    
    # Construire la commande
    cmd = ['python', SCRIPT_PATH]
    
    for key, value in config.items():
        cmd.extend([f'--{key.replace("_", "-")}', str(value)])
    
    # Nom du fichier de log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = "_".join([f"{k}{v}" for k, v in config.items()])
    log_file = LOG_DIR / f"exp_{config_str}_{timestamp}.log"
    
    print(f"\n{'='*70}")
    print(f"🚀 Lancement : {config}")
    print(f"📝 Log file  : {log_file}")
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
        
        # Afficher en temps réel ET sauvegarder
        for line in process.stdout:
            print(line, end='')
            f.write(line)
        
        process.wait()
    
    if process.returncode == 0:
        print(f"\n✅ Succès : {config}\n")
    else:
        print(f"\n❌ Erreur : {config} (code {process.returncode})\n")
    
    return process.returncode == 0


# Lancer toutes les expériences
results = []
for i, exp_config in enumerate(experiments, 1):
    print(f"\n{'#'*70}")
    print(f"# EXPÉRIENCE {i}/{len(experiments)}")
    print(f"{'#'*70}")
    
    success = run_experiment(exp_config)
    results.append({'config': exp_config, 'success': success})

# Résumé
print(f"\n{'='*70}")
print(f"RÉSUMÉ DU BENCHMARK")
print(f"{'='*70}")
print(f"Total : {len(experiments)} expériences")
print(f"Succès : {sum(r['success'] for r in results)}")
print(f"Échecs : {sum(not r['success'] for r in results)}")
print(f"\nLogs sauvegardés dans : {LOG_DIR}/")
print(f"{'='*70}\n")

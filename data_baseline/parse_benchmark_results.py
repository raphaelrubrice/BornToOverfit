"""
parse_benchmark_results.py - Analyse automatique des logs
"""

import re
import pandas as pd
from pathlib import Path

def parse_log(log_file):
    """Extrait les métriques d'un fichier de log."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    results = {
        'file': log_file.name,
        'best_mrr': 0.0,
        'best_r1': 0.0,
        'best_r5': 0.0,
        'best_r10': 0.0,
    }
    
    # Regex pour extraire les métriques
    # Format attendu: Epoch X/Y | loss=Z | {'MRR': A, 'R@1': B, ...}
    pattern = r"MRR': ([\d.]+).*?R@1': ([\d.]+).*?R@5': ([\d.]+).*?R@10': ([\d.]+)"
    
    matches = re.findall(pattern, content)
    if matches:
        # Prendre le meilleur MRR
        mrrs = [float(m[0]) for m in matches]
        best_idx = mrrs.index(max(mrrs))
        best_match = matches[best_idx]
        
        results['best_mrr'] = float(best_match[0])
        results['best_r1'] = float(best_match[1])
        results['best_r5'] = float(best_match[2])
        results['best_r10'] = float(best_match[3])
    
    # Extraire les hyperparamètres du nom de fichier
    filename = log_file.stem
    
    if 'infonce' in filename:
        results['loss'] = 'infonce'
        temp_match = re.search(r'temp([\d.]+)', filename)
        if temp_match:
            results['temperature'] = float(temp_match.group(1))
    elif 'triplet' in filename:
        results['loss'] = 'triplet'
        margin_match = re.search(r'margin([\d.]+)', filename)
        if margin_match:
            results['margin'] = float(margin_match.group(1))
    elif 'mse' in filename:
        results['loss'] = 'mse'
    
    # Batch size
    bs_match = re.search(r'bs(\d+)', filename)
    if bs_match:
        results['batch_size'] = int(bs_match.group(1))
    
    # Learning rate
    lr_match = re.search(r'lr([\de\-]+)', filename)
    if lr_match:
        results['lr'] = lr_match.group(1)
    
    return results


def analyze_benchmark(log_dir='benchmark_results'):
    """Analyse tous les logs et génère un rapport."""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"Erreur: {log_dir} n'existe pas")
        return
    
    all_results = []
    for log_file in sorted(log_dir.glob('*.log')):
        result = parse_log(log_file)
        all_results.append(result)
    
    if not all_results:
        print("Aucun fichier de log trouvé")
        return
    
    df = pd.DataFrame(all_results)
    
    # ========================================
    # RAPPORT PHASE 1 : Comparaison des losses
    # ========================================
    print("\n" + "="*60)
    print("PHASE 1 : COMPARAISON DES LOSSES")
    print("="*60)
    
    phase1 = df[df['file'].str.contains('phase1')]
    if not phase1.empty:
        phase1_sorted = phase1.sort_values('best_mrr', ascending=False)
        print("\nRésultats triés par MRR :")
        print(phase1_sorted[['loss', 'best_mrr', 'best_r1', 'best_r5', 'best_r10']].to_string(index=False))
        
        best_loss = phase1_sorted.iloc[0]['loss']
        print(f"\n🏆 MEILLEURE LOSS : {best_loss.upper()}")
        print(f"   MRR = {phase1_sorted.iloc[0]['best_mrr']:.4f}")
    
    # ========================================
    # RAPPORT PHASE 2 : Hyperparamètres
    # ========================================
    print("\n" + "="*60)
    print("PHASE 2 : TUNING DES HYPERPARAMÈTRES")
    print("="*60)
    
    phase2 = df[df['file'].str.contains('phase2')]
    if not phase2.empty:
        if 'temperature' in phase2.columns and phase2['temperature'].notna().any():
            print("\n📊 InfoNCE - Température :")
            temp_df = phase2[phase2['temperature'].notna()].sort_values('best_mrr', ascending=False)
            print(temp_df[['temperature', 'best_mrr', 'best_r1']].to_string(index=False))
            print(f"\n   Meilleure température : {temp_df.iloc[0]['temperature']}")
        
        if 'margin' in phase2.columns and phase2['margin'].notna().any():
            print("\n📊 Triplet - Marge :")
            margin_df = phase2[phase2['margin'].notna()].sort_values('best_mrr', ascending=False)
            print(margin_df[['margin', 'best_mrr', 'best_r1']].to_string(index=False))
            print(f"\n   Meilleure marge : {margin_df.iloc[0]['margin']}")
    
    # ========================================
    # RAPPORT PHASE 3 : Batch size
    # ========================================
    phase3 = df[df['file'].str.contains('phase3')]
    if not phase3.empty:
        print("\n" + "="*60)
        print("PHASE 3 : BATCH SIZE")
        print("="*60)
        bs_df = phase3.sort_values('best_mrr', ascending=False)
        print(bs_df[['batch_size', 'best_mrr', 'best_r1']].to_string(index=False))
        print(f"\n   Meilleur batch size : {bs_df.iloc[0]['batch_size']}")
    
    # ========================================
    # RAPPORT PHASE 4 : Learning rate
    # ========================================
    phase4 = df[df['file'].str.contains('phase4')]
    if not phase4.empty:
        print("\n" + "="*60)
        print("PHASE 4 : LEARNING RATE")
        print("="*60)
        lr_df = phase4.sort_values('best_mrr', ascending=False)
        print(lr_df[['lr', 'best_mrr', 'best_r1']].to_string(index=False))
        print(f"\n   Meilleur learning rate : {lr_df.iloc[0]['lr']}")
    
    # ========================================
    # CONFIGURATION FINALE RECOMMANDÉE
    # ========================================
    print("\n" + "="*60)
    print("🎯 CONFIGURATION FINALE RECOMMANDÉE")
    print("="*60)
    
    best_overall = df.loc[df['best_mrr'].idxmax()]
    print(f"\nLoss : {best_overall.get('loss', 'N/A')}")
    if 'temperature' in best_overall and pd.notna(best_overall['temperature']):
        print(f"Temperature : {best_overall['temperature']}")
    if 'margin' in best_overall and pd.notna(best_overall['margin']):
        print(f"Margin : {best_overall['margin']}")
    if 'batch_size' in best_overall and pd.notna(best_overall['batch_size']):
        print(f"Batch size : {int(best_overall['batch_size'])}")
    if 'lr' in best_overall and pd.notna(best_overall['lr']):
        print(f"Learning rate : {best_overall['lr']}")
    
    print(f"\nPerformance :")
    print(f"  MRR  : {best_overall['best_mrr']:.4f}")
    print(f"  R@1  : {best_overall['best_r1']:.4f}")
    print(f"  R@5  : {best_overall['best_r5']:.4f}")
    print(f"  R@10 : {best_overall['best_r10']:.4f}")
    
    # Sauvegarder le tableau complet
    csv_path = Path(log_dir) / 'benchmark_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n📁 Résultats sauvegardés dans : {csv_path}")


if __name__ == '__main__':
    analyze_benchmark()

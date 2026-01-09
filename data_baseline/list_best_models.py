"""
list_best_models.py - Liste tous les checkpoints avec leurs hyperparamètres
"""

import re
from pathlib import Path
import pandas as pd

def parse_model_filename(filepath):
    """Extrait les hyperparamètres du nom de fichier."""
    filename = filepath.stem
    
    info = {'filename': filepath.name}
    
    # Loss type
    if 'infonce' in filename:
        info['loss'] = 'infonce'
    elif 'triplet' in filename:
        info['loss'] = 'triplet'
    elif 'mse' in filename:
        info['loss'] = 'mse'
    
    # Temperature
    temp_match = re.search(r'temp([\d.]+)', filename)
    if temp_match:
        info['temperature'] = float(temp_match.group(1))
    
    # Margin
    margin_match = re.search(r'margin([\d.]+)', filename)
    if margin_match:
        info['margin'] = float(margin_match.group(1))
    
    # Batch size
    bs_match = re.search(r'bs(\d+)', filename)
    if bs_match:
        info['batch_size'] = int(bs_match.group(1))
    
    # Learning rate
    lr_match = re.search(r'lr([\d.e\-]+)', filename)
    if lr_match:
        info['lr'] = float(lr_match.group(1))
    
    # Hidden dimension
    hd_match = re.search(r'hd(\d+)', filename)
    if hd_match:
        info['hidden_dim'] = int(hd_match.group(1))
    
    # Number of layers
    nl_match = re.search(r'nl(\d+)', filename)
    if nl_match:
        info['num_layers'] = int(nl_match.group(1))
    
    # Taille du fichier
    info['size_mb'] = filepath.stat().st_size / (1024 * 1024)
    
    return info


def list_models(data_dir='data'):
    """Liste tous les modèles sauvegardés."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Erreur : {data_path} n'existe pas")
        return
    
    # Trouver tous les fichiers .pt
    model_files = list(data_path.glob('model_v3_gps_*.pt'))
    
    if not model_files:
        print(f"Aucun modèle trouvé dans {data_path}")
        return
    
    print(f"\n📦 Trouvé {len(model_files)} modèle(s) sauvegardé(s)\n")
    
    # Parser tous les modèles
    models_info = [parse_model_filename(f) for f in model_files]
    df = pd.DataFrame(models_info)
    
    # Trier par loss puis par hyperparamètres
    sort_cols = ['loss']
    if 'temperature' in df.columns:
        sort_cols.append('temperature')
    if 'margin' in df.columns:
        sort_cols.append('margin')
    if 'batch_size' in df.columns:
        sort_cols.append('batch_size')
    
    df_sorted = df.sort_values(sort_cols)
    
    # Afficher par loss type
    print("="*80)
    print("MODÈLES SAUVEGARDÉS PAR LOSS TYPE")
    print("="*80)
    
    for loss_type in df_sorted['loss'].unique():
        print(f"\n📊 {loss_type.upper()}")
        print("-"*80)
        
        loss_df = df_sorted[df_sorted['loss'] == loss_type]
        
        # Colonnes à afficher
        display_cols = ['filename']
        if loss_type == 'infonce' and 'temperature' in loss_df.columns:
            display_cols.append('temperature')
        elif loss_type == 'triplet' and 'margin' in loss_df.columns:
            display_cols.append('margin')
        
        for col in ['batch_size', 'lr', 'hidden_dim', 'num_layers', 'size_mb']:
            if col in loss_df.columns:
                display_cols.append(col)
        
        print(loss_df[display_cols].to_string(index=False))
    
    # Résumé global
    print("\n" + "="*80)
    print("RÉSUMÉ")
    print("="*80)
    print(f"Total : {len(model_files)} modèles")
    print(f"Espace disque : {df['size_mb'].sum():.2f} MB")
    print(f"\nRépartition par loss :")
    print(df['loss'].value_counts().to_string())
    
    # Sauvegarder le tableau
    csv_path = data_path / 'models_inventory.csv'
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n📁 Inventaire sauvegardé dans : {csv_path}")


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    list_models(data_dir)

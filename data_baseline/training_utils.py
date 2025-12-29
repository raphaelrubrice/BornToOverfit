import copy

# =========================================================
# UTILS: Early Stopping
# =========================================================
class EarlyStopping(object):
    """
    Early stopping that supports maximization (MRR) and safe weight saving.
    """
    def __init__(self, patience, mode='max', tol=0.0):
        self.patience = patience
        self.mode = mode  # 'max' for MRR, 'min' for Loss
        self.tol = tol
        self.patience_count = 0
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.best_state_dict = None
        self.best_epoch_idx = 0
    
    def check_stop(self, model, current_score, epoch_idx):
        improvement = False
        
        if self.mode == 'max':
            if current_score > self.best_score + self.tol:
                improvement = True
        else: # mode == 'min'
            if current_score < self.best_score - self.tol:
                improvement = True
                
        if improvement:
            self.best_score = current_score
            self.best_epoch_idx = epoch_idx
            self.best_state_dict = copy.deepcopy(model.state_dict())
            self.patience_count = 0
            return False, True # stop_signal, is_best
        else:
            self.patience_count += 1
            stop_signal = self.patience_count >= self.patience
            return stop_signal, False

    def load_best_weights(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            print(f"Restored best model from epoch {self.best_epoch_idx+1}")
import torch.nn.functional as F

def hierarchical_loss(logits_bin, logits_multi, y_bin, y_multi,
                      weight_bin,
                      weight_multi,
                      a: float = 1.0,
                      b: float = 1.0):
    """
    - logits_bin: [B,2], logits para gen/no-gen
    - logits_multi: [B,3], logits para CDS/UTR/INTRON
    - y_bin:    [B] en {0=no-gen, 1=gen}
    - y_multi:  [B] en {0,1,2} o -1 (“ignore”) cuando no-gen
    """
    # 1) Pérdida binaria
    loss_bin = F.cross_entropy(logits_bin, y_bin, weight=weight_bin)
    
    # 2) Pérdida multiclasificación con ignore_index
    #    Para las muestras y_multi == -1 el loss se descarta.
    loss_multi = F.cross_entropy(logits_multi, y_multi, ignore_index=-1, weight=weight_multi)
    
    return a * loss_bin + b * loss_multi


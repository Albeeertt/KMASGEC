import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.metrics import multilabel_confusion_matrix as confusion_matrix
import os
import json

def calculate_accuracy(logits, labels):
    # Para CrossEntropyLoss: logits.shape = [batch_size, 2]
    preds = torch.argmax(logits, dim=1)       # índice de la clase con mayor score
    correct = (preds == labels).sum().item()  # número de aciertos en el batch
    return correct / labels.size(0)



def calculate_accuracy_with_reject(logits, labels, threshold=0.5):
    probs = F.softmax(logits, dim=1)
    max_probs, argmaxes = probs.max(dim=1)
    reject_label = logits.size(1)
    rejection_tensor = torch.full_like(argmaxes, reject_label)
    preds = torch.where(
        max_probs >= threshold,
        argmaxes,
        rejection_tensor
    )
    correct = (preds == labels) 
    acc = correct.sum().item() / labels.size(0)
    return acc, preds


def train(
    dataloader,
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    route_dir_info
    ):

    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    all_preds, all_labels = [], []

    criterion.to(device)
    for seqs, types, mask in dataloader:
        seqs   = seqs.to(device, non_blocking=True)
        labels = types.to(device, non_blocking=True)
        mask   = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(seqs, attention_mask=mask) 
        loss   = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        correct = (preds == labels).sum().item()

        bs = labels.size(0)
        running_correct += correct
        running_count   += bs
        running_loss    += loss.item() * bs

        loss.backward()
        optimizer.step()

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        dataloader.set_postfix({
                "loss": f"{running_loss / running_count:.4f}",
                "acc":  f"{running_correct / running_count:.4f}",
        })

    avg_loss = running_loss / max(running_count, 1)
    avg_acc  = running_correct / max(running_count, 1)
    print(f"[Train] Epoch {num_epoch} | loss={avg_loss:.4f} | acc={avg_acc:.4f}")

    report = classification_report(
        all_labels, all_preds, digits=4, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(
        all_labels,
        all_preds
    )   

    cm_list = cm.tolist()
    report["confusion_matrix"] = cm_list
    os.makedirs(os.path.dirname(route_dir_info), exist_ok=True)
    with open(route_dir_info, "a") as f:
        json.dump({"epoch": num_epoch, "train": report}, f, indent=2)

def iteration_train_oneHead(
    dataloader,
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    scheduler = None,
    route_dir_info = "./report.json"
):
    scaler = GradScaler()
    model.train()
    total_loss = 0.0
    total_acc   = 0.0
    total_count = 0
    criterion = criterion.to(device)

    all_preds = []
    all_labels = []
    for seqs, types, mask in dataloader:
        # types: [B]        → labels
        # seqs:  [B, L]     → input_ids (ya codificados)
        # mask:  [B, L]     → attention_mask
        labels = types.to(device)             # [B]
        input_ids = seqs.to(device)           # [B, L]
        attention_mask = mask.to(device)      # [B, L]

        batch_size = labels.size(0)

        optimizer.zero_grad(set_to_none=True)
        # Forward
        with autocast(device_type=device.type):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
        # outputs: [B, num_classes]
        acc    = calculate_accuracy(outputs, labels)
        
        total_acc   += acc * batch_size
        total_count += batch_size

        # Cálculo de pérdidas
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Backprop
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * batch_size

        dataloader.set_postfix({
        'loss': f'{total_loss/ total_count:.4f}',
        'acc' : f'{total_acc/total_count:.4f}',
        })


    avg_loss = total_loss / total_count
    epoch_acc = total_acc / total_count
    print(f"Epoch {num_epoch} Train Loss: {avg_loss:.4f}")
    print(f"Accuracy en train: {epoch_acc:.4f}")
    report_dict = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    os.makedirs(os.path.dirname(route_dir_info), exist_ok=True)
    with open(route_dir_info, "a") as f:
        json.dump(report_dict, f, indent=4)

def iteration_validation_oneHead(
    dataloader,      # ahora es un DataLoader, no el Dataset crudo
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    best_val_loss,
    route_dir_info = "./report.json"
):
    model.eval()
    total_val_loss = 0.0
    total_acc   = 0.0
    total_count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seqs, types, mask in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)
            criterion = criterion.to(device) 

            batch_size = labels.size(0)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            total_val_loss     += loss.item() * batch_size
            
            preds = logits.argmax(dim=1)

            correct = (preds == labels).sum().item()
            
            total_acc += correct
            # total_acc   += acc * batch_size
            total_count += batch_size


            # preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            dataloader.set_postfix({
                'loss_validation': f'{total_val_loss/ total_count:.4f}',
                'acc_validation' : f'{total_acc/total_count:.4f}',
            })

        avg_val_loss     = total_val_loss     / total_count
        epoch_acc = total_acc / total_count

        # Guarda el mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"../models/model_other_obj.pt"
            torch.save({
                "epoch": num_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, save_path)
            print(f"→ Nuevo mejor modelo guardado en {save_path}")

    print(f"Epoch {num_epoch} Val   Loss: {avg_val_loss:.4f}")
    print(f"Accuracy en validation: {epoch_acc:.4f}")
    print("--------------------------------------------------")
    report_dict = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    cm = confusion_matrix(
        all_labels,
        all_preds
    )
    cm_list = cm.tolist()
    report_dict["confusion_matrix"] = cm_list
    os.makedirs(os.path.dirname(route_dir_info), exist_ok=True)
    with open(route_dir_info, "a") as f:
        json.dump(report_dict, f, indent=4)
    return best_val_loss

def iteration_test_oneHead(
    dataloader,      # ahora es un DataLoader, no el Dataset crudo
    model,
    device,
    criterion,
    num_classes
):
    model.eval()
    total_val_loss = 0.0
    total_acc   = 0.0
    total_count = 0

    all_trues = []
    all_preds = []
    all_places = []
    all_places_new = []
    all_softmax_official_values = []

    criterion = criterion.to(device)

    with torch.no_grad():
        for seqs, types, mask, place, place_new in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)

            batch_size = labels.size(0)

            outputs, _ = model(input_ids, attention_mask=attention_mask)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()

            total_acc   += correct
            total_count += batch_size

            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_places.extend(list(place))
            all_places_new.extend(list(place_new))
            # all_softmax_official_values.extend(F.softmax(outputs.cpu(), dim=1).tolist())
            # TODO: Descomentar la siguiente línea y comentar la posterior. Lo normal es transmitir el valor de 
            # las probabilidades no la de los logits.
            # all_softmax_official_values.extend(probs.tolist())
            all_softmax_official_values.extend(outputs.tolist())

            loss     = criterion(outputs, labels)
            total_val_loss     += loss.item() * batch_size

            dataloader.set_postfix({
                'loss_test': f'{total_val_loss/ total_count:.4f}',
                'acc_test' : f'{total_acc/total_count:.4f}'
            })

        avg_val_loss     = total_val_loss     / total_count
        epoch_acc = total_acc / total_count

    print(f"Accuracy en test: {epoch_acc:.4f}")
    print("--------------------------------------------------")
    # report_dict = classification_report(all_trues, all_preds, digits=4, output_dict=True)
    report_dict = []
    # cm = confusion_matrix(
    #     all_trues,
    #     all_preds
    # )
    # cm_list = cm.tolist()
    # report_dict["confusion_matrix"] = cm_list

    print("Fin")
    return report_dict, all_trues, all_preds, all_places, all_places_new, all_softmax_official_values


def iteration_test_oneHead_div(
    dataloader,      # ahora es un DataLoader, no el Dataset crudo
    model,
    device,
    criterion,
    num_classes,
    max_len_seq: int
):
    model.eval()
    total_val_loss = 0.0
    total_acc   = 0.0
    total_count = 0

    all_trues = []
    all_preds = []
    all_places = []
    all_softmax_official_values = []

    criterion = criterion.to(device)

    with torch.no_grad():
        for seqs, types, mask, place in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)

            batch_size = labels.size(0)

            outputs = model(input_ids, attention_mask=attention_mask)
            # preds = outputs.argmax(dim=1)
            # correct = (preds == labels).sum().item()
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            correct = (preds == labels).all(dim=1).sum().item()

            total_acc   += correct
            total_count += batch_size

            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_places.extend(list(place))
            # all_softmax_official_values.extend(F.softmax(outputs.cpu(), dim=1).tolist())
            # TODO: Descomentar la siguiente línea y comentar la posterior. Lo normal es transmitir el valor de 
            # las probabilidades no la de los logits.
            # all_softmax_official_values.extend(probs.tolist())
            all_softmax_official_values.extend(outputs.tolist())

            loss     = criterion(outputs, labels)
            total_val_loss     += loss.item() * batch_size

            dataloader.set_postfix({
                'loss_test': f'{total_val_loss/ total_count:.4f}',
                'acc_test' : f'{total_acc/total_count:.4f}'
            })

        avg_val_loss     = total_val_loss     / total_count
        epoch_acc = total_acc / total_count

    print(f"Accuracy en test: {epoch_acc:.4f}")
    print("--------------------------------------------------")
    report_dict = classification_report(all_trues, all_preds, digits=4, output_dict=True)
    # cm = confusion_matrix(
    #     all_trues,
    #     all_preds
    # )
    # cm_list = cm.tolist()
    # report_dict["confusion_matrix"] = cm_list

    print("Fin")
    return report_dict, all_trues, all_preds, all_places, all_softmax_official_values


def test(
    dataloader,      # ahora es un DataLoader, no el Dataset crudo
    model,
    device,
    criterion,
    num_classes
):
    model.eval()
    total_val_loss = 0.0
    total_acc   = 0.0
    total_count = 0

    all_trues = []
    all_preds = []

    with torch.no_grad():
        for seqs, types, mask in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)
            criterion = criterion.to(device)

            batch_size = labels.size(0)

            outputs = model(input_ids, attention_mask=attention_mask)


def inspect_one_batch(
    dataloader,
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    scheduler = None,
    route_dir_info = "./report.json",
    cls_value: int = 20,
    padding_value: int = 21
):
    batch = next(iter(dataloader))
    a, b, c = batch
    candidates = [a, b, c]

    labels = next(t for t in candidates if t.dim()==1 and t.dtype in (torch.int64, torch.long))

    seqs   = next(t for t in candidates if t.dim()==2 and t.dtype in (torch.int64, torch.long))

    mask   = next(t for t in candidates if t.dim()==2 and t.shape==seqs.shape and t is not seqs)

    print("labels uniq:", torch.unique(labels, return_counts=True))
    print("seqs shape:", seqs.shape)
    print("mask dtype/shape:", mask.dtype, mask.shape)

    assert cls_value != padding_value, "cls_value y padding_value no pueden coincidir"
    assert (seqs[:,0] == cls_value).all(), "El primer token no es CLS"
    assert mask.dtype == torch.bool, "mask debe ser bool (True=válido)"
    assert seqs.size(1) <= model.pos_embed.num_embeddings, \
        f"seq_len={seqs.size(1)} > max_seq_len={model.pos_embed.num_embeddings}"

    seqs = seqs.to(device)
    labels = labels.to(device)
    mask = mask.to(device)
    
    assert seqs.size(1) <= model.pos_embed.num_embeddings, \
        f"seq_len={input_ids.size(1)} > max_seq_len={model.pos_embed.num_embeddings}"

    pad_ratio = (~mask).float().mean().item()
    print(f"PAD ratio in batch: {pad_ratio:.3f}")

    valid = mask.sum(dim=1).float()  # True=válido
    for c in [0,1,2]:
            m = valid[labels==c].mean().item() if (labels==c).any() else float('nan')
            print(f"class {c} mean valid tokens: {m:.1f}")

    with torch.no_grad():
        logits = model(seqs, attention_mask=mask)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        for c in [0,1,2]:
            m = logits[labels==c, c].mean().item() if (labels==c).any() else float('nan')
            print(f"mean true-class logit (class {c}): {m:.3f}")

    print("logits shape:", logits.shape)
    print("preds uniq:", torch.unique(preds, return_counts=True))
    print("mean probs:", probs.mean(dim=0))
    print("OK: batch inspeccionado sin tocar tu training loop.")



def jesus(
    dataloader,
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    scheduler = None,
    route_dir_info = "./report.json",
    cls_value: int = 20,
    padding_value: int = 21
):

    seqs, labels, mask = next(iter(dataloader))
    seqs   = seqs.to(device).long()
    labels = labels.to(device).long()
    mask   = mask.to(device).bool()

    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    crit = torch.nn.CrossEntropyLoss()

    loss_hist = []
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        logits = model(seqs, attention_mask=mask)
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        loss_hist.append(float(loss.item()))
        if (step+1) % 20 == 0:
            print(f"step {step+1:3d}  loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(seqs, attention_mask=mask).argmax(1)
    print("loss steps:", [round(x,4) for x in loss_hist[:5]], "...", round(loss_hist[-1],4))
    print("preds uniq:", torch.unique(preds, return_counts=True))

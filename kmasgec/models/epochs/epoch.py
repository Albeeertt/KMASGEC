import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix

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

def iteration_train(
    dataloader,
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    weights, 
    weights2
):
    scaler = GradScaler()
    model.train()
    total_loss = 0.0
    total_acc   = 0.0
    total_acc2   = 0.0
    total_count = 0
    for types, types2, seqs, mask in dataloader:
        # types: [B]        → labels
        # seqs:  [B, L]     → input_ids (ya codificados)
        # mask:  [B, L]     → attention_mask
        labels = types.to(device)             # [B]
        labels2 = types2.to(device)
        input_ids = seqs.to(device)           # [B, L]
        attention_mask = mask.to(device)      # [B, L]
        weights   = weights.to(device)
        weights2 = weights2.to(device)

        batch_size = labels.size(0)

        optimizer.zero_grad(set_to_none=True)
        # Forward
        with autocast(device_type=device.type):
            outputs, outputs2 = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs2, outputs, labels2, labels, weights2, weights)
        # outputs: [B, num_classes]
        acc    = calculate_accuracy(outputs, labels)
        acc2 = calculate_accuracy(outputs2, labels2)

        total_acc   += acc * batch_size
        total_acc2   += acc2 * batch_size
        total_count += batch_size

        # Cálculo de pérdidas

        # Backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        
        dataloader.set_postfix({
        'loss': f'{total_loss/ total_count:.4f}',
        'acc' : f'{total_acc/total_count:.4f}',
        'acc_binary': f'{total_acc2/total_count:.4f}'
        })

    avg_loss = total_loss / total_count
    epoch_acc = total_acc / total_count
    epoch_acc2 = total_acc2 / total_count
    print(f"Epoch {num_epoch} Train Loss: {avg_loss:.4f}")
    print(f"Accuracy en train: {epoch_acc:.4f}")
    print(f"Accuracy en train salida 2: {epoch_acc2:.4f}")


def iteration_train_oneHead(
    dataloader,
    num_epoch,
    model,
    device,
    criterion,
    optimizer
):
    scaler = GradScaler()
    model.train()
    total_loss = 0.0
    total_acc   = 0.0
    total_count = 0
    for types, seqs, mask in dataloader:
        # types: [B]        → labels
        # seqs:  [B, L]     → input_ids (ya codificados)
        # mask:  [B, L]     → attention_mask
        labels = types.to(device)             # [B]
        input_ids = seqs.to(device)           # [B, L]
        attention_mask = mask.to(device)      # [B, L]
        criterion = criterion.to(device) 

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

        # Backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size

        dataloader.set_postfix({
        'loss': f'{total_loss/ total_count:.4f}',
        'acc' : f'{total_acc/total_count:.4f}',
        })

    avg_loss = total_loss / total_count
    epoch_acc = total_acc / total_count
    print(f"Epoch {num_epoch} Train Loss: {avg_loss:.4f}")
    print(f"Accuracy en train: {epoch_acc:.4f}")

def iteration_validation(
    dataloader,      # ahora es un DataLoader, no el Dataset crudo
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    best_val_loss,
    weights,
    weights2
):
    model.eval()
    total_val_loss = 0.0
    total_acc   = 0.0
    total_acc2   = 0.0
    total_count = 0

    with torch.no_grad():
        for types, types2, seqs, mask in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            labels2 = types2.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)
            weights   = weights.to(device)
            weights2 = weights2.to(device)

            batch_size = labels.size(0)

            outputs, outputs2 = model(input_ids, attention_mask=attention_mask)
            acc    = calculate_accuracy(outputs, labels)
            acc2 = calculate_accuracy(outputs2, labels2)

            total_acc   += acc * batch_size
            total_acc2   += acc2 * batch_size
            total_count += batch_size

            loss = criterion(outputs2, outputs, labels2, labels, weights2, weights)
            total_val_loss     += loss.item() * batch_size

            dataloader.set_postfix({
                'loss_validation': f'{total_val_loss/ total_count:.4f}',
                'acc_validation' : f'{total_acc/total_count:.4f}',
                'acc_binary_validation': f'{total_acc2/total_count:.4f}'
            })

        avg_val_loss     = total_val_loss     / total_count
        epoch_acc = total_acc / total_count
        epoch_acc2 = total_acc2 / total_count

        # Guarda el mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"../../../models/model_other_obj.pt"
            torch.save({
                "epoch": num_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, save_path)
            print(f"→ Nuevo mejor modelo guardado en {save_path}")

    print(f"Epoch {num_epoch} Val   Loss: {avg_val_loss:.4f}")
    print(f"Accuracy en validation: {epoch_acc:.4f}")
    print(f"Accuracy en validation: {epoch_acc:.4f}")
    print("--------------------------------------------------")
    return best_val_loss

def iteration_validation_oneHead(
    dataloader,      # ahora es un DataLoader, no el Dataset crudo
    num_epoch,
    model,
    device,
    criterion,
    optimizer,
    best_val_loss
):
    model.eval()
    total_val_loss = 0.0
    total_acc   = 0.0
    total_count = 0

    with torch.no_grad():
        for types, seqs, mask in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)
            criterion = criterion.to(device) 

            batch_size = labels.size(0)

            outputs = model(input_ids, attention_mask=attention_mask)
            acc    = calculate_accuracy(outputs, labels)

            total_acc   += acc * batch_size
            total_count += batch_size

            loss = criterion(outputs, labels)
            total_val_loss     += loss.item() * batch_size

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

    with torch.no_grad():
        for types, seqs, mask in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)
            criterion = criterion.to(device)

            batch_size = labels.size(0)

            outputs, attn = model(input_ids, attention_mask=attention_mask)
            acc    = calculate_accuracy(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_acc   += acc * batch_size
            total_count += batch_size

            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

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

    cm = confusion_matrix(all_trues, all_preds, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("Accuracy por clase:")
    for cls_idx, acc in enumerate(per_class_acc):
        print(f"  Clase {cls_idx}: {acc:.4f}")
    print("--------------------------------------------------")
    return cm, attn




def iteration_test_oneHead_w_reject(
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
        for types, seqs, mask in dataloader:
            # types: [B], seqs: [B, L], mask: [B, L]
            labels = types.to(device)
            input_ids = seqs.to(device)
            attention_mask = mask.to(device)
            criterion = criterion.to(device)

            batch_size = labels.size(0)

            outputs = model(input_ids, attention_mask=attention_mask)
            acc, preds    = calculate_accuracy_with_reject(outputs, labels, 0.5)

            total_acc   += acc * batch_size
            total_count += batch_size

            all_trues.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

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

    cm = confusion_matrix(
        all_trues,
        all_preds,
        labels=list(range(num_classes)) + [num_classes]
    )
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("Accuracy por clase:")
    for cls_idx, acc in enumerate(per_class_acc):
        name = cls_idx if cls_idx < num_classes else 'rechazo'
        print(f"  Clase {name}: {acc:.4f}")
    print("--------------------------------------------------")
    return cm
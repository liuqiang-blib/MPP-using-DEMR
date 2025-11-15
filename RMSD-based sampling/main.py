import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from dataloader import MoleculeDataloader
from CNNTransformer import CNNTransformerModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    recall_score, precision_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss
)
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
    print("GPU:", torch.cuda.device_count())
    print("device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Running on CPU.")



class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    with torch.no_grad():
        for x_batch, y_batch,rmsd_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.long().to(device)
            rmsd_batch = rmsd_batch.to(device)
            logits = model(x_batch,rmsd_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y_batch).sum().item()
            running_total += y_batch.size(0)

        avg_loss = running_loss / running_total
        accuracy = running_correct / running_total
    return avg_loss, accuracy



def train_model(model, train_loader, val_loader, epochs, optimizer, criterion, save_path):

    early_stopping = EarlyStopping(patience=5)
    best_val_loss = 1e-4

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    for epoch in range(epochs): 
        model.train()

        for x_batch, y_batch,rmsd_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            x_batch = x_batch.to(device)
            y_batch =  y_batch.long().to(device)
            rmsd_batch = rmsd_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch,rmsd_batch)

            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, val_loader, criterion)
        val_loss,val_acc = evaluate(model, val_loader, criterion)


        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}  | Train acc : {train_acc:.4f} | Val acc: {val_acc:.4f}")

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict(),
                       save_path)
            print(f"  ➜ Save best model (val_loss={best_val_loss:.4f}) -> {save_path}")
            break

        # Early Stopping 判断
        early_stopping(val_loss)
        if epoch > 10 and early_stopping.early_stop:
            print(f"EarlyStopping triggered at epoch {epoch}.")
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict(),
                       save_path)
            print(f"  ➜ Save best model (val_loss={val_loss:.4f}) -> {save_path}")
            break
        elif (epoch+1) == epochs:
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel)
                       else model.state_dict(),
                       save_path)
            print(f"  ➜ Save best model (val_loss={best_val_loss:.4f}) -> {save_path}")

    return history


def visualize_attention(model, input_tensor, label=None):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.unsqueeze(0).to(device)  # (1, T, 3, N, N)
        _ = model(input_tensor)
        attn = model.module.temporal.get_attention() if isinstance(model, nn.DataParallel) else model.temporal.get_attention()
        if attn is None:
            print("No Attention")
            return
        attn = attn.squeeze(0).cpu().numpy()  # (T, T)

    plt.figure(figsize=(10, 6))
    plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.xlabel("Time Step")
    plt.ylabel("Time Step")
    plt.title(f"Self-Attention Matrix" + (f" | Label: {label}" if label is not None else ""))
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def task_evaluate_metrics(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch,rmsd_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.long().to(device)
            rmsd_batch = rmsd_batch.to(device)
            logits = model(x_batch,rmsd_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]  

            running_correct += (preds == y_batch).sum().item()
            running_total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = running_loss / running_total
    accuracy = running_correct / running_total


    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = np.array(all_preds)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "Loss": avg_loss,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc_score(y_true, y_probs),
        "PR-AUC": average_precision_score(y_true, y_probs),
        "F1 Score": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity" : tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Balanced_acc" : balanced_accuracy_score(y_true, y_pred),
        "mcc" : matthews_corrcoef(y_true, y_pred),
        "brier" :  brier_score_loss(y_true, y_probs),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
        'time':[]
    }

    return metrics





if __name__ == "__main__":
    task_list = ['NR_AR', 'NR_AR_LBD', 'NR_AhR', 'NR_Aromatase', 'NR_ER', 'NR_ER_LBD', 'NR_PPAR_gamma', 'SR_ARE',
                 'SR_ATAD5', 'SR_HSE', 'SR_mmp', 'SR_p53', 'ADA17', 'EGFR', 'HIVPR']

    freq_list = [0]
    mode_list = ["low", "mid", "high"]
    frame_count_list = [2, 5, 10, 20, 50, 100, 1000]
    for task in task_list:
        for freq in freq_list:
            for mode in mode_list:
                for frame_count in frame_count_list:

                    df = pd.read_csv("/data/features_sampling_freq_" + str(freq) +  "/" + task + "_molecule_metadata.csv")
                    features = np.load("/data/features_sampling_freq_" + str(freq)  +  "/"  + task + "_molecule_features.npy", mmap_mode='r')
                    labels = df['label'].values
                    mol_id_list = df['mol_id'].values
                    rmsd_dir = "/data/MD_analysis"
                    print(f" Current dataset {task}， Class ratio{df['label'].value_counts()}")
                    data_loader = MoleculeDataloader(features, labels, mol_id_list=mol_id_list, rmsd_dir=rmsd_dir, mode=mode, frame_count=frame_count, batch_size=16, num_workers=4)
                    del features
                    gc.collect()
                    train_data = data_loader.get_train()
                    val_data = data_loader.get_val()
                    test_data = data_loader.get_test()
                    del data_loader
                    gc.collect()



                    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
                    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

                    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


                    model = CNNTransformerModel(num_classes=2, max_len=frame_count )
                    if torch.cuda.device_count() > 1:
                        print(f"Using {torch.cuda.device_count()}  GPU training！")
                        print(f"Using {torch.cuda.device_count()} GPUs")
                        model = torch.nn.DataParallel(model)
                    model = model.to(device)


                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


                    path  = "/data/RMSD_select_weighting/TOX21/results_CNNTransformer/" + mode
                    os.makedirs(path,exist_ok=True)
                    path = path + str(frame_count)
                    save_path =  path + task +   "_best_model.pt"
                    start_time = time.time()
                    train_history = train_model(model, train_data, val_data, 100, optimizer=optimizer, criterion=criterion, save_path=save_path)
                    end_time = time.time()
                    del train_data, val_data
                    gc.collect()


                    df = pd.DataFrame(train_history)
                    if os.path.exists( path  + task + "train_history.csv"):
                        df.to_csv( path + task  + "_train_history.csv", mode='a', index=False, header=False)
                    else:
                        df.to_csv( path + task + "_train_history.csv", mode='w', index=False, header=True)


                    best_state = torch.load(save_path, map_location=device)
                    if isinstance(model, nn.DataParallel):
                        model.module.load_state_dict(best_state)
                    else:
                        model.load_state_dict(best_state)

                    test_loss,test_acc = evaluate(model, test_data, criterion)
                    print(f"\n=== Test Loss: {test_loss:.4f} === | Test acc: {test_acc:.4f} ===")

                    results_metrics = task_evaluate_metrics(model, test_data, criterion, device)
                    results_metrics['time'].append(end_time - start_time)
                    print(results_metrics)
                    with open( path + task + "_test_results.txt", "w") as f:
                        for key, value in results_metrics .items():
                            f.write(f"{key}: {value}\n")

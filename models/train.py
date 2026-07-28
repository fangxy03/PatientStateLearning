import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from dataset import load_patient_data, PatientDataset
from patient_encoder import PatientEncoder



# ==========================
# config
# ==========================

DATA_PATH = (
    "data/processed/"
    "patient_observation.csv"
)


BATCH_SIZE = 512

EPOCHS = 20

LR = 1e-3


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ==========================
# load data
# ==========================

print("Loading data...")

X, y = load_patient_data(
    DATA_PATH
)


print(
    "Feature dimension:",
    X.shape[1]
)


print(
    "Samples:",
    X.shape[0]
)



# ==========================
# split
# ==========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)



train_dataset = PatientDataset(
    X_train,
    y_train
)

val_dataset = PatientDataset(
    X_val,
    y_val
)

test_dataset = PatientDataset(
    X_test,
    y_test
)



train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)



# ==========================
# model
# ==========================

model = PatientEncoder(
    input_dim=X.shape[1]
)


model = model.to(
    DEVICE
)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)



# ==========================
# training
# ==========================

best_f1 = 0



for epoch in range(EPOCHS):


    model.train()

    total_loss = 0


    for X_batch, y_batch in train_loader:


        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )


        optimizer.zero_grad()


        pred, state = model(
            X_batch
        )


        loss = criterion(
            pred,
            y_batch
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()



    # validation

    model.eval()

    preds=[]
    labels=[]


    with torch.no_grad():

        for X_batch,y_batch in val_loader:


            X_batch=X_batch.to(
                DEVICE
            )


            output,_=model(
                X_batch
            )


            pred=torch.argmax(
                output,
                dim=1
            )


            preds.extend(
                pred.cpu().numpy()
            )

            labels.extend(
                y_batch.numpy()
            )



    acc=accuracy_score(
        labels,
        preds
    )


    f1=f1_score(
        labels,
        preds,
        average="macro"
    )



    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss:{total_loss:.4f} "
        f"Val Acc:{acc:.4f} "
        f"Val F1:{f1:.4f}"
    )


    if f1 > best_f1:

        best_f1=f1

        os.makedirs(
            "../results",
            exist_ok=True
        )


        torch.save(
            model.state_dict(),
            "../results/best_patient_encoder.pt"
        )



# ==========================
# test
# ==========================


model.load_state_dict(
    torch.load(
        "../results/best_patient_encoder.pt"
    )
)


model.eval()


preds=[]
labels=[]


with torch.no_grad():

    for X_batch,y_batch in test_loader:

        X_batch=X_batch.to(
            DEVICE
        )


        output,_=model(
            X_batch
        )


        pred=torch.argmax(
            output,
            dim=1
        )


        preds.extend(
            pred.cpu().numpy()
        )

        labels.extend(
            y_batch.numpy()
        )



print("====================")

print(
    "Test Accuracy:",
    accuracy_score(
        labels,
        preds
    )
)


print(
    "Test Macro-F1:",
    f1_score(
        labels,
        preds,
        average="macro"
    )
)

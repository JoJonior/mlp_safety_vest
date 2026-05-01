import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
import numpy as np
from preprocessamento import load_dataset, load_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, log_loss

import joblib
import time

import optuna


def plot_loss(model,out_dir="OUTPUT"):
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.title("Curva de Perda (Loss Curve)")
    plt.xlabel("Épocas (Iterações)")
    plt.ylabel("Custo (Loss)")
    plt.grid(True)
    plt.savefig(f"{out_dir}/LOSS.png")
    plt.close()
    #plt.show()

def make_dirs(dir_name: str):
    contador = 1
    while True:
        novo_nome = f"{dir_name}-{contador}"
        if not os.path.exists(novo_nome):
            os.makedirs(novo_nome)
            return novo_nome
        contador += 1


        
    
        

def train_model(dir_data="DATA_NPZ",val_meta=0.95, max_epochs_=50,patience_=10,model_name="model_v1"):
    output_dir = f"run/{model_name}"

    output_dir=make_dirs(output_dir)

    dir_pesos = f"{output_dir}/pesos"
    dir_plots = f"{output_dir}/plots"

    os.makedirs(dir_pesos,exist_ok=True)
    os.makedirs(dir_plots,exist_ok=True)


    print("Carregando dados...")
    dir=f"{dir_data}/train.npz"
    X_train, Y_train = load_dataset(dir)
    dir=f"{dir_data}/test.npz"
    X_test, Y_test   = load_dataset(dir)
    dir=f"{dir_data}/valid.npz"
    X_valid, Y_valid = load_dataset(dir)


    X_train_full, y_train_full = shuffle(X_train, Y_train, random_state=42)

    print(X_train_full.shape)
    #neuronios = (512, 256, 64) #(1024, 512, 128)#(256, 128)#(512, 128)#(256, 128, 64)#(512, 256, 128) #(256, 128, 64) #
    neuronios = (512, 256, 64)
    model_mpl = MLPClassifier(
        hidden_layer_sizes=neuronios,
        activation='relu',
        solver='adam',
        alpha=  4,        
        learning_rate_init= 6.0520899374051026e-05,#0.00015,#6.0520899374051026e-05,
        learning_rate='adaptive',
        max_iter=100,
        random_state=42,

    )

    model_mpl_best = None
    
    classes = np.unique(y_train_full)
    print(classes)
    max_epochs = max_epochs_
    
    # Lista para guardar o histórico 
    history = []

    print(f"{'Epoch':<5} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val Accuracy':<12} | {'Tempo (s)':<10}")
    print("-" * 60)
    best_acc = 0.0

    patience = patience_ 
    patience_counter = 0
    best_loss = float('inf')
    batch_size = 128

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        
        X_train_full, y_train_full = shuffle(X_train_full, y_train_full, random_state=epoch)
        

        for i in range(0, len(X_train_full), batch_size):
            X_batch = X_train_full[i : i + batch_size]
            y_batch = y_train_full[i : i + batch_size]
            
            model_mpl.partial_fit(X_batch, y_batch, classes=classes)
      

        epoch_time = time.time() - start_time
        
        # Coletar métricas
        train_loss = model_mpl.loss_
        val_acc = model_mpl.score(X_valid, Y_valid)

        y_val_probs = model_mpl.predict_proba(X_valid)
        v_loss = log_loss(Y_valid, y_val_probs)

        
        print(f"{epoch:<5} | {train_loss:<12.6f} | {v_loss:<12.4f} | {val_acc:<12.4f} | {epoch_time:>5.2f}")

        # Salvar as métricas no histórico
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': v_loss,
            'val_accuracy': val_acc,
            'time_seconds': round(epoch_time, 2)
        })

        if v_loss < best_loss:
            best_loss = v_loss
            best_acc = val_acc

            patience_counter = 0
            # Salva o arquivo no disco apenas quando o modelo melhora
            model_mpl_best = model_mpl
            joblib.dump(model_mpl, f'{dir_pesos}/best.pkl')
            print(f"--- Novo melhor modelo salvo! (Val Loss: {v_loss:.4f})  (Acc: {best_acc:.4f}) ---")
        else:
            patience_counter += 1 # Não melhorou, gasta um ponto de paciência
        
        # Salva também o 'último' modelo (caso queira retomar depois)
        joblib.dump(model_mpl, f'{dir_pesos}/last.pkl')

        df_temp = pd.DataFrame(history)
        df_temp.to_csv(f"{output_dir}/results_epi_treino.csv", index=False)

        # Early Stopping Manual (Exemplo: parar se bater X% na validação)
        if val_acc >= val_meta:
            print("\nMeta de acurácia atingida! Interrompendo treino...")
            break
        if patience_counter >= patience:
            print(f"\nEarly Stopping! O modelo não melhora há {patience} épocas.")
            break
    


    print("\nTreino finalizado! Histórico salvo em 'results_epi_treino.csv'.")

    #plot_loss_manual(history,output_dir=dir_plots)
    plot_history_manual(history,output_dir=dir_plots)

    
    y_pred = model_mpl_best.predict(X_test) # Use os dados de teste encodados (0, 1)

    labls = ["Vest", "No-Vest"] # COLOQUEI ASSIM, Já salvao como 0 e 1 no dataset n uso o encoded


    plot_confusion_matrices( Y_test,y_pred,labls,dir_plots)



    # 3. Relatório métrico
    print(classification_report(Y_test, y_pred, target_names=labls))



def plot_loss_manual(history_list,output_dir):
    """
    Agora recebe a lista de histórico que você criou no loop
    """
    plt.figure(figsize=(8, 5))
    # Extrai apenas os valores de loss da lista de dicionários
    losses = [h['train_loss'] for h in history_list]
    
    plt.plot(losses, label='Train Loss')
    plt.title("Curva de Perda (Manual Training)")
    plt.xlabel("Épocas")
    plt.ylabel("Custo (Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/LOSS.png")
    plt.close()

def plot_history_manual_old(history_list, output_dir):
    plt.figure(figsize=(12, 5))
    
    # --- SUBPLOT 1: LOSS (Treino) ---
    plt.subplot(1, 2, 1)
    losses = [h['train_loss'] for h in history_list]
    plt.plot(losses, label='Train Loss', color='blue', linewidth=2)
    plt.title("Curva de Perda (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Custo")
    plt.legend()
    plt.grid(True, linestyle='--')

    # --- SUBPLOT 2: ACCURACY (Validação) ---
    plt.subplot(1, 2, 2)
    val_accs = [h['val_accuracy'] for h in history_list]
    # Linha vermelha pontilhada como você sugeriu
    plt.plot(val_accs, label='Val Accuracy', color='red', linestyle='--', marker='o', markersize=4)
    plt.title("Acurácia de Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia (0 a 1)")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout() # Ajusta o espaçamento entre os gráficos
    plt.savefig(f"{output_dir}/METRICS.png")
    plt.close()

def plot_history_manual(history_list, output_dir):
    plt.figure(figsize=(14, 5))
    
    # --- SUBPLOT 1: LOSS (Treino e Validação) ---
    plt.subplot(1, 2, 1)
    train_losses = [h['train_loss'] for h in history_list]
    val_losses = [h['val_loss'] for h in history_list] # Precisa salvar no loop!
    
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='red', linestyle='--', linewidth=2)
    
    plt.title("Curva de Perda (Loss)")
    plt.xlabel("Épocas")
    plt.ylabel("Custo (Log-Loss)")
    plt.legend()
    plt.grid(True, linestyle='--')

    # --- SUBPLOT 2: ACCURACY (Validação) ---
    plt.subplot(1, 2, 2)
    val_accs = [h['val_accuracy'] for h in history_list]
    plt.plot(val_accs, label='Val Accuracy', color='green', marker='o', markersize=4)
    
    plt.title("Acurácia de Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia (0 a 1)")
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/METRICS.png")
    plt.close()

def plot_confusion_matrices(Y_test, y_pred, labls, dir_plots):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Matriz 1: Valores Absolutos (Quantidade)
    cmd = ConfusionMatrixDisplay.from_predictions(
        Y_test, 
        y_pred, 
        display_labels=labls, 
        cmap=plt.cm.Blues,
        ax=ax[0]
    )
    ax[0].set_title("Matriz de Confusão (Valores)")

    # Matriz 2: Valores Normalizados (Percentual %)
    cmd_norm = ConfusionMatrixDisplay.from_predictions(
        Y_test, 
        y_pred, 
        display_labels=labls, 
        cmap=plt.cm.Greens,
        normalize='true', # Aqui acontece a normalização
        ax=ax[1]
    )
    ax[1].set_title("Matriz de Confusão (Percentual %)")

    plt.tight_layout()
    plt.savefig(f"{dir_plots}/ConfusionMatrices_Comparison.png")
    plt.close()



def objective(trial):
    dir_data="DATA_NPZ"
    dir=f"{dir_data}/train.npz"
    X_train, Y_train = load_dataset(dir)
    dir=f"{dir_data}/test.npz"
    X_test, Y_test   = load_dataset(dir)
    dir=f"{dir_data}/valid.npz"
    X_valid, Y_valid = load_dataset(dir)
    # O Optuna sugere valores para testar
    n_camada1 = trial.suggest_int("n1", 64, 512)
    n_camada2 = trial.suggest_int("n2", 32, 256)
    batch_size_ = trial.suggest_int("batch_size", 4, 32)

    v_alpha = trial.suggest_float("alpha", 1e-5, 0.5, log=True)
    ler_rate = trial.suggest_float("learning_rate_init", 1e-5, 0.5, log=True)
    
    # Você cria o modelo com as sugestões
    model = MLPClassifier(
        hidden_layer_sizes=(n_camada1, n_camada2),
        alpha=v_alpha,
        learning_rate_init=ler_rate,
        batch_size=batch_size_,
        max_iter=5,             
        early_stopping=True,     
        activation='relu',        
        solver='adam',           
        validation_fraction=0.2, 
        verbose=True  
        
        
    )
    
    # Treina e retorna a acurácia (o Optuna quer maximizar isso)
    model.fit(X_train, Y_train)
    return model.score(X_valid, Y_valid)


if __name__ == "__main__":
    if False:
        # Cria o estudo e começa a "caçada" por 50 tentativas
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print(f"Melhores parâmetros: {study.best_params}")
        objective()
    if True:
       
        start = time.perf_counter()
        train_model(val_meta=0.95,max_epochs_=100,patience_=15,model_name="model_clasifier") 
        end = time.perf_counter()
        print(f"Elapsed: {end - start:.2f} seconds")


import cv2
import joblib
import pandas as pd
import numpy as np
import os
from scipy import ndimage
from skimage.feature import hog

import matplotlib.pyplot as plt
import seaborn as sns
from skimage.measure import block_reduce
from sklearn.neural_network import MLPClassifier

def plot_class_distribution(dir_datachunk, output_dir="plots"):
    """
    Lê as anotações e plota a contagem de cada classe em um gráfico de barras.
    """
    # 1. Carregar o CSV
    csv_path = os.path.join(dir_datachunk, "_annotations.csv")
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo {csv_path} não encontrado.")
        return

    df = pd.read_csv(csv_path)

    # 2. Configuração visual do gráfico
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Criar o gráfico de contagem
    ax = sns.countplot(data=df, x='class', palette='viridis')
    filename = os.path.basename(dir_datachunk.strip("/"))
    # 3. Adicionar os valores exatos em cima das barras
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=12, fontweight='bold')

    # 4. Títulos e rótulos
    plt.title(f'Distribuição de Classes - {filename}', fontsize=15)
    plt.xlabel('Classe', fontsize=12)
    plt.ylabel('Quantidade de Imagens', fontsize=12)
    
    # 5. Salvar o resultado
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"distribuicao_{filename}.png")
    plt.savefig(save_path)
    #plt.show()
    
    print(f"Gráfico de distribuição salvo em: {save_path}")

def conveter_dataset_old(dir_datachunk,output="DATA_NPZ"):
    """
    Lê o CSV de anotações, processa todas as imagens e salva os dados prontos.
    """
    df = pd.read_csv(f"{dir_datachunk}/_annotations.csv",sep=",", usecols=["filename","class"])
    X_data = [] # Entradas (pixels)
    y_data = [] # Saídas (classes)

    print(f"Processando {len(df)} imagens. Aguarde...")

    for index, row in df.iterrows():
        nome_arquivo = row["filename"]
        classe = row["class"]
        
        caminho_imagem = os.path.join(dir_datachunk, nome_arquivo)
        frame = cv2.imread(caminho_imagem)
        
        if frame is None:
            print(f"Aviso: Não foi possível ler a imagem {nome_arquivo}")
            continue
            
        input_vetor = image_to_input(frame)
        
        # Guarda na lista, # Salvar Classe com string dá errom e preicsa de encoder
        X_data.append(input_vetor)
        if (classe=="Safety Vest"):
            y_data.append(1)
        elif(classe=="NO-Safety Vest"):
            y_data.append(0)
    # np.array é mais rapido que list
    X_array = np.array(X_data)
    y_array = np.array(y_data)

    os.makedirs(output,exist_ok=True)
    file_name = os.path.basename(dir_datachunk.strip("/"))

    # Salvar em formato Numpy (.npz) em vez de CSV, NPZ é melhor para salva matriz matematica
    caminho_salvar = os.path.join(output, f"{file_name}.npz")
    np.savez_compressed(caminho_salvar, X=X_array, y=y_array)
    
    print(f"Processamento concluído! Dados salvos em: {caminho_salvar}")


def conveter_dataset(dir_datachunk, output="DATA_NPZ"):
    df = pd.read_csv(f"{dir_datachunk}/_annotations.csv", sep=",", usecols=["filename", "class"])
    X_data = []
    y_data = []

    print(f"Processando {len(df)} imagens com Augmentation para Coletes...")

    for index, row in df.iterrows():
        nome_arquivo = row["filename"]
        classe = row["class"]
        caminho_imagem = os.path.join(dir_datachunk, nome_arquivo)
        frame = cv2.imread(caminho_imagem)
        
        if frame is None: continue
            
        # 1. Processa a imagem original
        input_vetor = image_to_input(frame)
        X_data.append(input_vetor)
        
        label = 0
        if (classe=="Safety Vest"):
            label = 1
        elif(classe=="NO-Safety Vest"):
            label = 0
        y_data.append(label)

        # 2. DATA AUGMENTATION: Se for colete, vamos criar variantes
        if label == 0:
            # Flip Horizontal: Dobra os exemplos de fundo
            flip_h = cv2.flip(frame, 1)
            X_data.append(image_to_input(flip_h))
            y_data.append(0)
            
            # Brilho variado: Ajuda a não confundir reflexos com faixas refletivas
            bright = cv2.convertScaleAbs(frame, alpha=0.8, beta=-10) # Um pouco mais escuro
            X_data.append(image_to_input(bright))
            y_data.append(0)
            
            # Brilho extra: Ajuda a ignorar luzes estouradas no fundo
            bright_alt = cv2.convertScaleAbs(frame, alpha=1.3, beta=15) # Mais claro
            X_data.append(image_to_input(bright_alt))
            y_data.append(0)

    X_array = np.array(X_data)
    y_array = np.array(y_data)

    os.makedirs(output, exist_ok=True)
    file_name = os.path.basename(dir_datachunk.strip("/"))
    caminho_salvar = os.path.join(output, f"{file_name}.npz")
    np.savez_compressed(caminho_salvar, X=X_array, y=y_array)
    
    print(f"Concluído! Total de amostras geradas: {len(y_data)}")


def load_dataset(path_ntz: str):
    """
    Carrega dados pre-procesados .NPZ
    """
    try:
        dados = np.load(path_ntz, allow_pickle=True)
        X = dados["X"]
        Y = dados["y"]
        return X,Y    
    except:
        print("Erro ao ler dados")
        return None
    
# region IMAGE TO INPUT
# A usada é que está como "image_to_input" caso quiser usar outra apenas altere o nome da função. 

def image_to_input_old(frame: np.ndarray) -> np.ndarray:
    """
    Recebe um frame (imagem OpenCV), converte para HSV, redimensiona,
    normaliza e achata (flatten) para alimentar a MLP.
    """
    hsv_frame  = cv2.cvtColor(frame,  cv2.COLOR_BGR2HSV)
    resized_frame = cv2.resize(hsv_frame,(64,64))
    normalized_frame = resized_frame.astype('float32') / 255.0
    flattened_input = normalized_frame.flatten()
    return flattened_input

def image_to_input_hog(frame):
    # 1. Grayscale (HOG funciona melhor em cinza)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    
    # 2. Extrair características HOG
    # Isso transforma 4096 pixels em um vetor muito menor e mais inteligente
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
    features = features.astype('float32') / 255.0
    return features # Agora a MLP recebe "formas" em vez de "pixels"

def image_to_input_CNN_ORIGINAL(image):
    # 1. Redimensionar
    img = cv2.resize(image, (64, 64))
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. CONVOLUÇÃO MANUAL (Filtros de Detecção de Bordas)
    # Simulando o que uma CNN aprenderia na primeira camada
    kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    feat_h = ndimage.convolve(img, kernel_horizontal)
    feat_v = ndimage.convolve(img, kernel_vertical)
    
    # 3. POOLING (Redução de dados)
    # Pegamos apenas 1 pixel a cada 2 (simplificando a imagem)
    pooled = feat_h[::2, ::2] + feat_v[::2, ::2]
    
    # 4. FLATTEN
    return pooled.flatten()

def image_to_input_CNN_V1_BUF(image):
    # 1. Redimensionar para 128 (Equilíbrio entre detalhe e processamento)
    img = cv2.resize(image, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Convolução de Sobel (Bordas)
    k_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    k_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    feat_h = ndimage.convolve(img, k_h)
    feat_v = ndimage.convolve(img, k_v)
    
    # Magnitude das bordas (Hipotenusa) - destaca o colete
    mag = np.hypot(feat_h, feat_v)
    
    # 3. Pooling Agressivo (128 -> 32)
    # Reduzimos para um vetor de 1024 neurônios, perfeito para MLP
    pooled = block_reduce(mag, block_size=(4, 4), func=np.max)
    
    # 4. Normalização (Crucial para o Adam solver)
    final = pooled.flatten()
    if final.max() > 0:
        final = final / final.max()
        
    return final

def image_to_input_cnn_v2(image):
    """
    CNN Caseira V2: Extrai Bordas (Forma) + Matiz (Cor) + Saturação (Intensidade)
    """
    # 1. RESOLUÇÃO MAIOR (Já que tamanho não é problema)
    img_resized = cv2.resize(image, (64, 64))
    
    # 2. SEPARAÇÃO DE CANAIS
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(float)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV).astype(float)
    
    hue = hsv[:, :, 0] # Matiz (Qual é a cor? Laranja/Amarelo)
    sat = hsv[:, :, 1] # Saturação (Quão "Neon" é essa cor?)
    
    # 3. CONVOLUÇÃO (Detecção de Bordas Melhorada)
    kernel_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    feat_h = ndimage.convolve(gray, kernel_horizontal)
    feat_v = ndimage.convolve(gray, kernel_vertical)
    
    # Pitágoras para achar a magnitude real da borda
    edge_magnitude = np.hypot(feat_h, feat_v) 
    
    # 4. POOLING (Reduz de 128x128 para 64x64 pegando pixels alternados)
    edge_pooled = edge_magnitude[::2, ::2]
    hue_pooled = hue[::2, ::2]
    sat_pooled = sat[::2, ::2]
    
    # 5. NORMALIZAÇÃO INDIVIDUAL (Essencial para a MLP)
    # Se não normalizar, a rede vai dar mais peso para quem tem números maiores
    max_edge = np.max(edge_pooled)
    edge_norm = edge_pooled / max_edge if max_edge > 0 else edge_pooled
    hue_norm = hue_pooled / 180.0  # Hue no OpenCV vai de 0 a 180
    sat_norm = sat_pooled / 255.0  # Saturação vai de 0 a 255
    
    # 6. FLATTEN E FUSÃO DOS MAPAS
    # Empilhamos a Forma, a Cor e o "Neon" em um vetor gigante
    return np.concatenate([
        edge_norm.flatten(), 
        hue_norm.flatten(), 
        sat_norm.flatten()
    ])

def image_to_input_BEST(image):
    """
    CNN Caseira V3: Extrai Bordas (Forma) + Matiz (Cor) + Saturação (Intensidade)
    """
    # 1. Resize para 128x128
    img = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- FORMA: Filtro de Bordas com Max Pooling ---
    k_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    k_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    edge_h = ndimage.convolve(gray, k_h)
    edge_v = ndimage.convolve(gray, k_v)
    edges = np.hypot(edge_h, edge_v)
    
    # Max Pooling Manual (Blocos 4x4) - Reduz 128 -> 32
    # Isso traz estabilidade: "Existe uma borda forte nesta região?"
    edges_pooled = block_reduce(edges, block_size=(4, 4), func=np.max)

    # --- COR: Histograma por Quadrantes (Simula Convolução de Cor) ---
    # Dividimos a imagem em 4 partes (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    # Isso ajuda a rede a saber ONDE a cor está sem precisar de 12k pixels.
    h, w, _ = hsv.shape
    quadrantes = [
        hsv[0:h//2, 0:w//2], hsv[0:h//2, w//2:w],
        hsv[h//2:h, 0:w//2], hsv[h//2:h, w//2:w]
    ]
    
    cor_features = []
    for q in quadrantes:
        # Histograma de Hue (16 bins) e Sat (8 bins) por quadrante
        hist_h = cv2.calcHist([q], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([q], [1], None, [8], [0, 255]).flatten()
        cor_features.extend(hist_h / (hist_h.sum() + 1e-6))
        cor_features.extend(hist_s / (hist_s.sum() + 1e-6))

    # --- FINALIZAÇÃO ---
    edges_final = edges_pooled.flatten() / (edges_pooled.max() + 1e-6)
    #return edges_final

    return np.concatenate([edges_final, cor_features])


def image_to_input(image):
    """
    Mais Proximo de CNN Real, Mult Pool + Mult Conv
    """
    #  Resize inicial grande para manter detalhes
    img_640 = cv2.resize(image, (640, 640))
    gray = cv2.cvtColor(img_640, cv2.COLOR_BGR2GRAY).astype(float)
    
    
    k_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    k_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    e1_h = ndimage.convolve(gray, k_h)
    e1_v = ndimage.convolve(gray, k_v)
    mapa1 = np.hypot(e1_h, e1_v)
    mapa1 = np.maximum(0, mapa1) # ReLU manual para limpar ruído
    
    # POOLING 1: 640 -> 160 (Redução de 4x)
    # Mantém as bordas mais fortes de cada bloco de 4x4 pixels
    camada1_out = block_reduce(mapa1, block_size=(4, 4), func=np.max)
    
    # Aplicamos a convolução novamente sobre o mapa já reduzido
    e2_h = ndimage.convolve(camada1_out, k_h)
    e2_v = ndimage.convolve(camada1_out, k_v)
    mapa2 = np.hypot(e2_h, e2_v)
    mapa2 = np.maximum(0, mapa2)
    
    # POOLING 2: 160 -> 40 (Redução de mais 4x)
    # Resultado final: mapa de características de 40x40
    edges_final = block_reduce(mapa2, block_size=(4, 4), func=np.max)
    
    # --- INTEGRAÇÃO DE COR ---

    hsv = cv2.cvtColor(img_640, cv2.COLOR_BGR2HSV)
    cor_features = extrair_cores_quadrantes(hsv) # Sua função de histogramas
    
   
    return np.concatenate([edges_final.flatten() / (edges_final.max() + 1e-6), cor_features])

    

def extrair_cores_quadrantes(hsv):
    h, w, _ = hsv.shape
    quadrantes = [
        hsv[0:h//2, 0:w//2], hsv[0:h//2, w//2:w],
        hsv[h//2:h, 0:w//2], hsv[h//2:h, w//2:w]
    ]
    
    cor_features = []
    for q in quadrantes:
        # Histograma de Hue (16 bins) e Sat (8 bins) por quadrante
        hist_h = cv2.calcHist([q], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([q], [1], None, [8], [0, 255]).flatten()
        cor_features.extend(hist_h / (hist_h.sum() + 1e-6))
        cor_features.extend(hist_s / (hist_s.sum() + 1e-6))
    return cor_features

def relu(x):
    return np.maximum(0, x)

def image_to_input_mult(image):
    "MULTIPLOS POOLING DEEP LEARNING YAY"
    # --- PREPARAÇÃO ---
    img = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- CAMADA 1: BORDAS PRIMÁRIAS (640x640) ---
    k_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    k_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


    
    e1_h = ndimage.convolve(gray, k_h)
    e1_v = ndimage.convolve(gray, k_v)
    mapa1 = relu(np.hypot(e1_h, e1_v)) # Aplica ReLU para limpar ruído negativo
    
    # POOLING 1: 640 -> 160 (Redução de 4x)
    camada1_out = block_reduce(mapa1, block_size=(4, 4), func=np.max)
    
    # --- CAMADA 2: BORDAS SECUNDÁRIAS (160x160) ---
    # Aqui a rede olha para as bordas das bordas (detecta formas maiores)
    e2_h = ndimage.convolve(camada1_out, k_h)
    e2_v = ndimage.convolve(camada1_out, k_v)
    mapa2 = relu(np.hypot(e2_h, e2_v))
    
    # POOLING 2: 160 -> 40 (Redução de mais 4x)
    camada2_out = block_reduce(mapa2, block_size=(4, 4), func=np.max)
    

    
    # --- FINALIZAÇÃO ---
    edges_final = camada2_out.flatten()
    # Normalização final para a MLP
    
    if edges_final.max() > 0:
        edges_final = edges_final / edges_final.max()
    
        # --- EXTRAÇÃO DE COR (Mesma lógica de quadrantes) ---
    # ... (mantenha sua lógica de histogramas de cor aqui) ...
    # Use o hsv original para extrair as cores por quadrantes
    
    h, w, _ = hsv.shape
    quadrantes = [
        hsv[0:h//2, 0:w//2], hsv[0:h//2, w//2:w],
        hsv[h//2:h, 0:w//2], hsv[h//2:h, w//2:w]
    ]
    cor_features = []
    for q in quadrantes:
        # Histograma de Hue (16 bins) e Sat (8 bins) por quadrante
        hist_h = cv2.calcHist([q], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([q], [1], None, [8], [0, 255]).flatten()
        cor_features.extend(hist_h / (hist_h.sum() + 1e-6))
        cor_features.extend(hist_s / (hist_s.sum() + 1e-6))
        
    return np.concatenate([edges_final, cor_features])



def load_model(dir_models_npz: str) -> MLPClassifier:
    try:
        model = joblib.load(dir_models_npz) # np.load(dir_models_npz, allow_pickle=True)
        return model
    except:
        print("Modelo não carregou!")
        return None
def image_to_input_hybrid_hog(frame):
    # 1. Parte Geométrica (HOG)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features_hog = hog(cv2.resize(gray, (64, 64)), orientations=9, 
                       pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    # 2. Parte de Cor (Histograma HSV reduzido)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Pegamos apenas o canal H (Matiz/Cor) e criamos um histograma simples
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    hist_h = hist_h / hist_h.sum() # Normaliza
    
    # 3. JUNTAR TUDO
    return np.hstack((features_hog, hist_h))

def image_to_input_HOG_FINAL(frame):
    """
    HOG FINAL
    Simula uma CNN: 
    1. Extrai Forma (HOG - Convolução de bordas)
    2. Extrai Cor (Histograma HSV - Sensores de cor)
    """
    # --- PARTE 1: REDIMENSIONAMENTO ---
    # 64x64 é o ponto ideal para MLPs não ficarem lentas
    img_resized = cv2.resize(frame, (64, 64))
    
    # --- PARTE 2: FORMA (HOG) ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Extraímos as bordas. O colete tem linhas muito retas (faixas e gola)
    # orientations=9 captura ângulos a cada 20 graus
    features_hog = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        visualize=False
    )
    #cv2.imwrite("visualizacao_hog.png", features_hog)
    # --- PARTE 3: COR (HISTOGRAMA HSV) ---
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Calculamos a distribuição de cores apenas no canal H (Hue/Matiz)
    # Isso identifica se a imagem tem muitos pixels "Laranja/Amarelo"
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    
    # Normalização: Importante para a MLP não dar peso excessivo à cor
    if hist_h.sum() > 0:
        hist_h = hist_h / hist_h.sum()
        
    # --- PARTE 4: FUSÃO (CONCATENAÇÃO) ---
    # Juntamos o vetor de forma (1764 itens) com o de cor (32 itens)
    # Total de entrada para a MLP: 1796 neurônios
    return np.concatenate([features_hog, hist_h])

#endregion

if __name__ == "__main__":
    dir_test = "DATA/test/test/"
    dir_train= "DATA/train/train/"
    dir_val="DATA/valid/valid/"
    if True:
        conveter_dataset(dir_test)
        conveter_dataset(dir_val)
        conveter_dataset(dir_train)



        print("CABOU")

    if False:
        plot_class_distribution(dir_test)
        plot_class_distribution(dir_train)
        plot_class_distribution(dir_val)



    


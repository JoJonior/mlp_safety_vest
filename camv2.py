import streamlit as st
import cv2
import threading
import time
import numpy as np
from PIL import Image

from preprocessamento import image_to_input, load_model

# --- LOGICA DE BACKEND ---

class CameraServer:
    """Gerencia um stream de câmera em thread separada (ESP32 ou webcam local)."""

    def __init__(self, url):
        self.url = url
        self.frame = None
        self.rodando = False
        self.thread = None

    def atualizar(self):
        cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.rodando:
            ret, frame = cap.read()
            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(self.url)
        cap.release()

    def iniciar(self, nova_url=None):
        if nova_url is not None:
            self.url = nova_url
        if not self.rodando:
            self.rodando = True
            self.thread = threading.Thread(target=self.atualizar, daemon=True)
            self.thread.start()

    def parar(self):
        self.rodando = False
        self.frame = None


# --- SINGLETON DE GERENCIAMENTO --- LOGICA DA UNITY XD

@st.cache_resource
def get_camera_manager():
    # URL padrão da ESP32, Exemplo: CameraWebServer
    ip_esp = "192.168.0.115:81"
    return CameraServer(f"http://{ip_esp}/stream")


@st.cache_resource
def get_model():
    model_name = "model_clasifier-3"
   
    return load_model( f"run/{model_name}/pesos/best.pkl")
    


cam_manager = get_camera_manager()
model = get_model()


# --- HELPERS ---

def classificar_frame(frame: np.ndarray):
    """Roda o modelo e desenha o resultado sobre o frame."""
    input_atual = image_to_input(frame).reshape(1, -1)

    probs = model.predict_proba(input_atual)

    classe_id = probs.argmax()
    confianca = probs.max()

    nome_classe = "Com Colete" if classe_id == 1 else "Sem Colete"
    cor = (0, 200, 80) if classe_id == 1 else (220, 40, 40)

    texto_classe = f"Classe: {nome_classe}, Conf: {confianca:.2f}"
    frame_display = cv2.resize(frame, (480, 480))
    cv2.putText(
        frame_display, texto_classe,
        (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA
    )
    cv2.putText(
        frame_display, texto_classe,
        (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2, cv2.LINE_AA
    )
    return frame_display, nome_classe


# --- INTERFACE ---

st.set_page_config(page_title="Hub de Câmera – EPI", layout="wide")
st.title("📡 Hub Central de Câmera – Detecção de EPI")

menu_col, view_col = st.columns([1, 3])

with menu_col:
    st.markdown("### 🎛️ Fonte de Entrada")

    fonte = st.radio(
        "Selecione a fonte:",
        options=["ESP32 (stream)", "Webcam / Câmera local", "Enviar imagem"],
        index=0,
    )

    st.divider()

    # ── ESP32 ──────────────────────────────────────────────────────────────
    if fonte == "ESP32 (stream)":
        esp_url = st.text_input(
            "URL do stream ESP32",
            value="http://192.168.0.115:81/stream",
        )

        if st.button("▶ Ligar ESP32"):
            cam_manager.parar()
            time.sleep(0.3)
            cam_manager.iniciar(nova_url=esp_url)
            st.success("Stream ESP32 ativo")

        if st.button("⏹ Desligar"):
            cam_manager.parar()
            st.warning("Stream parado")

        st.info("Outros PCs podem acessar este hub pelo IP da máquina na porta 8501.")

    # ── WEBCAM ─────────────────────────────────────────────────────────────
    elif fonte == "Webcam / Câmera local":
        device_id = st.number_input(
            "Índice do dispositivo (0 = padrão)", min_value=0, max_value=10,
            value=0, step=1
        )

        if st.button("▶ Ligar Webcam"):
            cam_manager.parar()
            time.sleep(0.3)
            cam_manager.iniciar(nova_url=int(device_id))
            st.success(f"Webcam {device_id} ativa")

        if st.button("⏹ Desligar"):
            cam_manager.parar()
            st.warning("Stream parado")

    # ── IMAGEM ─────────────────────────────────────────────────────────────
    else:  # "Enviar imagem"
        # Para quando vier de stream
        if cam_manager.rodando:
            cam_manager.parar()

        uploaded = st.file_uploader(
            "Selecione uma imagem (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"],
        )

        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            frame_upload = np.array(pil_img)

            st.image(frame_upload, caption="Imagem enviada", width='stretch') #use_container_width=True)

            if st.button("🔍 Classificar imagem"):
                resultado, nome = classificar_frame(frame_upload)
                st.session_state["resultado_upload"] = resultado
                st.session_state["nome_upload"] = nome

        if "resultado_upload" in st.session_state:
            with view_col:
                st.image(
                    st.session_state["resultado_upload"],
                    caption=f"Resultado: {st.session_state['nome_upload']}",
                    width='stretch',
                )
            st.stop()

# ── LOOP DE STREAM (ESP32 ou Webcam) ───────────────────────────────────────

with view_col:
    if not cam_manager.rodando:
        st.info("Selecione uma fonte e ligue a câmera no menu à esquerda.")
    else:
        placeholder = st.empty()
        while cam_manager.rodando:
            if cam_manager.frame is not None:
                frame = cam_manager.frame.copy()
                frame_display, _ = classificar_frame(frame)
                placeholder.image(frame_display, use_container_width=True)
            time.sleep(0.03)  # ~30 FPS
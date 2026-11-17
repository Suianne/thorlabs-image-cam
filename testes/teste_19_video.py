from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, ROI

import cv2
import csv
import queue
import threading
import time
import numpy as np


# ==========================================================
# CONFIGURACOES
# ==========================================================

EXPOSICAO_US = 1

LARGURA_SENSOR = 112
ALTURA_SENSOR = 88

DURACAO_AQUISICAO_S = 10
TEMPO_AQUECIMENTO_S = 1

MAX_INTENSIDADE = 1023

# Arquivos gerados
ARQUIVO_RAW = "aquisicao_thorlabs.raw"
ARQUIVO_CSV = "frames_thorlabs.csv"
ARQUIVO_VIDEO = "aquisicao_thorlabs.mp4"

# Buffer entre a camera e a gravacao em disco.
# Se encher, o programa contabiliza a perda em vez de bloquear a aquisicao.
TAMANHO_FILA = 5000

# Codec AVI. XVID costuma estar disponivel em instalacoes OpenCV comuns.
FOURCC = "mp4v"


# ==========================================================
# ESTADO COMPARTILHADO
# ==========================================================

sdk = None
camera = None

executando = threading.Event()
fila_frames = queue.Queue(maxsize=TAMANHO_FILA)

lock = threading.Lock()

frames_recebidos = 0
frames_perdidos_camera = 0
frames_descartados_fila = 0
frames_gravados_raw = 0

fps_aquisicao = 0.0
inicio_aquisicao = None
fim_aquisicao = None

erro_aquisicao = None
erro_gravacao = None

primeiro_frame_id = None
ultimo_frame_id = None


# ==========================================================
# THREAD DE GRAVACAO RAW
# ==========================================================

def gravar_frames_raw():
    """Grava os frames recebidos em formato RAW, sem compressao."""

    global frames_gravados_raw
    global erro_gravacao

    try:
        with open(ARQUIVO_RAW, "wb") as arquivo_raw, \
             open(ARQUIVO_CSV, "w", newline="") as arquivo_csv:

            escritor_csv = csv.writer(arquivo_csv)
            escritor_csv.writerow(["indice_arquivo", "frame_id", "timestamp_s"])

            indice = 0

            while executando.is_set() or not fila_frames.empty():
                try:
                    item = fila_frames.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    fila_frames.task_done()
                    break

                frame_id, timestamp, imagem = item

                # Cada imagem e armazenada como matriz uint16 contigua.
                arquivo_raw.write(np.ascontiguousarray(imagem).tobytes())
                escritor_csv.writerow([indice, frame_id, f"{timestamp:.9f}"])

                indice += 1
                with lock:
                    frames_gravados_raw += 1

                fila_frames.task_done()

    except Exception as erro:
        with lock:
            erro_gravacao = erro
        executando.clear()


# ==========================================================
# THREAD DE AQUISICAO
# ==========================================================

def adquirir_frames():
    global frames_recebidos
    global frames_perdidos_camera
    global frames_descartados_fila
    global fps_aquisicao
    global inicio_aquisicao
    global fim_aquisicao
    global erro_aquisicao
    global primeiro_frame_id
    global ultimo_frame_id

    ultimo_id = None
    contador_fps = 0
    ultimo_calculo = time.monotonic()

    inicio_aquisicao = ultimo_calculo

    try:
        while executando.is_set():

            frame = camera.get_pending_frame_or_null()

            if frame is None:
                continue

            agora = time.monotonic()
            frame_id = frame.frame_count
            imagem = np.asarray(frame.image_buffer).copy()

            with lock:
                frames_recebidos += 1
                contador_fps += 1

                if primeiro_frame_id is None:
                    primeiro_frame_id = frame_id

                if ultimo_id is not None:
                    diferenca = frame_id - ultimo_id
                    if diferenca > 1:
                        frames_perdidos_camera += diferenca - 1

                ultimo_id = frame_id
                ultimo_frame_id = frame_id

            # Nao fazemos conversao de imagem nem codificacao de video
            # nesta thread. Apenas entregamos o frame ao gravador.
            try:
                fila_frames.put_nowait((frame_id, agora, imagem))
            except queue.Full:
                with lock:
                    frames_descartados_fila += 1

            if agora - ultimo_calculo >= 1.0:
                with lock:
                    fps_aquisicao = contador_fps / (agora - ultimo_calculo)

                contador_fps = 0
                ultimo_calculo = agora

    except Exception as erro:
        with lock:
            erro_aquisicao = erro
        executando.clear()

    finally:
        fim_aquisicao = time.monotonic()


# ==========================================================
# CONVERSAO RAW -> VIDEO
# ==========================================================

def converter_raw_para_video(fps_video):
    """Converte o RAW uint16 em AVI usando o FPS medido na aquisicao."""

    bytes_por_frame = LARGURA_SENSOR * ALTURA_SENSOR * np.dtype(np.uint16).itemsize
    tamanho_raw = __import__("os").path.getsize(ARQUIVO_RAW)

    quantidade_frames = tamanho_raw // bytes_por_frame

    if quantidade_frames == 0:
        raise RuntimeError("Nenhum frame foi gravado no arquivo RAW.")

    if fps_video <= 0:
        raise RuntimeError("FPS medido invalido; nao foi possivel criar o video.")

    # O FFmpeg pode rejeitar FPS fracionario muito alto porque a base
    # temporal resultante fica com denominador acima do limite do MPEG-4.
    # Usar um FPS inteiro mantem a base temporal simples (1/FPS).
    fps_video_arquivo = max(1, int(round(fps_video)))

    print(
        f"FPS medido: {fps_video:.3f} | "
        f"FPS usado no AVI: {fps_video_arquivo}"
    )

    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    video = cv2.VideoWriter(
        ARQUIVO_VIDEO,
        fourcc,
        fps_video_arquivo,
        (LARGURA_SENSOR, ALTURA_SENSOR),
        isColor=True
    )

    if not video.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir o VideoWriter com o codec {FOURCC}. "
            "Tente trocar FOURCC para 'MJPG'."
        )

    try:
        with open(ARQUIVO_RAW, "rb") as arquivo_raw:
            for _ in range(quantidade_frames):
                dados = arquivo_raw.read(bytes_por_frame)

                if len(dados) != bytes_por_frame:
                    break

                imagem_16bits = np.frombuffer(
                    dados,
                    dtype=np.uint16
                ).reshape((ALTURA_SENSOR, LARGURA_SENSOR))

                # Converte 10 bits para 8 bits para compatibilidade com AVI.
                imagem_8bits = np.clip(
                    imagem_16bits.astype(np.float32) *
                    (255.0 / MAX_INTENSIDADE),
                    0,
                    255
                ).astype(np.uint8)

                imagem_bgr = cv2.cvtColor(
                    imagem_8bits,
                    cv2.COLOR_GRAY2BGR
                )

                video.write(imagem_bgr)

    finally:
        video.release()

    return quantidade_frames


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

thread_gravacao = None
thread_aquisicao = None

try:
    print("\\n[1] Inicializando SDK...")
    sdk = TLCameraSDK()

    cameras = sdk.discover_available_cameras()
    if not cameras:
        raise RuntimeError("Nenhuma camera Thorlabs encontrada.")

    camera = sdk.open_camera(cameras[0])

    print("[2] Configurando camera...")
    camera.roi = ROI(
        0,
        0,
        LARGURA_SENSOR - 1,
        ALTURA_SENSOR - 1
    )

    camera.exposure_time_us = EXPOSICAO_US
    camera.frames_per_trigger_zero_for_unlimited = 0
    camera.is_frame_rate_control_enabled = False

    largura_real = camera.image_width_pixels
    altura_real = camera.image_height_pixels

    if (largura_real, altura_real) != (LARGURA_SENSOR, ALTURA_SENSOR):
        raise RuntimeError(
            f"ROI aplicada ({largura_real}x{altura_real}) difere da "
            f"configuracao solicitada ({LARGURA_SENSOR}x{ALTURA_SENSOR})."
        )

    print("\\n========== CONFIGURACAO ==========")
    print(f"Resolucao: {largura_real} x {altura_real}")
    print(f"Exposicao: {EXPOSICAO_US} us")
    print(f"Duracao programada: {DURACAO_AQUISICAO_S} s")
    print(f"Fila de gravacao: {TAMANHO_FILA} frames")
    print(f"Arquivo RAW: {ARQUIVO_RAW}")
    print(f"Arquivo de video: {ARQUIVO_VIDEO}")
    print("==================================")

    print("\\n[3] Armando camera...")
    camera.arm(16)
    camera.issue_software_trigger()

    print("[4] Aquecendo camera...")
    inicio_aquecimento = time.monotonic()

    while time.monotonic() - inicio_aquecimento < TEMPO_AQUECIMENTO_S:
        camera.get_pending_frame_or_null()

    # Inicia o gravador antes da aquisicao.
    thread_gravacao = threading.Thread(
        target=gravar_frames_raw,
        name="gravacao-raw",
        daemon=True
    )
    thread_gravacao.start()

    print("[5] Iniciando aquisicao...")
    executando.set()

    thread_aquisicao = threading.Thread(
        target=adquirir_frames,
        name="aquisicao-camera",
        daemon=True
    )
    thread_aquisicao.start()

    print("\\nAQUISICAO E GRAVACAO INICIADAS.")
    print(f"O teste terminara automaticamente em {DURACAO_AQUISICAO_S} s.")
    print("Pressione Ctrl+C para interromper antes.\\n")

    time.sleep(DURACAO_AQUISICAO_S)

except KeyboardInterrupt:
    print("\\nInterrupcao solicitada pelo usuario.")

except Exception as erro:
    print(f"\\nERRO: {type(erro).__name__}: {erro}")

finally:
    print("\\nEncerrando aquisicao...")
    executando.clear()

    if thread_aquisicao is not None:
        thread_aquisicao.join()

    # Aguarda o gravador esvaziar a fila e finalizar o arquivo RAW.
    if thread_gravacao is not None:
        thread_gravacao.join()

    if camera is not None:
        try:
            if camera.is_armed:
                camera.disarm()
        finally:
            camera.dispose()

    if sdk is not None:
        sdk.dispose()

    # ======================================================
    # RESULTADOS E CONVERSAO
    # ======================================================

    print("\\n========== RESULTADOS DA AQUISICAO ==========")
    print("Frames recebidos pela aplicacao:", frames_recebidos)
    print("Frames perdidos segundo frame_count:", frames_perdidos_camera)
    print("Frames descartados por fila cheia:", frames_descartados_fila)
    print("Frames gravados em RAW:", frames_gravados_raw)

    tempo_total = None
    fps_medio = 0.0

    if inicio_aquisicao is not None and fim_aquisicao is not None:
        tempo_total = fim_aquisicao - inicio_aquisicao
        print("Tempo efetivo:", round(tempo_total, 4), "s")

        if tempo_total > 0:
            fps_medio = frames_recebidos / tempo_total
            print("FPS medio de aquisicao:", round(fps_medio, 2))

    print("Ultimo FPS instantaneo:", round(fps_aquisicao, 2))

    if erro_aquisicao is not None:
        print("Erro na aquisicao:", repr(erro_aquisicao))

    if erro_gravacao is not None:
        print("Erro na gravacao RAW:", repr(erro_gravacao))

    print("=============================================")

    if erro_gravacao is None and frames_gravados_raw > 0 and fps_medio > 0:
        try:
            print("\\nConvertendo RAW para video...")
            total_video = converter_raw_para_video(fps_medio)

            print("\\nVIDEO MP4 GERADO COM SUCESSO")
            print("Arquivo:", ARQUIVO_VIDEO)
            print("Frames no video:", total_video)
            print("FPS configurado no video:", round(fps_medio, 2))
            print(
                "Duracao estimada do video:",
                round(total_video / fps_medio, 3),
                "s"
            )
        except Exception as erro:
            print(f"\\nFalha ao gerar video: {type(erro).__name__}: {erro}")
            print("O arquivo RAW e o CSV foram preservados para conversao posterior.")

    print("\\nArquivos auxiliares:")
    print("-", ARQUIVO_RAW)
    print("-", ARQUIVO_CSV)
    print("\\nSDK encerrado.")

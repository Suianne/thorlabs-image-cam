from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, ROI
import numpy as np
from PIL import Image

import time
import threading
import queue
import os

from datetime import datetime


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

SERIAL = "36717"

EXPOSICAO = 100

LARGURA_SENSOR = 1280
ALTURA_SENSOR = 1024

DURACAO = 10  # segundos

BUFFER_CAMERA = 64

TAMANHO_FILA = 64

PASTA_BASE = "capturas_teste_17"


# ==========================================================
# VARIÁVEIS COMPARTILHADAS
# ==========================================================

fila_frames = queue.Queue(maxsize=TAMANHO_FILA)

erro_aquisicao = None
erro_gravacao = None

frames_recebidos = 0
frames_salvos = 0
frames_perdidos = 0

executando = threading.Event()

lock = threading.Lock()


# ==========================================================
# THREAD DE AQUISIÇÃO
# ==========================================================

def adquirir_frames(camera):

    global frames_recebidos
    global frames_perdidos
    global erro_aquisicao

    ultimo_id = None

    inicio = time.perf_counter()
    limite = inicio + DURACAO

    try:

        while time.perf_counter() < limite:

            if erro_gravacao is not None:
                break

            frame = camera.get_pending_frame_or_null()

            if frame is None:
                continue

            # Copia os dados antes que o SDK reutilize o buffer
            imagem = np.asarray(frame.image_buffer).copy()

            frame_id = frame.frame_count

            with lock:

                frames_recebidos += 1

                # Verifica lacunas na sequência
                if ultimo_id is not None:

                    diferenca = frame_id - ultimo_id

                    if diferenca > 1:
                        frames_perdidos += diferenca - 1

                ultimo_id = frame_id

            # Envia para a thread de gravação
            # Aguarda espaço sem descartar silenciosamente frames
            while True:

                if erro_gravacao is not None:
                    break

                try:

                    fila_frames.put(
                        (frame_id, imagem),
                        timeout=0.1
                    )

                    break

                except queue.Full:
                    continue

            if erro_gravacao is not None:
                break

    except Exception as erro:

        erro_aquisicao = erro

    finally:

        executando.clear()


# ==========================================================
# THREAD DE GRAVAÇÃO
# ==========================================================

def salvar_frames():

    global frames_salvos
    global erro_gravacao

    try:

        while True:

            item = fila_frames.get()

            try:

                if item is None:
                    break

                frame_id, imagem = item

                nome_arquivo = os.path.join(
                    PASTA_CAPTURA,
                    f"frame_{frame_id:08d}.tiff"
                )

                # Preserva os valores originais em 16 bits
                imagem_tiff = Image.fromarray(
                    imagem.astype(np.uint16)
                )

                imagem_tiff.save(
                    nome_arquivo,
                    format="TIFF"
                )

                with lock:
                    frames_salvos += 1

            finally:

                fila_frames.task_done()

    except Exception as erro:

        erro_gravacao = erro


# ==========================================================
# PROGRAMA PRINCIPAL
# ==========================================================

sdk = None
camera = None

camera_armada = False

thread_aquisicao = None
thread_gravacao = None

try:

    # ------------------------------------------------------
    # Criação da pasta de saída
    # ------------------------------------------------------

    identificador = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    PASTA_CAPTURA = os.path.join(
        PASTA_BASE,
        f"captura_{identificador}"
    )

    os.makedirs(
        PASTA_CAPTURA,
        exist_ok=True
    )

    print("\n========== TESTE 17 ==========")

    print(f"Pasta de saída: {PASTA_CAPTURA}")

    # ------------------------------------------------------
    # Inicialização do SDK
    # ------------------------------------------------------

    print("\n[1] Inicializando SDK...")

    sdk = TLCameraSDK()

    cameras = sdk.discover_available_cameras()

    if not cameras:
        raise RuntimeError("Nenhuma câmera encontrada.")

    print("Câmeras disponíveis:", cameras)

    # Seleção da câmera pelo número de série
    if SERIAL in cameras:

        serial = SERIAL

    else:

        serial = cameras[0]

        print(
            f"Serial {SERIAL} não encontrado. "
            f"Utilizando {serial}."
        )

    print(f"\n[2] Abrindo câmera: {serial}")

    camera = sdk.open_camera(serial)

    print("Modelo:", camera.model)
    print("Serial:", camera.serial_number)

    # ------------------------------------------------------
    # Configuração da câmera
    # ------------------------------------------------------

    camera.roi = ROI(
        0,
        0,
        LARGURA_SENSOR - 1,
        ALTURA_SENSOR - 1
    )

    camera.exposure_time_us = EXPOSICAO

    camera.frames_per_trigger_zero_for_unlimited = 0

    camera.is_frame_rate_control_enabled = False

    print("\n========== CONFIGURAÇÃO ==========")

    print(
        f"Resolução: "
        f"{camera.image_width_pixels} x "
        f"{camera.image_height_pixels}"
    )

    print(f"Exposição: {camera.exposure_time_us} us")
    print(f"Buffer da câmera: {BUFFER_CAMERA}")
    print(f"Fila de gravação: {TAMANHO_FILA}")
    print(f"Duração: {DURACAO} segundos")
    print("Formato: TIFF 16 bits")

    print("==================================")

    # ------------------------------------------------------
    # Preparação da aquisição
    # ------------------------------------------------------

    print("\n[3] Armando câmera...")

    camera.arm(BUFFER_CAMERA)

    camera_armada = True

    camera.issue_software_trigger()

    # ------------------------------------------------------
    # Aquecimento
    # ------------------------------------------------------

    print("\n[4] Aquecendo câmera por 1 segundo...")

    inicio_aquecimento = time.perf_counter()

    while time.perf_counter() - inicio_aquecimento < 1:

        camera.get_pending_frame_or_null()

    print("Aquecimento concluído.")

    # ------------------------------------------------------
    # Inicialização das threads
    # ------------------------------------------------------

    print("\n[5] Iniciando gravação e aquisição...")

    thread_gravacao = threading.Thread(
        target=salvar_frames,
        daemon=False
    )

    thread_aquisicao = threading.Thread(
        target=adquirir_frames,
        args=(camera,),
        daemon=False
    )

    thread_gravacao.start()

    executando.set()

    inicio_teste = time.perf_counter()

    thread_aquisicao.start()

    # ------------------------------------------------------
    # Acompanhamento
    # ------------------------------------------------------

    while thread_aquisicao.is_alive():

        thread_aquisicao.join(timeout=1)

        with lock:

            recebidos = frames_recebidos
            salvos = frames_salvos

        print(
            f"\rRecebidos: {recebidos} | "
            f"Salvos: {salvos} | "
            f"Na fila: {fila_frames.qsize()}",
            end="",
            flush=True
        )

    fim_aquisicao = time.perf_counter()

    # ------------------------------------------------------
    # Encerramento da aquisição
    # ------------------------------------------------------

    executando.clear()

    thread_aquisicao.join()

    # A câmera não será mais utilizada pela thread
    if camera_armada:

        camera.disarm()

        camera_armada = False

    # ------------------------------------------------------
    # Finalização da gravação
    # ------------------------------------------------------

    print("\n\n[6] Finalizando gravação dos frames restantes...")

    if thread_gravacao.is_alive():

        fila_frames.put(None)

        thread_gravacao.join()

    fim_gravacao = time.perf_counter()

    # ------------------------------------------------------
    # Resultados
    # ------------------------------------------------------

    tempo_aquisicao = fim_aquisicao - inicio_teste

    tempo_gravacao = fim_gravacao - inicio_teste

    with lock:

        recebidos = frames_recebidos
        salvos = frames_salvos
        perdidos = frames_perdidos

    fps_aquisicao = recebidos / tempo_aquisicao

    fps_gravacao = salvos / tempo_gravacao

    total_sequencia = recebidos + perdidos

    percentual_lacunas = (
        perdidos / total_sequencia * 100
        if total_sequencia > 0
        else 0
    )

    print("\n========== RESULTADOS ==========")

    print(f"Frames recebidos: {recebidos}")
    print(f"Frames salvos: {salvos}")
    print(f"Frames perdidos: {perdidos}")

    print(f"Tempo de aquisição: {tempo_aquisicao:.2f} s")
    print(f"Tempo total até finalizar gravação: {tempo_gravacao:.2f} s")

    print(f"FPS médio de aquisição: {fps_aquisicao:.2f}")
    print(f"Taxa média de gravação: {fps_gravacao:.2f}")

    print(f"Percentual de lacunas: {percentual_lacunas:.2f}%")

    print(
        f"Frames não salvos: "
        f"{recebidos - salvos}"
    )

    print(f"\nImagens salvas em: {PASTA_CAPTURA}")

    print("================================")

    if erro_aquisicao is not None:

        print("\nERRO DE AQUISIÇÃO:")
        print(erro_aquisicao)

    if erro_gravacao is not None:

        print("\nERRO DE GRAVAÇÃO:")
        print(erro_gravacao)


except Exception as erro:

    print(
        f"\nERRO: {type(erro).__name__}: {erro}"
    )


finally:

    executando.clear()

    if thread_aquisicao is not None:

        thread_aquisicao.join()

    # Finaliza a thread de gravação, caso ainda esteja ativa
    if thread_gravacao is not None and thread_gravacao.is_alive():

        fila_frames.put(None)

        thread_gravacao.join()

    if camera is not None:

        if camera.is_armed:

            camera.disarm()

        camera.dispose()

    if sdk is not None:

        sdk.dispose()

    print("\n[7] SDK encerrado.")
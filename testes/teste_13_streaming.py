from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, ROI

import cv2
import time
import threading


# ==================================================
# CONFIGURAÇÕES
# ==================================================

EXPOSICAO = 5000

LARGURA_SENSOR = 320
ALTURA_SENSOR = 256

MAX_INTENSIDADE = 1023

TEMPO_AQUECIMENTO = 1

LARGURA_JANELA = 1000
ALTURA_JANELA = 800


# ==================================================
# VARIÁVEIS COMPARTILHADAS
# ==================================================

sdk = None
camera = None

executando = threading.Event()
lock = threading.Lock()

ultimo_frame = None
ultimo_frame_id = None

frames_recebidos = 0
frames_perdidos = 0

fps_aquisicao = 0
fps_visualizacao = 0

inicio_aquisicao = None
fim_aquisicao = None

erro_aquisicao = None


# ==================================================
# THREAD DE AQUISIÇÃO
# ==================================================

def adquirir_frames():

    global ultimo_frame
    global ultimo_frame_id

    global frames_recebidos
    global frames_perdidos

    global fps_aquisicao

    global inicio_aquisicao
    global fim_aquisicao

    global erro_aquisicao

    ultimo_id = None

    contador_fps = 0
    ultimo_calculo = time.monotonic()

    inicio_aquisicao = ultimo_calculo

    try:

        while executando.is_set():

            frame = camera.get_pending_frame_or_null()

            if frame is None:
                continue

            atual = frame.frame_count

            # Cópia independente do buffer da câmera
            imagem = frame.image_buffer.copy()

            with lock:

                frames_recebidos += 1
                contador_fps += 1

                # Identificação de lacunas na sequência
                if ultimo_id is not None:

                    diferenca = atual - ultimo_id

                    if diferenca > 1:
                        frames_perdidos += diferenca - 1

                ultimo_id = atual

                # Atualiza o frame disponível para visualização
                ultimo_frame = imagem
                ultimo_frame_id = atual

            agora = time.monotonic()

            # Cálculo do FPS de aquisição
            if agora - ultimo_calculo >= 1:

                novo_fps = contador_fps / (
                    agora - ultimo_calculo
                )

                with lock:
                    fps_aquisicao = novo_fps

                contador_fps = 0
                ultimo_calculo = agora

    except Exception as erro:

        with lock:
            erro_aquisicao = erro

        executando.clear()

    finally:

        fim_aquisicao = time.monotonic()


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

try:

    print("\n[1] Inicializando SDK...")

    sdk = TLCameraSDK()

    cameras = sdk.discover_available_cameras()

    if not cameras:
        raise RuntimeError("Nenhuma câmera encontrada!")

    camera = sdk.open_camera(cameras[0])

    print("[2] Configurando câmera...")

    # Sensor completo
    camera.roi = ROI(
        0,
        0,
        LARGURA_SENSOR - 1,
        ALTURA_SENSOR - 1
    )

    camera.exposure_time_us = EXPOSICAO

    # Aquisição contínua
    camera.frames_per_trigger_zero_for_unlimited = 0

    # Desabilita controle automático de FPS
    camera.is_frame_rate_control_enabled = False

    print("\n========== CONFIGURAÇÃO ==========")

    print(
        "Resolução:",
        camera.image_width_pixels,
        "x",
        camera.image_height_pixels
    )

    print("Exposição:", EXPOSICAO, "us")

    print("ROI: sensor completo")

    print("Modo: aquisição contínua")

    print("==================================")

    # ----------------------------------------------
    # PREPARAÇÃO DA CÂMERA
    # ----------------------------------------------

    print("\n[3] Armando câmera...")

    camera.arm(16)

    camera.issue_software_trigger()

    print("[4] Aquecendo câmera...")

    inicio = time.monotonic()

    while time.monotonic() - inicio < TEMPO_AQUECIMENTO:

        camera.get_pending_frame_or_null()

    # ----------------------------------------------
    # CONFIGURAÇÃO DA JANELA
    # ----------------------------------------------

    cv2.namedWindow(
        "ThorLabs - Sensor Completo",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "ThorLabs - Sensor Completo",
        LARGURA_JANELA,
        ALTURA_JANELA
    )

    # ----------------------------------------------
    # INÍCIO DA AQUISIÇÃO INDEPENDENTE
    # ----------------------------------------------

    print("[5] Iniciando thread de aquisição...")

    executando.set()

    thread_aquisicao = threading.Thread(
        target=adquirir_frames,
        daemon=True
    )

    thread_aquisicao.start()

    print("\nSTREAMING INICIADO")
    print("Pressione Q para encerrar.\n")

    # ----------------------------------------------
    # VISUALIZAÇÃO
    # ----------------------------------------------

    ultimo_frame_id_exibido = None

    contador_visualizacao = 0

    ultimo_calculo_visual = time.monotonic()

    while executando.is_set():

        with lock:

            imagem = ultimo_frame
            frame_id = ultimo_frame_id

            fps_aq = fps_aquisicao

            erro = erro_aquisicao

        if erro is not None:
            raise RuntimeError(
                f"Erro na thread de aquisição: {erro}"
            )

        # Processa somente frames novos
        if (
            imagem is not None
            and frame_id != ultimo_frame_id_exibido
        ):

            # Conversão para visualização
            imagem_8bits = cv2.convertScaleAbs(
                imagem,
                alpha=255 / MAX_INTENSIDADE
            )

            contador_visualizacao += 1

            agora = time.monotonic()

            if agora - ultimo_calculo_visual >= 1:

                fps_visualizacao = contador_visualizacao / (
                    agora - ultimo_calculo_visual
                )

                contador_visualizacao = 0

                ultimo_calculo_visual = agora

            # Informações na imagem
            cv2.putText(
                imagem_8bits,
                f"Aquisicao: {fps_aq:.1f} FPS",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2
            )

            cv2.putText(
                imagem_8bits,
                f"Visualizacao: {fps_visualizacao:.1f} FPS",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                255,
                2
            )

            cv2.imshow(
                "ThorLabs - Sensor Completo",
                imagem_8bits
            )

            ultimo_frame_id_exibido = frame_id

        # Mantém a janela responsiva
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as erro:

    print("\nERRO:", type(erro).__name__, erro)


# ==================================================
# FINALIZAÇÃO
# ==================================================

finally:

    print("\nEncerrando aquisição...")

    executando.clear()

    if 'thread_aquisicao' in locals():

        thread_aquisicao.join()

    cv2.destroyAllWindows()

    if camera is not None:

        if camera.is_armed:
            camera.disarm()

        camera.dispose()

    if sdk is not None:
        sdk.dispose()

    # ----------------------------------------------
    # RESULTADOS
    # ----------------------------------------------

    print("\n========== RESULTADOS ==========")

    print("Frames recebidos:", frames_recebidos)

    print("Frames perdidos:", frames_perdidos)

    if (
        inicio_aquisicao is not None
        and fim_aquisicao is not None
    ):

        tempo_total = fim_aquisicao - inicio_aquisicao

        print("Tempo:", round(tempo_total, 3), "s")

        if tempo_total > 0:

            print(
                "FPS médio de aquisição:",
                round(frames_recebidos / tempo_total, 2)
            )

    print("Último FPS de aquisição:", round(fps_aquisicao, 2))

    print("Último FPS de visualização:", round(fps_visualizacao, 2))

    print("================================")

    print("SDK encerrado.")
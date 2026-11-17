from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
import numpy as np
import time
from PIL import Image

SERIAL = "36717"
NOME_IMAGEM = "captura_teste.png"

sdk = None
camera = None

try:
    print("[1] Inicializando SDK...")
    sdk = TLCameraSDK()

    print("[2] Procurando câmeras...")
    cameras = sdk.discover_available_cameras()

    if not cameras:
        raise RuntimeError("Nenhuma câmera encontrada!")

    print(f"Câmeras disponíveis: {cameras}")

    serial = cameras[0]

    print(f"[3] Abrindo câmera identificada: {serial}")
    camera = sdk.open_camera(serial)

    print("[4] Câmera aberta com sucesso!")
    print("Modelo:", camera.model)
    print("Serial:", camera.serial_number)

    # Configuração da aquisição
    camera.exposure_time_us = 1000
    camera.frames_per_trigger_zero_for_unlimited = 1

    print("[5] Configurações:")
    print(f"Exposição: {camera.exposure_time_us} us")
    print(f"Resolução: {camera.image_width_pixels} x "
          f"{camera.image_height_pixels}")
    print(f"Profundidade: {camera.bit_depth} bits")

    print("\n[6] Armando câmera...")
    camera.arm(2)

    print("[7] Disparando software trigger...")
    camera.issue_software_trigger()

    print("[8] Aguardando frame...")

    frame = None
    inicio = time.time()

    while frame is None:
        frame = camera.get_pending_frame_or_null()

        if time.time() - inicio > 5:
            raise TimeoutError("Tempo limite aguardando imagem.")

        time.sleep(0.01)

    print("[9] Frame recebido!")

    imagem = np.asarray(frame.image_buffer)

    print("\nInformações do frame:")
    print("Formato:", imagem.shape)
    print("Tipo de dado:", imagem.dtype)
    print("Quantidade de pixels:", imagem.size)
    print("Valor mínimo:", imagem.min())
    print("Valor máximo:", imagem.max())

    # Ajuste dimensional caso o buffer seja unidimensional
    imagem = imagem.reshape(
        camera.image_height_pixels,
        camera.image_width_pixels
    )

    # Salvar imagem
    imagem_png = (imagem.astype(np.uint32) * 255 // 1023).astype(np.uint8)

    Image.fromarray(imagem_png).save(NOME_IMAGEM)

    print(f"\n[10] Imagem salva: {NOME_IMAGEM}")

    camera.disarm()

    print("[11] Aquisição concluída!")

except Exception as erro:
    print(f"\nERRO: {type(erro).__name__}: {erro}")

finally:
    if camera is not None:
        if camera.is_armed:
            camera.disarm()

        camera.dispose()

    if sdk is not None:
        sdk.dispose()

    print("\n[12] SDK encerrado.")
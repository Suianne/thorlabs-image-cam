from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, ROI
import time

sdk = None
camera = None

try:
    sdk = TLCameraSDK()
    cameras = sdk.discover_available_cameras()

    if not cameras:
        raise RuntimeError("Nenhuma câmera encontrada!")

    camera = sdk.open_camera(cameras[0])

    # Configuração da ROI
    camera.roi = ROI(0, 0, 127, 101)

    # Exposição mínima
    camera.exposure_time_us = 100

    # Aquisição contínua
    camera.frames_per_trigger_zero_for_unlimited = 0
    camera.is_frame_rate_control_enabled = False

    print("ROI:", camera.roi)
    print("Resolução:",
          camera.image_width_pixels,
          "x",
          camera.image_height_pixels)

    print("Exposição:", camera.exposure_time_us, "us")
    print("Tempo de leitura:", camera.sensor_readout_time_ns, "ns")

    camera.arm(16)
    camera.issue_software_trigger()

    print("\nAquecendo câmera por 1 segundo...")

    inicio = time.monotonic()

    while time.monotonic() - inicio < 1:
        camera.get_pending_frame_or_null()

    frames = 0
    perdidos = 0
    ultimo = None

    inicio = time.monotonic()
    duracao = 10

    while time.monotonic() - inicio < duracao:
        frame = camera.get_pending_frame_or_null()

        if frame is not None:
            frames += 1

            atual = frame.frame_count

            if ultimo is not None:
                diferenca = atual - ultimo

                if diferenca > 1:
                    perdidos += diferenca - 1

            ultimo = atual

    tempo = time.monotonic() - inicio

    print("\n========== RESULTADO ==========")
    print("Frames recebidos:", frames)
    print("Frames perdidos:", perdidos)
    print("Tempo:", round(tempo, 3), "s")
    print("FPS médio:", round(frames / tempo, 2))
    print("================================")

except Exception as erro:
    print("ERRO:", type(erro).__name__, erro)

finally:
    if camera is not None:
        if camera.is_armed:
            camera.disarm()
        camera.dispose()

    if sdk is not None:
        sdk.dispose()

    print("SDK encerrado.")
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

def main():
    print("[1] Iniciando teste...", flush=True)

    sdk = None

    try:
        print("[2] Inicializando SDK...", flush=True)
        sdk = TLCameraSDK()

        print("[3] SDK inicializado!", flush=True)

        print("[4] Procurando câmeras...", flush=True)
        cameras = sdk.discover_available_cameras()

        print("[5] Resultado da busca:", flush=True)

        if cameras:
            for camera in cameras:
                print(f"    Câmera encontrada: {camera}", flush=True)
        else:
            print("    Nenhuma câmera encontrada.", flush=True)

        print("[6] Teste concluído. Iniciando encerramento.", flush=True)

    except Exception as e:
        print(f"[ERRO PYTHON] {e}", flush=True)

    finally:
        print("[7] Encerrando SDK...", flush=True)

        if sdk is not None:
            sdk.dispose()

        print("[8] SDK encerrado.", flush=True)


if __name__ == "__main__":
    main()
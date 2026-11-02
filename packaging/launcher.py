"""PyInstaller entry point; the application remains entirely Python/native Qt."""
import multiprocessing


if __name__ == '__main__':
    # Must run before Qt/torch imports when a frozen child process is spawned.
    multiprocessing.freeze_support()
    from scout.desktop.bundle import main
    raise SystemExit(main())

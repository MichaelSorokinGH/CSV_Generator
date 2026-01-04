# build_exe.py
import subprocess
import os
import sys


def build():
    print("=" * 60)
    print("Сборка CSV Generator")
    print("=" * 60)

    # Команда сборки
    cmd = [
        sys.executable,  # Используем текущий Python
        "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        f"--icon={os.path.abspath('6195711.ico')}",  # Абсолютный путь
        "--name=CSV_Generator",
        f"--add-data={os.path.abspath('6195711.ico')};.",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui",
        "--hidden-import=PyQt5.QtWidgets",
        "--clean",
        "--noconfirm",
        os.path.abspath("Generator.py")
    ]

    print("Выполняемая команда:")
    print(" ".join(cmd))
    print("-" * 60)

    # Запускаем сборку
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("\n✅ Сборка успешно завершена!")
        print(f"\nEXE файл: {os.path.abspath('dist/CSV_Generator.exe')}")

        # Проверяем размер
        exe_path = 'dist/CSV_Generator.exe'
        if os.path.exists(exe_path):
            size = os.path.getsize(exe_path) / 1024 / 1024
            print(f"Размер: {size:.2f} MB")
    else:
        print("\n❌ Ошибка сборки:")
        print(result.stderr)

    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    build()
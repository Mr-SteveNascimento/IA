import sys
from cx_Freeze import setup, Executable

setup(name="cnn",
      version="0.1",
      description="My GUI application!",
      executables=[Executable("cnn.py", base=None)])

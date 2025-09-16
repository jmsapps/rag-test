from importlib import import_module as importModule
from os import (
    getcwd as getCwd,
    listdir as listDir,
)

cwd = getCwd()
scripts = {}
for fileName in listDir(f"{cwd}/src/scripts"):
    if fileName.endswith(".py") and fileName != "__init__.py":
        module = fileName.replace(".py", "")

        for funcName, func in importModule(f"scripts.{module}").__dict__.items():
            if callable(func) and funcName == "main":
                scripts[module] = func

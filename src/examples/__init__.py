from importlib import import_module as importModule
from os import (
    getcwd as getCwd,
    listdir as listDir,
)

cwd = getCwd()
examples = {}
for fileName in listDir(f"{cwd}/src/examples"):
    if fileName.endswith(".py") and fileName != "__init__.py":
        module = fileName.replace(".py", "")

        for funcName, func in importModule(f"examples.{module}").__dict__.items():
            if callable(func) and funcName == "main":
                examples[module] = func

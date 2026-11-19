"""Run an installer right after a named module is executed (shared by the ledgers)."""
import importlib.util
import sys


class PostImportHook:
    def __init__(self, target, install):
        self.target = target
        self.install = install

    def find_spec(self, name, path=None, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module
        install = self.install

        def exec_and_patch(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def on_import(target, install):
    """install(module) now if already imported, else right after import."""
    mod = sys.modules.get(target)
    if mod is not None:
        install(mod)
    elif not any(isinstance(h, PostImportHook) and h.target == target
                 for h in sys.meta_path):
        sys.meta_path.insert(0, PostImportHook(target, install))

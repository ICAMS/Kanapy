# src/kanapy/core/backend.py
def get_backend(name: str = "orix"):
    if name == "orix":
        from kanapy_orix.adapter import Backend
        return Backend()
    elif name == "mtex":
        from kanapy_mtex.adapter import Backend
        return Backend()
    else:
        raise ValueError(f"Unknown backend: {name}")
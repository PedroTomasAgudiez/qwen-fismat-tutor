"""
Archivo fantasma para evitar el error de importacion de soundfile en qwen-agent.
"""

def read(*args, **kwargs):
    return None, 22050

def write(*args, **kwargs):
    pass

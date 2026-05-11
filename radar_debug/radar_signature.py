"""
radar_signature.py — посмотреть КАКИЕ параметры реально принимает
beamngpy.sensors.Radar в твоей версии.

Это решает несоответствие: документация BeamNG.tech это Lua API
(camelCase, fovY), а Python-обёртка beamngpy может иметь другие имена
(snake_case, обрезанные, переименованные).

Запусти ОДИН РАЗ, BeamNG не нужен — это просто введение.
"""

import inspect
from beamngpy.sensors import Radar


# Сигнатура конструктора
sig = inspect.signature(Radar.__init__)
print("=" * 70)
print("Radar.__init__ — реальные параметры в твоей beamngpy:")
print("=" * 70)
for name, p in sig.parameters.items():
    if name == 'self':
        continue
    default = p.default if p.default is not inspect.Parameter.empty else "<required>"
    print(f"  {name:35s} default = {default}")

# Docstring
print()
print("=" * 70)
print("Docstring (если есть):")
print("=" * 70)
print(Radar.__init__.__doc__ or "(нет docstring)")

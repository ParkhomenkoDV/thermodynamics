from enum import Enum


class Parameters(Enum):
    """Параметры"""

    Cp = "heat_capacity_at_constant_pressure"  # теплокмкость при постоянном давлении
    Cv = "heat_capacity_at_constant_volume"  # теплокмкость при постоянном объеме
    k = "adiabatic_index"  # показатель адиабаты
    R = "gas_const"  # газовая постоянная

    # статические термодинамические параметры
    T = "static_temperature"  # статическая темпрература
    P = "static_pressure"  # статическое давление
    D = "staticdensity"  # статическая плотность

    # полные термодинамические параметры
    TT = "total_pressure"  # полная температура
    PP = "total_pressure"  # полное давление
    DD = "total_density"  # полная плотность

    m = "mass"  # масса
    V = "volume"  # объем

    # скорости
    c = "absolute_velocity"  # абсолютная скорость
    u = "portable_velocity"  # переносная скорость
    w = "relative_velocity"  # относительная скорость

    G = "mass_flow"  # массовый расход

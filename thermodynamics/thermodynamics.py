import numpy as np
from numpy import isnan, log, nan
from scipy import integrate, interpolate

from .parameters import parameters as tdp

np.seterr(invalid="ignore")  # игнорирование ошибок с nan

T0 = 273.15  # Абсолютный ноль температуры
GAS_CONST = 8.314_462_618_153_24  # Универсальная газовая постоянная


def gdf(
    parameter: str,
    equivalent_speed: float,
    adiabatic_index: float = None,
) -> float:
    """Газодинамические функции"""
    assert isinstance(parameter, str), TypeError(f"type {parameter} must be str")
    assert isinstance(equivalent_speed, (int, float, np.number)), TypeError(
        f"type {equivalent_speed} must be numeric"
    )
    parameter = parameter.upper()
    if parameter == "T":
        assert isinstance(adiabatic_index, (int, float, np.number)), TypeError(
            f"type {adiabatic_index} must be numeric"
        )
        return 1 - equivalent_speed**2 * ((adiabatic_index - 1) / (adiabatic_index + 1))
    elif parameter == "P":
        return gdf(
            "T",
            equivalent_speed=equivalent_speed,
            adiabatic_index=adiabatic_index,
        ) ** (adiabatic_index / (adiabatic_index - 1))
    elif parameter == "D":
        return gdf(
            "T",
            equivalent_speed=equivalent_speed,
            adiabatic_index=adiabatic_index,
        ) ** (1 / (adiabatic_index - 1))
    elif parameter in ("G", "MF"):
        return (
            ((adiabatic_index + 1) / 2) ** (1 / (adiabatic_index - 1))
            * equivalent_speed
            * gdf(
                "D",
                equivalent_speed=equivalent_speed,
                adiabatic_index=adiabatic_index,
            )
        )
    elif parameter in ("I", "MV"):
        return equivalent_speed + 1 / equivalent_speed
    else:
        raise ValueError(f'{parameter} not in ("T", "P", "D", "G", "MF", "I", "MV")')


def temperature_atmosphere_standard(height) -> tuple[float, str]:
    """Статическая температура стандартной атмосферы"""
    return 288.15 - 0.00651 * height if height < 11_000 else 216.65, "K"


def pressure_atmosphere_standard(height) -> tuple[float, str]:
    """Статическое давление стандартной атмосферы"""
    return (
        101_325 * (temperature_atmosphere_standard(height)[0] / 288.15) ** 5.2533
        if height < 11_000
        else 22_699.9 * np.exp((11_000 - height) / 6318)
    ), "Pa"


def atmosphere_standard(height: int | float) -> dict[str : tuple[float, str]]:
    """Атмосфера стандартная ГОСТ 4401-81"""
    return {
        tdp.T: temperature_atmosphere_standard(height),
        tdp.P: pressure_atmosphere_standard(height),
    }


def efficiency_polytropic(process="", pipi=nan, effeff=nan, k=nan) -> float:
    """Политропический КПД"""
    if pipi == 1:
        return 1
    if process.upper() in ("C", "COMPRESSION"):
        return ((k - 1) / k) * log(pipi) / log((pipi ** ((k - 1) / k) - 1) / effeff + 1)
    if process.upper() in ("E", "EXTENSION"):
        return (
            -(k / (k - 1))
            / log(pipi)
            * log(effeff * (1 / (pipi ** ((k - 1) / k)) - 1) + 1)
        )
    raise Exception(f'{process} not in ("C", "E")')


def chemical_formula_to_dict(formula: str) -> dict[str:int]:
    """Разбор хмической формулы поэлементно"""
    result = dict()

    i = 0
    while i < len(formula):
        if i + 1 < len(formula) and formula[i + 1].islower():
            atom = formula[i : i + 2]
            i += 2
        else:
            atom = formula[i]
            i += 1

        count = 0
        while i < len(formula) and formula[i].isdigit():
            count = count * 10 + int(formula[i])
            i += 1

        if count == 0:
            count = 1

        if atom in result:
            result[atom] += count
        else:
            result[atom] = count

    return result


def adiabatic_index(gas_const: float, cp: float) -> float:
    """Показатель адиабаты"""
    if cp == gas_const:
        return nan
    return cp / (cp - gas_const)


def gas_const(substance: str, excess_oxidizing=nan, fuel: str = "") -> float:
    """Газовая постоянная [Дж/кг/К]"""
    assert isinstance(substance, str), TypeError(f"type {substance} must be str")
    substance = substance.upper()
    if substance in ("AIR", "ВОЗДУХ"):
        """Газовая постоянная воздуха"""
        return 287.14
    elif substance in ("EXHAUST", "ВЫХЛОП") and fuel != "":
        """Газовая постоянная продуктов сгорания"""
        assert isinstance(excess_oxidizing, (int, float, np.number)) or not isnan(
            excess_oxidizing
        ), TypeError(f"type {excess_oxidizing} must be numeric")
        assert excess_oxidizing > 0, ValueError(f"{excess_oxidizing} must be > 0")
        assert isinstance(fuel, str), TypeError(f"type {fuel} must be str")
        fuel = fuel.upper()
        if fuel in ("C2H8N2", "KEROSENE", "КЕРОСИН"):
            return 288.1954313 + 0.691695880 / excess_oxidizing
        elif fuel in ("T1", "Т1", "РЕАКТИВНОЕ ТОПЛИВО"):
            return 288.1856907 - 0.213996376 / excess_oxidizing
        elif fuel.upper() in ("TC1", "ТС-1", "ТС1", "TC-1"):
            return 288.1901130 + 0.196844775 / excess_oxidizing
        elif fuel in (
            "DIESEL",
            "DT",
            "ДИЗЕЛЬ",
            "ДТ",
            "ДИЗЕЛЬНОЕ ТОПЛИВО",
            "ДИЗЕЛЬНОЕ_ТОПЛИВО",
        ):
            return 288.1792281 - 0.813445523 / excess_oxidizing
        elif fuel in (
            "SOLAR",
            "SOLAR OIL",
            "SOLAR_OIL",
            "СОЛЯРКА",
            "СОЛЯРОВОЕ МАСЛО",
            "СОЛЯРОВОЕ_МАСЛО",
        ):
            return 288.1729604 - 1.432301634 / excess_oxidizing
        elif fuel in ("MAZUT", "МАЗУТ", "Ф12"):
            return 288.1635509 - 2.254443192 / excess_oxidizing
        elif fuel in ("ПРИРОДНЫЙ ГАЗ", "ПРИРОДНЫЙ_ГАЗ"):
            return 290.0288864 + 12.207960640 / excess_oxidizing
        elif fuel in ("КОКСОВЫЙ ГАЗ", "КОКСОВЫЙ_ГАЗ"):
            return 288.4860344 + 20.146254880 / excess_oxidizing
        elif fuel in ("BIOGAS", "БИОГАЗ"):
            return 289.9681764 + 2.138861745 / excess_oxidizing
        else:
            raise ValueError(f"{fuel} not found")
    else:
        raise ValueError(f"{substance} not found")


def stoichiometry(fuel: str) -> float:
    """Стехиометрический коэффициент []"""
    assert isinstance(fuel, str), TypeError(f"type {fuel} must be str")
    fuel = fuel.upper()
    if fuel in (
        "C2H8N2",
        "KEROSENE",
        "T-1",
        "T-2",
        "TC-1",
        "ТС1",
        "КЕРОСИН",
        "Т-1",
        "Т-2",
        "ТС-1",
        "TC1",
    ):
        return 14.61
    elif fuel in ("PETROL", "GASOLINE", "БЕНЗИН"):
        return 14.91
    elif fuel in (
        "SOLAR",
        "SOLAR OIL",
        "SOLAR_OIL",
        "СОЛЯРКА",
        "СОЛЯРОВОЕ МАСЛО",
        "СОЛЯРОВОЕ_МАСЛО",
        "DIESEL",
        "ДИЗЕЛЬ",
    ):
        return 14.35
    elif fuel in ("MAZUT", "МАЗУТ", "Ф5", "Ф12"):
        return 13.31
    elif fuel in ("ПРИРОДНЫЙ ГАЗ", "ПРИРОДНЫЙ_ГАЗ"):
        return np.mean(
            tuple(
                {
                    "Березовский": 15.83,
                    "Войвожский": 13.69,
                    "Дашавский": 16.84,
                    "Карадагеказский": 16.51,
                    "Ленинградский": 15.96,
                    "Саратовский": 16.34,
                    "Ставропольский": 16.85,
                    "Щебеленский": 15.93,
                }.values()
            )
        )
    elif fuel in ("КОКСОВЫЙ ГАЗ", "КОКСОВЫЙ_ГАЗ"):
        return 9.908
    else:
        raise ValueError(f"{fuel} not found")


'''
def Cp(substance: str, T=nan, P=nan, a_ox=nan, fuel: str = "", **kwargs) -> float:
    """Теплоемкость при постоянном давлении"""

    if substance.upper() in ("AIR", "ВОЗДУХ"):
        """Теплоемкость воздуха"""
        if P is not nan and 1 == 0:  # TODO интерполяция по поверхности
            return Cp_air(T, P)[0][0]
        else:
            # PTM 1677-83
            _T = T / 1000
            coefs = (
                0.2521923,
                -0.1186612,
                0.3360775,
                -0.3073812,
                0.1382207,
                -0.03090246,
                0.002745383,
            )
            return 4187 * sum([coef * _T**i for i, coef in enumerate(coefs)])

    if (
        substance.upper()
        in ("ЧИСТЫЙ ВЫХЛОП", "ЧИСТЫЙ_ВЫХЛОП", "CLEAN_EXHAUST", "CLEAN EXHAUST")
        or substance.upper() in ("EXHAUST", "ВЫХЛОП")
        and a_ox == 1
    ):
        """Чистая теплоемкость выхлопа"""
        if fuel.upper() in (
            "C2H8N2",
            "KEROSENE",
            "КЕРОСИН",
            "ТС-1",
            "PETROL",
            "БЕНЗИН",
        ):
            return Cp_clean_kerosene(T)
        elif fuel.upper() in ("ДИЗЕЛЬ", "DIESEL"):
            return Cp_clean_diesel(T)

    if substance.upper() in ("EXHAUST", "ВЫХЛОП"):
        """Теплоемкость выхлопа"""
        if a_ox is not nan:
            return (
                (1 + l_stoichiometry(fuel)) * Cp("EXHAUST", T=T, a_ox=1, fuel=fuel)
                + (a_ox - 1) * l_stoichiometry(fuel) * Cp("AIR", T=T)
            ) / (1 + a_ox * l_stoichiometry(fuel))
        else:
            # PTM 1677-83
            _T = T / 1000
            coefs = (
                0.2079764,
                1.211806,
                -1.464097,
                1.291195,
                -0.6385396,
                0.1574277,
                -0.01518199,
            )
            return 4187 * sum([coef * _T**i for i, coef in enumerate(coefs)])

    if substance == "CO2":
        # PTM 1677-83
        _T = T / 1000
        coefs = (
            0.1047056,
            0.4234367,
            -0.3953561,
            0.2249471,
            -0.07729786,
            0.01462170,
            -0.001166819,
        )
        return 4187 * sum([coef * _T**i for i, coef in enumerate(coefs)])

    if substance == "H2O":
        # PTM 1677-83
        _T = T / 1000
        coefs = (
            0.4489375,
            -0.1088401,
            0.4027652,
            -0.2638393,
            0.07993751,
            -0.0115716,
            0.0006241951,
        )
        return 4187 * sum([coef * _T**i for i, coef in enumerate(coefs)])

    if substance == "O2":
        # PTM 1677-83
        _T = T / 1000
        coefs = (
            0.2083632,
            -0.0112279,
            0.2235868,
            -0.2732668,
            0.1461334,
            -0.03687021,
            0.003584204,
        )
        return 4187 * sum([coef * _T**i for i, coef in enumerate(coefs)])

    if substance == "H2":
        # PTM 1677-83
        _T = T / 1000
        coefs = (
            3.070881,
            2.230734,
            -4.909,
            5.321652,
            -2.756533,
            0.6851526,
            -0.06596988,
        )
        return 4187 * sum([coef * _T**i for i, coef in enumerate(coefs)])

    if substance == "N2":
        # https://www.highexpert.ru/content/gases/nitrogen.html
        return 1041 - 0.021 * T + 0.0003814 * T**2

    if substance == "Ar":  # TODO
        return 523

    if substance == "Ne":  # TODO
        return 1038

    if substance.upper() in ("C2H8N2", "KEROSENE", "TC-1", "КЕРОСИН", "ТС-1"):
        """Теплоемкость жидкого керосина"""
        return Cp_kerosene(T)

    print(
        Fore.RED
        + f"Not found substance {substance} in function {Cp.__name__}"
        + Fore.RESET
    )
    return nan
'''

'''
def Qa1(fuel) -> float:
    """Низшая теплота сгорания горючего при коэффициенте избытка окислителя = 1"""
    if fuel.upper() in (
        "C2H8N2",
        "KEROSENE",
        "TC1",
        "TC-1",
        "PETROL",
        "КЕРОСИН",
        "ТС1",
        "ТС-1",
        "БЕНЗИН",
    ):
        return 0.5 * (43_600 + 42_700) * 1000
    elif fuel.upper() in ("T-6", "T-8", "Т-6", "Т-8"):
        return 42_900 * 1000
    elif fuel.upper in ("ДИЗЕЛЬ", "DIESEL"):
        return nan
    elif fuel.upper in ("ПРИРОДНЫЙ ГАЗ", "ПРИРОДНЫЙ_ГАЗ"):
        return nan
    elif fuel.upper in ("КОКСОВЫЙ ГАЗ", "КОКСОВЫЙ_ГАЗ"):
        return nan
    else:
        print(
            Fore.RED + "not found fuel!" + " in function " + Qa1.__name__ + Fore.RESET
        )
        return nan
'''


def dynamic_exhaust_viscosity(temperature=nan, a_ox=nan) -> float:
    """Динамическая вязкость"""
    coefs = (+0.505, +4.849, -1.333, +0.229)
    t = temperature / 1_000
    return 10 ** (-5) * (sum(coefs[i] * t**i for i in range(len(coefs))) - 0.275 / a_ox)


def g_cool_ciam(temperature_input, temperature_output, temperature_lim) -> float:
    """Эмпирический относительный массовый расход на охлаждение
    по max температурам до и после КС и допустимой температуре,
    отнесенный к расходу на входе в горячую часть"""
    θ = (temperature_output - temperature_lim) / (
        temperature_output - temperature_input
    )
    g_cool = 0.059 * θ / (1 - 1.42 * θ)
    return g_cool if g_cool > 0 else 0


def g_cool_bmstu(temperature_max, temperature_lim=1000) -> float:
    """Эмпирический относительный массовый расход на охлаждение по max и допустимой температуре,
    отнесенный к расходу на входе в горячую часть"""
    coefs = (0.01, 0.09, 0.2, 0.16)
    dt = (temperature_max - temperature_lim) / 1_000
    g_cool = sum(coefs[i] * dt**i for i in range(len(coefs)))
    return g_cool if g_cool > 0 else 0

import numpy as np
import pytest
from numpy import isclose, isnan, nan

from thermodynamics import *

from .parameters import parameters as tdp


def test_gdf():
    # Проверяем корректный расчет для λ и k
    assert isclose(gdf("T", λ=0.5, k=1.4), 1 - 0.5**2 * ((1.4 - 1) / (1.4 + 1)))
    # Проверяем граничное значение λ = 0
    assert gdf("T", λ=0, k=1.4) == 1.0
    # Проверяем граничное значение λ = 1
    assert isclose(gdf("T", λ=1, k=1.4), 1 - 1**2 * ((1.4 - 1) / (1.4 + 1)))
    # Проверяем корректный расчет через T
    T_value = gdf("T", λ=0.5, k=1.4)
    expected_P = T_value ** (1.4 / (1.4 - 1))
    assert isclose(gdf("P", λ=0.5, k=1.4), expected_P)
    # Проверяем корректный расчет через T
    T_value = gdf("T", λ=0.5, k=1.4)
    expected_rho = T_value ** (1 / (1.4 - 1))
    assert isclose(gdf("ρ", λ=0.5, k=1.4), expected_rho)
    # Проверяем корректный расчет через ρ
    rho_value = gdf("ρ", λ=0.5, k=1.4)
    expected_G = ((1.4 + 1) / 2) ** (1 / (1.4 - 1)) * 0.5 * rho_value
    assert isclose(gdf("G", λ=0.5, k=1.4), expected_G)
    # Тесты для параметра "p" (или "mv")
    assert gdf("p", λ=1.0) == 2.0  # λ + 1/λ при λ=1 → 1 + 1 = 2
    assert gdf("p", λ=2.0) == 2.5  # 2 + 0.5 = 2.5
    # Тест на обработку некорректного параметра
    with pytest.raises(Exception) as excinfo:
        gdf("invalid_param")
    assert "parameter not in" in str(excinfo.value)
    # Тест на обработку nan значений (если требуется)
    assert isnan(gdf("T", λ=nan, k=1.4))
    assert isnan(gdf("T", λ=0.5, k=nan))


class TestAtmosphereStandard:
    """Тесты для функций стандартной атмосферы ГОСТ 4401-81"""

    # Тестовые данные: высота, ожидаемая температура (K), ожидаемое давление (Pa)
    TEST_DATA = (
        (0, 288.15, 101325),
        (5000, 255.65, 54019.9),
        (11000, 216.65, 22699.9),  # Граница тропопаузы
        (15000, 216.65, 12044.6),
        (20000, 216.65, 5474.89),
    )

    @pytest.mark.parametrize("height, expected_temp, _", TEST_DATA)
    def test_temperature_atmosphere_standard(self, height, expected_temp, _):
        """Тест расчета температуры стандартной атмосферы"""
        temp, units = temperature_atmosphere_standard(height)
        assert units == "K"
        assert temp == pytest.approx(expected_temp, rel=1e-3)
        assert isinstance(temp, float)

    @pytest.mark.parametrize("height, _, expected_pres", TEST_DATA)
    def test_pressure_atmosphere_standard(self, height, _, expected_pres):
        """Тест расчета давления стандартной атмосферы"""
        pres, units = pressure_atmosphere_standard(height)
        assert units == "Pa"
        assert pres == pytest.approx(expected_pres, rel=0.01)
        assert isinstance(pres, float)

    def test_atmosphere_standard_structure(self):
        """Тест структуры возвращаемого словаря"""
        result = atmosphere_standard(5000)
        assert isinstance(result, dict)
        assert set(result.keys()) == {tdp.T, tdp.P}
        assert all(isinstance(v, tuple) and len(v) == 2 for v in result.values())

    @pytest.mark.parametrize("height", [0, 5000, 11000, 20000])
    def test_atmosphere_standard_consistency(self, height):
        """Тест согласованности функций"""
        result = atmosphere_standard(height)
        temp_func, _ = temperature_atmosphere_standard(height)
        pres_func, _ = pressure_atmosphere_standard(height)

        assert result[tdp.T][0] == pytest.approx(temp_func)
        assert result[tdp.P][0] == pytest.approx(pres_func)

    def test_edge_cases(self):
        """Тест граничных случаев и исключений"""
        # Нечисловые значения
        with pytest.raises(TypeError):
            pressure_atmosphere_standard("1000")

    @pytest.mark.parametrize("height", np.linspace(0, 20000, 10))
    def test_pressure_temperature_relationship(self, height):
        """Тест соотношения давления и температуры"""
        temp, _ = temperature_atmosphere_standard(height)
        pres, _ = pressure_atmosphere_standard(height)

        if height < 11_000:
            # Проверка барометрической формулы для тропосферы
            expected_pres = 101325 * (temp / 288.15) ** 5.2533
            assert pres == pytest.approx(expected_pres, rel=1e-3)
        else:
            # Проверка экспоненциального закона для стратосферы
            expected_pres = 22699.9 * np.exp((11_000 - height) / 6318)
            assert pres == pytest.approx(expected_pres, rel=1e-3)


class TestAdiabaticIndex:
    """Тесты для расчета показателя адиабаты"""

    # Параметризованные тесты для нормальных случаев
    @pytest.mark.parametrize(
        "gas_const, cp, expected",
        [
            # Стандартные значения для воздуха
            (287.0, 1005.0, pytest.approx(1.4, rel=1e-3)),
            # Другие газы
            (297.0, 1040.0, pytest.approx(1.4, rel=1e-2)),
            (189.0, 1300.0, pytest.approx(1.17, rel=1e-2)),
            # Крайние случаи
            (100.0, 150.0, 3.0),  # γ = 150/(150-100) = 3
            (200.0, 400.0, 2.0),  # γ = 400/(400-200) = 2
        ],
    )
    def test_normal_cases(self, gas_const, cp, expected):
        """Тест корректных расчетов"""
        assert adiabatic_index(gas_const, cp) == expected

    # Тест для особого случая (cp == gas_const)
    def test_equal_values(self):
        """Тест случая, когда cp == gas_const"""
        assert adiabatic_index(300.0, 300.0) is nan

    # Тесты для обработки ошибок
    @pytest.mark.parametrize(
        "gas_const, cp",
        [
            (0.0, 100.0),  # Нулевой gas_const
            (100.0, 0.0),  # Нулевой cp
            (100.0, 99.0),  # cp < gas_const
            (-100.0, 200.0),  # Отрицательный gas_const
            (100.0, -200.0),  # Отрицательный cp
        ],
    )
    def test_invalid_values(self, gas_const, cp):
        """Тест некорректных входных значений"""
        result = adiabatic_index(gas_const, cp)
        assert isnan(result) or isinstance(result, (float, np.number))

    # Тесты для типов данных
    @pytest.mark.parametrize(
        "gas_const, cp",
        [
            ("300", 400.0),  # Строка вместо числа
            (300.0, None),  # None вместо числаs
            ([300], 400.0),  # Список вместо числа
        ],
    )
    def test_invalid_types(self, gas_const, cp):
        """Тест нечисловых входных данных"""
        with pytest.raises(TypeError):
            adiabatic_index(gas_const, cp)

    # Тест возвращаемого типа
    def test_return_type(self):
        """Проверка что возвращается float"""
        result = adiabatic_index(287.0, 1005.0)
        assert isinstance(result, float)

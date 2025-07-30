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

        if height < 11000:
            # Проверка барометрической формулы для тропосферы
            expected_pres = 101325 * (temp / 288.15) ** 5.2533
            assert pres == pytest.approx(expected_pres, rel=1e-3)
        else:
            # Проверка экспоненциального закона для стратосферы
            expected_pres = 22699.9 * np.exp((11000 - height) / 6318)
            assert pres == pytest.approx(expected_pres, rel=1e-3)

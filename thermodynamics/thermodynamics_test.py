import pytest
from numpy import isclose, isnan, nan

from thermodynamics import *


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

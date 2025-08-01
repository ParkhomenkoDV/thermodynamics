import numpy as np
import pytest
from numpy import isclose, isnan, nan

from thermodynamics import *

from .parameters import parameters as tdp


class TestGDF:
    """Тесты для функции gas_dynamic_function()"""

    # Общие параметры для тестов
    TEST_LAMBDA = 0.5
    TEST_K = 1.4

    # Тесты для параметра "T" (температура)
    def test_T_calculation(self):
        expected = 1 - self.TEST_LAMBDA**2 * ((self.TEST_K - 1) / (self.TEST_K + 1))
        result = gdf(
            "T", equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        assert isclose(result, expected)

    @pytest.mark.parametrize(
        "lambda_val, expected",
        [
            (0.0, 1.0),  # При λ=0 должно быть 1
            (1.0, 1 - 1**2 * ((1.4 - 1) / (1.4 + 1))),  # Граничное значение λ=1
        ],
    )
    def test_T_boundary_values(self, lambda_val, expected):
        result = gdf("T", equivalent_speed=lambda_val, adiabatic_index=1.4)
        assert isclose(result, expected)

    # Тесты для параметра "P" (давление)
    def test_P_calculation(self):
        T_value = gdf(
            "T", equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        expected = T_value ** (self.TEST_K / (self.TEST_K - 1))
        result = gdf(
            "P", equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        assert isclose(result, expected)

    # Тесты для параметра "D" (плотность)
    def test_D_calculation(self):
        T_value = gdf(
            "T", equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        expected = T_value ** (1 / (self.TEST_K - 1))
        result = gdf(
            "D", equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        assert isclose(result, expected)

    # Тесты для параметров "G" и "MF" (массовый расход)
    @pytest.mark.parametrize("param", ["G", "MF"])
    def test_mass_flow_calculation(self, param):
        D_value = gdf(
            "D", equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        expected = (
            ((self.TEST_K + 1) / 2) ** (1 / (self.TEST_K - 1))
            * self.TEST_LAMBDA
            * D_value
        )
        result = gdf(
            param, equivalent_speed=self.TEST_LAMBDA, adiabatic_index=self.TEST_K
        )
        assert isclose(result, expected)

    # Тесты для параметров "I" и "MV" (импульс)
    @pytest.mark.parametrize(
        "param, lambda_val, expected",
        [
            ("I", 1.0, 2.0),  # λ + 1/λ при λ=1 → 2
            ("MV", 2.0, 2.5),  # 2 + 1/2 = 2.5
            ("I", 0.5, 2.5),  # 0.5 + 1/0.5 = 2.5
        ],
    )
    def test_momentum_calculation(self, param, lambda_val, expected):
        result = gdf(param, equivalent_speed=lambda_val)
        assert isclose(result, expected)

    # Тесты на обработку ошибок
    def test_invalid_parameter(self):
        with pytest.raises(ValueError) as excinfo:
            gdf("invalid", equivalent_speed=0.5)
        assert 'not in ("T", "P", "D", "G", "MF", "I", "MV")' in str(excinfo.value)

    def test_non_numeric_lambda(self):
        with pytest.raises((AssertionError, TypeError)):
            gdf("T", equivalent_speed="not_a_number", adiabatic_index=1.4)

    def test_non_numeric_k_for_T(self):
        with pytest.raises((AssertionError, TypeError)):
            gdf("T", equivalent_speed=0.5, adiabatic_index="not_a_number")

    def test_missing_k_for_T(self):
        with pytest.raises((AssertionError, TypeError)):
            gdf("T", equivalent_speed=0.5)

    # Тесты на обработку специальных значений
    def test_nan_lambda(self):
        assert isnan(gdf("T", equivalent_speed=np.nan, adiabatic_index=1.4))

    def test_nan_k(self):
        assert isnan(gdf("T", equivalent_speed=0.5, adiabatic_index=np.nan))

    # Тесты на регистронезависимость
    def test_case_insensitivity(self):
        assert gdf("t", equivalent_speed=0.5, adiabatic_index=1.4) == gdf(
            "T", equivalent_speed=0.5, adiabatic_index=1.4
        )
        assert gdf("g", equivalent_speed=0.5, adiabatic_index=1.4) == gdf(
            "G", equivalent_speed=0.5, adiabatic_index=1.4
        )

    # Тест на очень большое значение λ
    def test_large_lambda(self):
        """Проверка, что функция не ломается на больших λ"""
        result = gdf("I", equivalent_speed=1e6)
        assert isclose(result, 1e6 + 1e-6)  # λ + 1/λ ≈ λ при больших λ


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


class TestGasConst:
    """Тесты для функции gas_const()"""

    # Тесты для воздуха
    def test_air_english(self):
        assert gas_const("air") == 287.14

    def test_air_russian(self):
        assert gas_const("ВОЗДУХ") == 287.14

    def test_air_case_insensitive(self):
        assert gas_const("AiR") == 287.14

    # Тесты для продуктов сгорания (EXHAUST)
    @pytest.mark.parametrize(
        "fuel, alpha, expected",
        [
            ("C2H8N2", 1.0, 288.88712718),
            ("kerosene", 2.0, 288.54104318),
            ("КЕРОСИН", 5.0, 288.33373438),
            ("T1", 1.0, 287.971694324),
            ("РЕАКТИВНОЕ ТОПЛИВО", 2.0, 288.078848932),
            ("TC-1", 1.0, 288.386957775),
            ("ТС1", 5.0, 288.22958048),
            ("DIESEL", 1.0, 287.365782577),
            ("ДТ", 2.0, 287.7725053385),
            ("SOLAR", 1.0, 286.740658766),
            ("СОЛЯРКА", 5.0, 287.8864998532),
            ("MAZUT", 1.0, 285.909107708),
            ("ПРИРОДНЫЙ ГАЗ", 1.0, 302.23684704),
            ("КОКСОВЫЙ ГАЗ", 1.0, 308.63228928),
            ("BIOGAS", 1.0, 292.107038145),
        ],
    )
    def test_exhaust_with_valid_fuels(self, fuel, alpha, expected):
        assert gas_const("exhaust", excess_oxidizing=alpha, fuel=fuel) == pytest.approx(
            expected
        )

    # Тесты на ошибки
    def test_invalid_substance(self):
        with pytest.raises(ValueError):
            gas_const("invalid_substance")

    def test_exhaust_without_fuel(self):
        with pytest.raises(ValueError):
            gas_const("exhaust")

    def test_invalid_fuel(self):
        with pytest.raises(ValueError):
            gas_const("exhaust", excess_oxidizing=1.0, fuel="invalid_fuel")

    def test_non_numeric_alpha(self):
        with pytest.raises(TypeError):
            gas_const("exhaust", excess_oxidizing="not_a_number", fuel="kerosene")

    def test_negative_alpha(self):
        with pytest.raises((AssertionError, ValueError)):
            gas_const("exhaust", excess_oxidizing=-1.0, fuel="kerosene")

    def test_zero_alpha(self):
        with pytest.raises((AssertionError, ValueError)):
            gas_const("exhaust", excess_oxidizing=0.0, fuel="kerosene")

    def test_nan_alpha(self):
        with pytest.raises((AssertionError, ValueError)):
            gas_const("exhaust", excess_oxidizing=np.nan, fuel="kerosene")

    # Тесты на типы аргументов
    def test_non_string_substance(self):
        with pytest.raises((AssertionError, ValueError)):
            gas_const(123)

    def test_non_string_fuel(self):
        with pytest.raises((AssertionError, ValueError)):
            gas_const("exhaust", excess_oxidizing=1.0, fuel=123)

    def test_very_large_alpha(self):
        # Проверяем что не возникает ошибок при большом alpha
        result = gas_const("exhaust", excess_oxidizing=1e10, fuel="kerosene")
        assert result == pytest.approx(288.1954313)  # Второе слагаемое стремится к 0


class TestStoichiometry:
    """Тесты для функции stoichiometry()"""

    # Тесты для керосиновой группы
    @pytest.mark.parametrize(
        "fuel",
        [
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
        ],
    )
    def test_kerosene_group(self, fuel):
        """Проверка стехиометрического коэффициента для керосиновой группы"""
        assert stoichiometry(fuel) == pytest.approx(14.61)

    # Тесты для бензиновой группы
    @pytest.mark.parametrize("fuel", ["PETROL", "GASOLINE", "БЕНЗИН"])
    def test_petrol_group(self, fuel):
        assert stoichiometry(fuel) == pytest.approx(14.91)

    # Тесты для дизельной группы
    @pytest.mark.parametrize(
        "fuel",
        [
            "SOLAR",
            "SOLAR OIL",
            "SOLAR_OIL",
            "СОЛЯРКА",
            "СОЛЯРОВОЕ МАСЛО",
            "СОЛЯРОВОЕ_МАСЛО",
            "DIESEL",
            "ДИЗЕЛЬ",
        ],
    )
    def test_diesel_group(self, fuel):
        assert stoichiometry(fuel) == pytest.approx(14.35)

    # Тесты для мазутной группы
    @pytest.mark.parametrize("fuel", ["MAZUT", "МАЗУТ", "Ф5", "Ф12"])
    def test_mazut_group(self, fuel):
        assert stoichiometry(fuel) == pytest.approx(13.31)

    # Тесты для природного газа
    def test_natural_gas(self):
        expected = np.mean([15.83, 13.69, 16.84, 16.51, 15.96, 16.34, 16.85, 15.93])
        assert stoichiometry("ПРИРОДНЫЙ ГАЗ") == pytest.approx(expected)
        assert stoichiometry("ПРИРОДНЫЙ_ГАЗ") == pytest.approx(expected)

    # Тесты для коксового газа
    def test_coke_gas(self):
        assert stoichiometry("КОКСОВЫЙ ГАЗ") == pytest.approx(9.908)
        assert stoichiometry("КОКСОВЫЙ_ГАЗ") == pytest.approx(9.908)

    # Тесты на обработку ошибок
    def test_invalid_fuel_type(self):
        """Проверка TypeError при передаче не строкового аргумента"""
        with pytest.raises((AssertionError, TypeError)):
            stoichiometry(123)
        with pytest.raises((AssertionError, TypeError)):
            stoichiometry(None)

    def test_unknown_fuel(self):
        """Проверка ValueError при передаче неизвестного топлива"""
        with pytest.raises(ValueError):
            stoichiometry("unknown_fuel")
        with pytest.raises(ValueError):
            stoichiometry("")

    # Тесты на регистронезависимость
    def test_case_insensitivity(self):
        """Проверка работы функции с разным регистром"""
        assert stoichiometry("kerosene") == stoichiometry("KEROSENE")
        assert stoichiometry("бензин") == stoichiometry("БЕНЗИН")
        assert stoichiometry("t-1") == stoichiometry("T-1")

    # Тесты на пробелы и подчеркивания
    def test_space_underscore_equivalence(self):
        """Проверка эквивалентности вариантов с пробелами и подчеркиваниями"""
        assert stoichiometry("SOLAR OIL") == stoichiometry("SOLAR_OIL")
        assert stoichiometry("ПРИРОДНЫЙ ГАЗ") == stoichiometry("ПРИРОДНЫЙ_ГАЗ")
        assert stoichiometry("СОЛЯРОВОЕ МАСЛО") == stoichiometry("СОЛЯРОВОЕ_МАСЛО")

    # Тест на пустую строку
    def test_empty_string(self):
        """Проверка обработки пустой строки"""
        with pytest.raises(ValueError):
            stoichiometry("")

# simulation/grid.py

import numpy as np


class Grid:
    """
    Models electricity grid with:
    - Time-of-use tariffs
    - Carbon intensity by hour
    - Solar generation profile
    - Grid capacity constraints (used by Task 3)
    """
    def __init__(self):
        self.tariff_schedule  = self._build_tariff()
        self.carbon_schedule  = self._build_carbon()
        self.solar_schedule   = self._build_solar()
        self.max_capacity_kw  = 50.0   # hostel grid limit

    # ------------------------------------------------------------------
    def _build_tariff(self):
        tariff = {}
        for h in range(24):
            if 9 <= h <= 12 or 18 <= h <= 22:
                tariff[h] = 8.5    # peak — expensive
            elif 0 <= h <= 5:
                tariff[h] = 4.0    # night — cheap
            else:
                tariff[h] = 6.0    # normal
        return tariff

    def _build_carbon(self):
        carbon = {}
        for h in range(24):
            if 9 <= h <= 12 or 18 <= h <= 22:
                carbon[h] = 0.82   # peak = thermal plants running
            elif 0 <= h <= 5:
                carbon[h] = 0.45   # night = more renewables
            else:
                carbon[h] = 0.63
        return carbon

    def _build_solar(self):
        """Solar output profile (0-1 multiplier)"""
        solar = {}
        for h in range(24):
            if 6 <= h <= 18:
                # Bell curve peaking at noon
                peak_dist = abs(h - 12)
                solar[h]  = max(0.0, 1.0 - peak_dist / 7.0)
            else:
                solar[h] = 0.0
        return solar

    # ------------------------------------------------------------------
    def get_tariff(self, hour):
        return self.tariff_schedule.get(hour % 24, 6.0)

    def get_carbon_rate(self, hour):
        return self.carbon_schedule.get(hour % 24, 0.63)

    def get_solar_output(self, hour):
        return self.solar_schedule.get(hour % 24, 0.0)

    def get_cost(self, power_kw, hour):
        """Cost for using power_kw for 1 hour"""
        return round(np.clip(power_kw, 0, self.max_capacity_kw)
                     * self.get_tariff(hour), 4)

    def get_carbon(self, power_kw, hour):
        """Carbon for using power_kw for 1 hour"""
        return round(np.clip(power_kw, 0, self.max_capacity_kw)
                     * self.get_carbon_rate(hour), 4)

    def is_peak_hour(self, hour):
        return 9 <= (hour % 24) <= 12 or 18 <= (hour % 24) <= 22

    def summary(self, hour):
        print(f"\n🔌 Grid Status at Hour {hour:02d}")
        print(f"   Tariff        : Rs {self.get_tariff(hour)}/kWh")
        print(f"   Carbon Rate   : {self.get_carbon_rate(hour)} gCO2/kWh")
        print(f"   Solar Output  : {self.get_solar_output(hour):.2f}")
        print(f"   Peak Hour     : {'Yes ⚠️' if self.is_peak_hour(hour) else 'No'}")
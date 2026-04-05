# simulation/appliances.py

import random
import numpy as np


class Appliance:
    def __init__(self, name, power_kw, deferrable=False):
        self.name        = name
        self.power_kw    = power_kw
        self.deferrable  = deferrable
        self.is_on       = False
        self.deferred    = False

    def turn_on(self):
        self.is_on    = True
        self.deferred = False

    def turn_off(self):
        self.is_on = False

    def defer(self):
        """Shift to off-peak — used by Task 2 & 3 enforcement"""
        if self.deferrable:
            self.is_on    = False
            self.deferred = True

    def get_power(self):
        return self.power_kw if self.is_on else 0.0

    def __repr__(self):
        status = ("ON" if self.is_on
                  else "DEFERRED" if self.deferred
                  else "OFF")
        return f"{self.name:<20} | {self.power_kw} kW | {status}"


class ApplianceManager:
    def __init__(self):
        self.appliances = [
            Appliance("Washing Machine",  power_kw=2.0, deferrable=True),
            Appliance("Water Heater",     power_kw=3.0, deferrable=True),
            Appliance("Common Area AC",   power_kw=2.5, deferrable=False),
            Appliance("Gym Equipment",    power_kw=1.5, deferrable=True),
            Appliance("Kitchen",          power_kw=2.0, deferrable=False),
            Appliance("Induction Stove",  power_kw=2.0, deferrable=True),
            Appliance("Study Room Lights",power_kw=0.5, deferrable=False),
            Appliance("EV Charger",       power_kw=3.5, deferrable=True),
        ]

    def total_power(self):
        return round(sum(a.get_power() for a in self.appliances), 3)

    def defer_all_deferrable(self):
        """Called during peak hours or by Task 2/3 enforcement action"""
        deferred = 0
        for a in self.appliances:
            if a.deferrable and a.is_on:
                a.defer()
                deferred += 1
        return deferred

    def turn_on_all(self):
        for a in self.appliances:
            a.turn_on()

    def turn_off_all(self):
        for a in self.appliances:
            a.turn_off()

    def get_deferrable_load(self):
        """Total power that CAN be deferred — used by Task 3 battery logic"""
        return sum(
            a.power_kw for a in self.appliances
            if a.deferrable and a.is_on
        )

    def get_status(self):
        return {a.name: a.is_on for a in self.appliances}

    def summary(self):
        print("\n⚡ Appliance Status")
        print("-" * 45)
        for a in self.appliances:
            print(f"   {a}")
        print("-" * 45)
        print(f"   Total Load : {self.total_power()} kW")
        print(f"   Deferrable : {self.get_deferrable_load()} kW")
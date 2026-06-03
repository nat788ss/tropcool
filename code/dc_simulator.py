#!/usr/bin/env python3
"""Data center mechanical cooling plant — hourly steady-state snapshot.

Models a **configurable water-cooled chiller plant** sized for roughly **5–50 MW** IT
(nameplate via ``chiller_rated_capacity_mw``). End-use:

**Inputs** (``simulate_hour``)

- ``outdoor_temp_c`` — outdoor dry-bulb °C (condenser / tower entering air).
- ``outdoor_relative_humidity_percent`` — RH (%); used only if ``wet_bulb_c`` is omitted.
- ``wet_bulb_c`` (optional) — outdoor wet-bulb °C; if provided, RH is ignored for psychrometrics.
- ``it_load_mw`` — IT heat at the room/evaporator (MW).
- ``chiller_setpoint_c`` — chilled-water supply setpoint °C.
- ``crah_fan_speed``, ``cooling_tower_fan_speed`` — normalized speeds in ``[0, 1]``.

**Outputs** (``SimulationResult``)

- ``rack_inlet_temperature_c`` / alias ``rack_inlet_temp`` — estimated supply air at rack.
- ``total_cooling_power_kw`` — chiller (compressor + aux), tower fan, CRAH fan, pumps, overhead.
- ``water_consumption_liters`` — tower makeup water (L/h).
- ``pue`` — ``(IT_kW + cooling_kW) / IT_kW``.

**Physics (compact)**

- **Chillers** — :math:`\\mathrm{COP} = \\mathrm{COP}_{ref} \\times f_{amb}(T_{out}) \\times f_{PLR}(PLR)`.
  ``reference_cop`` defaults to **6.0** at **AHRI-style** design outdoor temperature
  (``design_outdoor_temp_c``). Ambient factor is **linear in**
  :math:`T_{out} - T_{design}` (bounded): COP rises when outdoor temperature falls below design
  condenser conditions and falls when it rises (Carrier/Trane-style trend). Part-load factor
  captures turndown inefficiency at low PLR.

- **Cooling tower** — evaporative effectiveness :math:`\\eta` increases with **wet-bulb
  depression** :math:`T_{db} - T_{wb}`. Condenser heat rejection
  :math:`Q_{rej} = Q_{evap} + W_{comp}`. Makeup water
  :math:`\\dot{m} = Q_{rej} / (\\eta \\, h_{fg})` with :math:`h_{fg} \\approx 2450` kJ/kg.

- **CRAH / tower fans** — **affinity law**: :math:`P \\propto \\omega^3` (speed cubed).

- **Optional chilled-water TES** — ``ChilledWaterStorage`` can discharge to shave instantaneous
  chiller load (minimal stateful model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from weather_utils import wet_bulb_temperature_celsius

# Latent heat of vaporization for water (kJ/kg), weak function of T; use mid-range value.
_LATENT_VAP_KJ_PER_KG = 2450.0

# Typical nameplate range for this simulator (MW IT / chiller capacity); not hard-enforced.
FACILITY_IT_LOAD_MW_MIN = 5.0
FACILITY_IT_LOAD_MW_MAX = 50.0


@dataclass(frozen=True)
class SimulationResult:
    """One hour (or one timestep) energy and comfort indices."""

    rack_inlet_temperature_c: float
    total_cooling_power_kw: float
    water_consumption_liters: float
    pue: float
    chiller_compressor_power_kw: float
    chiller_auxiliary_power_kw: float
    cooling_tower_fan_power_kw: float
    crah_fan_power_kw: float
    pump_and_parasitic_kw: float
    cop_effective: float
    part_load_ratio: float
    wet_bulb_temperature_c: float

    @property
    def rack_inlet_temp(self) -> float:
        """Alias matching prompt naming (°C)."""
        return self.rack_inlet_temperature_c


@dataclass
class ChilledWaterStorage:
    """Simple chilled-water tank: SOC in kWh_thermal; discharge shaves instantaneous IT load."""

    capacity_kwh: float = 8000.0
    soc_kwh: float = field(default_factory=lambda: 4000.0)
    max_discharge_kw: float = 2500.0
    max_charge_kw: float = 2000.0

    def reset(self, soc_fraction: float = 0.5) -> None:
        self.soc_kwh = max(0.0, min(self.capacity_kwh, self.capacity_kwh * soc_fraction))


class DataCenterSimulator:
    """Configurable data-center cooling simulator (chillers, tower, CRAH, optional TES)."""

    def __init__(
        self,
        *,
        chiller_rated_capacity_mw: float = 12.5,
        reference_cop: float = 6.0,
        design_outdoor_temp_c: float = 27.5,
        outdoor_cop_penalty_per_k: float = 0.0025,
        plr_low_threshold: float = 0.35,
        plr_low_penalty: float = 0.55,
        crah_fan_rated_power_kw: float = 300.0,
        cooling_tower_fan_rated_power_kw: float = 205.0,
        chiller_auxiliary_fraction: float = 0.13,
        pump_kw_per_kw_load: float = 0.078,
        facility_overhead_fraction_of_it: float = 0.32,
        tower_effectiveness_base: float = 0.42,
        tower_effectiveness_per_k_depression: float = 0.038,
        tower_effectiveness_max: float = 0.88,
        rack_coil_approach_c: float = 3.8,
        rack_mixing_penalty_c: float = 5.2,
        thermal_storage: ChilledWaterStorage | None = None,
        cop_min: float = 2.2,
        cop_max: float = 7.5,
    ) -> None:
        self._rated_mw = float(chiller_rated_capacity_mw)
        self._cop_ref = float(reference_cop)
        self._t_design = float(design_outdoor_temp_c)
        self._k_out = float(outdoor_cop_penalty_per_k)
        self._plr_low_t = float(plr_low_threshold)
        self._plr_low_pen = float(plr_low_penalty)
        self._crah_rated = float(crah_fan_rated_power_kw)
        self._ct_rated = float(cooling_tower_fan_rated_power_kw)
        self._ch_aux = float(chiller_auxiliary_fraction)
        self._pump_per_kw = float(pump_kw_per_kw_load)
        self._fac_frac = float(facility_overhead_fraction_of_it)
        self._eta0 = float(tower_effectiveness_base)
        self._eta_wb = float(tower_effectiveness_per_k_depression)
        self._eta_max = float(tower_effectiveness_max)
        self._rack_app = float(rack_coil_approach_c)
        self._rack_mix = float(rack_mixing_penalty_c)
        self._storage = thermal_storage
        self._cop_min = float(cop_min)
        self._cop_max = float(cop_max)

    @property
    def thermal_storage_enabled(self) -> bool:
        return self._storage is not None

    def _cop(self, outdoor_temp_c: float, plr: float) -> float:
        """Effective COP: reference_COP × f_amb(outdoor) × f_PLR, with overload clip."""
        plr_raw = max(1e-6, min(2.0, plr))
        plr_c = min(1.0, plr_raw)
        # Ambient: COP rises below ``design_outdoor_temp_c`` (lower condenser lift) and
        # falls above it — symmetric linear model around AHRI-style design (Carrier/Trane trend).
        f_amb = 1.0 - self._k_out * (outdoor_temp_c - self._t_design)
        f_amb = max(0.52, min(1.18, f_amb))
        overload_penalty = 1.0 - 0.04 * max(0.0, plr_raw - 1.0)
        overload_penalty = max(0.88, min(1.0, overload_penalty))
        # Part load: mild penalty near full load; stronger penalty deep turndown (IPLV-like).
        if plr_c >= self._plr_low_t:
            f_plr = 1.0 - 0.12 * (1.0 - plr_c) ** 1.25
        else:
            f_plr = 1.0 - self._plr_low_pen * (self._plr_low_t - plr_c) / self._plr_low_t
        f_plr = max(0.45, min(1.05, f_plr))
        cop = self._cop_ref * f_amb * f_plr * overload_penalty
        return max(self._cop_min, min(self._cop_max, cop))

    def _tower_effectiveness(self, wb_depression_c: float) -> float:
        dep = max(0.5, wb_depression_c)
        eta = self._eta0 + self._eta_wb * dep
        return max(0.15, min(self._eta_max, eta))

    def simulate_hour(
        self,
        outdoor_temp_c: float,
        outdoor_relative_humidity_percent: float,
        it_load_mw: float,
        chiller_setpoint_c: float,
        crah_fan_speed: float,
        cooling_tower_fan_speed: float,
        *,
        wet_bulb_c: float | None = None,
    ) -> SimulationResult:
        """Simulate one timestep (e.g. one hour).

        If ``wet_bulb_c`` is omitted, wet bulb is computed from dry bulb + RH (Stull 2011).
        """
        if it_load_mw < 0:
            raise ValueError("it_load_mw must be non-negative")
        for name, x in (
            ("crah_fan_speed", crah_fan_speed),
            ("cooling_tower_fan_speed", cooling_tower_fan_speed),
        ):
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {x}")

        it_kw = it_load_mw * 1000.0
        if wet_bulb_c is None:
            wb = wet_bulb_temperature_celsius(
                outdoor_temp_c, outdoor_relative_humidity_percent
            )
        else:
            wb = float(wet_bulb_c)
        # Psychrometric sanity: wet bulb cannot exceed dry bulb.
        wb = min(wb, outdoor_temp_c)
        wb_dep = max(0.1, outdoor_temp_c - wb)

        q_storage_kw = 0.0
        if self._storage is not None and it_kw > 0:
            shave = min(
                0.22 * it_kw,
                self._storage.max_discharge_kw,
                self._storage.soc_kwh,
            )
            q_storage_kw = max(0.0, shave)
            self._storage.soc_kwh = max(0.0, self._storage.soc_kwh - q_storage_kw)

        q_evap_kw = max(0.0, it_kw - q_storage_kw)
        rated_kw = self._rated_mw * 1000.0
        plr = q_evap_kw / rated_kw if rated_kw > 0 else 0.0

        cop = self._cop(outdoor_temp_c, plr)
        w_comp_kw = q_evap_kw / cop if q_evap_kw > 0 else 0.0
        w_chiller_aux_kw = w_comp_kw * self._ch_aux
        w_chiller_total_kw = w_comp_kw + w_chiller_aux_kw

        q_reject_kw = q_evap_kw + w_comp_kw
        eta_evap = self._tower_effectiveness(wb_dep)
        m_dot_kg_s = q_reject_kw / (eta_evap * _LATENT_VAP_KJ_PER_KG)
        water_l_per_h = m_dot_kg_s * 3600.0

        crah_kw = self._crah_rated * (crah_fan_speed**3)
        ct_kw = self._ct_rated * (cooling_tower_fan_speed**3)
        pump_kw = self._pump_per_kw * it_kw
        overhead_kw = self._fac_frac * it_kw

        total_kw = w_chiller_total_kw + ct_kw + crah_kw + pump_kw + overhead_kw
        pue = (it_kw + total_kw) / it_kw if it_kw > 0 else float("nan")

        mix = max(0.0, 1.0 - crah_fan_speed)
        rack_in = chiller_setpoint_c + self._rack_app + self._rack_mix * mix

        return SimulationResult(
            rack_inlet_temperature_c=rack_in,
            total_cooling_power_kw=total_kw,
            water_consumption_liters=water_l_per_h,
            pue=pue,
            chiller_compressor_power_kw=w_comp_kw,
            chiller_auxiliary_power_kw=w_chiller_aux_kw,
            cooling_tower_fan_power_kw=ct_kw,
            crah_fan_power_kw=crah_kw,
            pump_and_parasitic_kw=pump_kw + overhead_kw,
            cop_effective=cop,
            part_load_ratio=plr,
            wet_bulb_temperature_c=wb,
        )


def _default_test() -> None:
    sim = DataCenterSimulator(thermal_storage=None)
    r = sim.simulate_hour(
        outdoor_temp_c=32.0,
        outdoor_relative_humidity_percent=80.0,
        it_load_mw=10.0,
        chiller_setpoint_c=7.0,
        crah_fan_speed=0.7,
        cooling_tower_fan_speed=0.7,
    )
    print("Baseline (no TES)")
    print(f"  PUE={r.pue:.3f}  total_cooling_kw={r.total_cooling_power_kw:.1f}")
    print(f"  water_L/h={r.water_consumption_liters:.0f}  rack_inlet_C={r.rack_inlet_temperature_c:.2f}")
    print(f"  COP_eff={r.cop_effective:.3f}  PLR={r.part_load_ratio:.3f}  Twb={r.wet_bulb_temperature_c:.2f}")
    low, high = 1.6, 1.8
    assert low <= r.pue <= high, f"PUE {r.pue} not in [{low}, {high}] — retune plant constants."

    r_wb = sim.simulate_hour(
        32.0,
        80.0,
        10.0,
        7.0,
        0.7,
        0.7,
        wet_bulb_c=28.5,
    )
    assert abs(r_wb.wet_bulb_temperature_c - 28.5) < 1e-6

    sim_tes = DataCenterSimulator(thermal_storage=ChilledWaterStorage())
    r2 = sim_tes.simulate_hour(
        outdoor_temp_c=32.0,
        outdoor_relative_humidity_percent=80.0,
        it_load_mw=10.0,
        chiller_setpoint_c=7.0,
        crah_fan_speed=0.7,
        cooling_tower_fan_speed=0.7,
    )
    print("With chilled-water storage (load shave)")
    print(f"  PUE={r2.pue:.3f}  storage_SOC_kWh={sim_tes._storage.soc_kwh:.0f}")


if __name__ == "__main__":
    _default_test()

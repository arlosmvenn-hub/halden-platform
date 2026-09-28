# HX-4410 Differential Temperature Sensor — Installation Guide

Document HI-MAN-HX4410, revision B. Applies to firmware 3.x.

## 1 Overview

The HX-4410 measures the temperature difference across a heat exchanger using two matched platinum resistance elements. It replaces the discontinued HX-441 temperature switch and reports a continuous 4–20 mA signal instead of a switched contact.

## 2 Specifications

| Parameter | Value |
|---|---|
| Accuracy | ±0.1 °C |
| Response time | under 2 seconds |
| Range | −50 to 250 °C |
| Output | 4–20 mA with HART 7 |

## 3 Installation

### 3.1 Thermowells

Mount each HX-4410 probe in a thermowell. Never insert a probe directly into a pressurized line; removing it later would release process fluid.

### 3.2 Matching probes

The two probes are calibrated as a pair. Keep them together and install the probe marked A on the inlet side.

## 4 Firmware

Firmware 3.2 fixes a drift issue that appeared after 90 days of continuous operation. Units shipped before 2025 should be updated using the Halden device tool.

## 5 Safety

Apply lockout/tagout before removing a probe from service. Thermowells may be hot; wear heat-resistant gloves.

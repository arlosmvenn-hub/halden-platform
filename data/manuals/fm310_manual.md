# FM-310 Electromagnetic Flow Meter — Installation and Operation Manual

Document HI-MAN-FM310, revision C. Applies to firmware 2.x.

## 1 Overview

The FM-310 measures the volumetric flow of electrically conductive liquids using Faraday's law of induction. A magnetic field is applied across the pipe, and the voltage induced in the moving liquid is measured by two electrodes. The meter has no moving parts and causes no pressure drop. It cannot measure hydrocarbons, gases, or deionized water, because their conductivity is too low.

## 2 Specifications

| Parameter | Value |
|---|---|
| Nominal sizes | DN15 to DN600 |
| Accuracy | ±0.3% of reading |
| Minimum conductivity | 5 µS/cm |
| Fluid temperature | up to 150 °C with PTFE liner, 80 °C with rubber liner |
| Output | 4–20 mA, pulse, and Modbus RTU |
| Supply voltage | 24 V DC or 100–240 V AC |

## 3 Installation

### 3.1 Straight runs

Install the FM-310 with at least ten pipe diameters of straight pipe upstream and five downstream. Elbows, valves, and pumps closer than this distort the flow profile and reduce accuracy.

### 3.2 Full pipe

The pipe must remain completely full of liquid at the sensor. Install the meter in a rising pipe section or at a low point; never at the top of a pipe loop.

### 3.3 Grounding

Ground the meter body and both flanges to the process liquid using grounding rings or the built-in grounding electrode. Poor grounding is the most common cause of unstable readings.

## 4 Configuration

Set the pipe size, flow unit, and full-scale flow using the local display or Modbus. The low-flow cutoff suppresses readings below 1% of full scale by default.

## 5 Maintenance

### 5.1 Electrode cleaning

Scale and grease deposits on the electrodes reduce the measured signal. If the meter reads zero or unusually low flow while liquid is moving, remove the meter and clean the electrodes with a soft brush.

### 5.2 Gasket replacement

Replace the flange gaskets whenever the meter is removed from the line. Reusing gaskets causes leaks.

## 6 Troubleshooting

### 6.1 Error codes

| Code | Meaning | Action |
|---|---|---|
| E-05 | Coil circuit open | Check coil cable connection; contact service if it persists |
| E-12 | Signal saturation | Check grounding; check for electrical interference |
| E-17 | Empty pipe detected | Ensure the pipe is full at the sensor; check installation position |
| E-24 | Totalizer overflow | Reset the totalizer |

## 7 Safety

Isolate and drain the line before removing the meter. Hot liquids can cause burns.

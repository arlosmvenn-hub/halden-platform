# PX-200 Pressure Transmitter — Installation and Operation Manual

Document HI-MAN-PX200, revision F. Applies to firmware 4.x.

## 1 Overview

The PX-200 is a two-wire pressure transmitter for gauge and absolute measurement of gases, steam, and liquids. It converts process pressure into a 4–20 mA signal with superimposed HART digital communication. The sensing element is a 316L stainless steel diaphragm backed by a piezoresistive silicon cell. Typical applications include refinery process lines, compressor discharge monitoring, and water treatment filtration skids.

## 2 Specifications

| Parameter | Value |
|---|---|
| Measuring ranges | 0–1 bar up to 0–400 bar |
| Accuracy | ±0.075% of calibrated span |
| Long-term stability | ±0.1% of upper range limit per 5 years |
| Supply voltage | 10.5–30 V DC |
| Output | 4–20 mA with HART 7 |
| Process temperature | −40 to 120 °C |
| Ambient temperature | −40 to 85 °C |
| Wetted materials | 316L stainless steel, optional Hastelloy C-276 |
| Ingress protection | IP66/IP68 |

## 3 Installation

### 3.1 Mounting

Mount the transmitter with the process connection facing downward for liquid service and upward for gas service, so that condensate or gas bubbles drain away from the diaphragm. Use a two-valve or three-valve manifold to allow isolation and zeroing without removing the transmitter.

### 3.2 Electrical connection

Connect the loop with shielded twisted-pair cable. Ground the shield at the control-system end only. HART communication requires a loop resistance of at least 250 ohms; if the control system input has lower impedance, install a 250-ohm resistor in series.

### 3.3 Impulse lines

Keep impulse lines as short as possible and slope them at least 1:12. In gas service, condensation in impulse lines is the most common cause of reading drift; install a drain valve at the lowest point.

## 4 Calibration

### 4.1 Zero adjustment

To recalibrate the zero point, isolate the transmitter from the process and vent it to atmosphere (gauge models) or apply full vacuum (absolute models). Open the cover on the electronics housing and press and hold the ZERO button for five seconds. The status LED blinks green three times when the new zero is accepted. Zero adjustment can also be performed over HART using the command "Set PV Zero".

### 4.2 Span adjustment

Span calibration requires a reference pressure source with an accuracy of 0.05% of full scale or better. Apply the upper range value, then press and hold the SPAN button for five seconds. Span adjustment should be performed only in a calibration laboratory.

### 4.3 Calibration interval

The PX-200 does not require periodic recalibration when installed in clean, dry gas service. In liquid or steam service, verify zero annually and span every three years. Record all adjustments in the maintenance log.

## 5 HART communication

The PX-200 supports HART revision 7. Device variables include primary pressure, sensor temperature, and electronics temperature. Burst mode can publish pressure and status every 0.5 seconds. The device description (DD) file is available from the Halden support portal.

## 6 Maintenance

Inspect the process connection and manifold for leaks every six months. Do not open the sensor module; it contains no user-serviceable parts. When returning a unit for repair, clean all wetted parts and include a completed decontamination form.

## 7 Troubleshooting

### 7.1 Error codes

| Code | Meaning | Action |
|---|---|---|
| E-03 | Supply voltage below 10.5 V | Check loop power supply and wiring resistance |
| E-11 | Electronics temperature out of range | Shield the housing from direct sun or heat sources |
| E-17 | Sensor diaphragm fault | Return the transmitter for repair; do not attempt field repair |
| E-22 | Configuration memory checksum error | Reload configuration via HART; replace electronics if it persists |
| E-30 | Pressure above upper range limit | Check process conditions; inspect diaphragm after overpressure |

### 7.2 Reading drift

A reading that drifts slowly upward in gas service usually indicates condensate accumulating in the impulse line. Drain the line and verify the slope described in section 3.3. Drift after an overpressure event may indicate diaphragm damage; check for error code E-17.

## 8 Safety

Before servicing the transmitter on a live process, apply lockout/tagout, depressurize the process connection, and verify zero energy. Venting process media can cause injury; wear protective equipment appropriate to the process fluid.

## Revision history

| Revision | Date | Change |
|---|---|---|
| E | 2024-03 | Added HART 7 burst mode |
| F | 2025-11 | Added error code E-30; clarified calibration interval |

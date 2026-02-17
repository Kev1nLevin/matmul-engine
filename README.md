# tinyTPU — 32×32 BF16 Systolic Array Neural Network Accelerator

## RTL-to-GDS | Cadence Flow | ASAP7 7nm PDK

[![Flow](https://img.shields.io/badge/Flow-RTL%20to%20GDS-brightgreen)](.)
[![PDK](https://img.shields.io/badge/PDK-ASAP7%207nm-blue)](.)
[![Tools](https://img.shields.io/badge/Tools-Cadence%20Genus%20%2B%20Innovus-orange)](.)
[![Timing](https://img.shields.io/badge/Timing-Met%20%E2%9C%93-success)](.)

---

## Overview

A fully synthesizable **parameterized output-stationary systolic array** for neural network inference, targeting ASAP7 7nm FinFET. The design has evolved from a validated INT8 4×4/8×8 baseline through the full RTL-to-GDS flow to the current **32×32 BF16** configuration with AXI4 SoC integration and GeLU activation.

**Current RTL configuration:**
- **32×32 array** — 1024 PEs, 32-bit FP32 accumulators
- **BF16 inputs** — 2-stage pipelined BF16×BF16→FP32 MAC per PE
- **AXI4 slave interface** — 64-bit bus, full burst support for RV64 SoC integration
- **GeLU activation** — 256-entry FP32 LUT on the result read path (combinational, no pipeline)
- **IRQ support** — level-sensitive interrupt on computation complete

Earlier INT8 4×4 and 8×8 configurations have been taken through the full P&R flow and are timing-clean at **667 MHz** — results documented below.

---

## Architecture

```
              B columns (top edge, flow top-to-bottom)
                ↓        ↓        ↓   ...   ↓
             a_col[0] a_col[1] a_col[2]  a_col[N-1]
                |        |        |         |
  b_row[0] → [PE00] → [PE01] → [PE02] → [PE0,N-1]
  b_row[1] → [PE10] → [PE11] → [PE12] → [PE1,N-1]
     ...
  b_row[N-1]→[PEN0] → [PEN1] → [PEN2] → [PEN,N-1]

  A rows (left edge, flow left-to-right)

  Each PE: acc[i][j] += a_in * b_in  (BF16 multiply, FP32 accumulate)
  Results C = A × B accumulate in-place (output-stationary)
  Inputs are staggered by the skew controllers before entering the array.
```

### Block Diagram

```
  ┌──────────────────────────────────────────────────────────────┐
  │                        tpu_top                               │
  │  ┌─────────────────────────────────────────────────────────┐ │
  │  │                  tpu_axi4_wrapper                        │ │
  │  │  AXI4 Slave (64-bit, INCR burst)  IRQ  CTRL/MAT/RESULT  │ │
  │  └───────────────────────┬─────────────────────────────────┘ │
  │                          │ MMIO handshake signals             │
  │  ┌───────────────────────▼─────────────────────────────────┐ │
  │  │                    systolic_top                          │ │
  │  │                                                          │ │
  │  │  ┌──────────┐  ┌───────────┐                            │ │
  │  │  │ Matrix A  │  │   Skew    │                            │ │
  │  │  │ Reg File  │─▶│Controller │──▶ a_col[0..N-1]          │ │
  │  │  └──────────┘  └───────────┘         │                  │ │
  │  │                                       ▼                  │ │
  │  │  ┌──────────┐  ┌───────────┐  ┌─────────────────┐       │ │
  │  │  │ Matrix B  │  │   Skew    │  │  32×32 Systolic │       │ │
  │  │  │ Reg File  │─▶│Controller │─▶│     Array       │       │ │
  │  │  └──────────┘  └───────────┘  │   (1024 PEs)    │       │ │
  │  │                                └────────┬────────┘       │ │
  │  │  ┌───────────┐                          │ result[32][32]  │ │
  │  │  │    FSM    │── control ──────────────▶│                 │ │
  │  │  │Controller │               ┌──────────▼──────────┐     │ │
  │  │  └───────────┘               │  GeLU Units (×32)   │     │ │
  │  │                              │  (on result read)    │     │ │
  │  │                              └─────────────────────┘     │ │
  │  └─────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────┘
```

---

## Module Hierarchy

```
tpu_top
├── tpu_axi4_wrapper          AXI4 slave, MMIO, IRQ, burst FSMs
└── systolic_top              32×32 systolic array top-level
    ├── systolic_controller   FSM: IDLE→CLEAR→LOAD→DRAIN→DONE
    ├── input_skew_controller (×2)  Stagger A rows and B columns
    ├── systolic_array_32x32  1024-PE mesh
    │   └── processing_element (×1024)
    │       └── mac_unit      BF16×BF16→FP32, 2-stage pipeline
    └── gelu_unit (×32)       Combinational GeLU on result read path
```

---

## AXI4 Memory Map

Base-relative, 64-bit data bus. Connect `tpu_top` to a RV64 AXI4 master.

| Address | Register | Description |
|---|---|---|
| `0x0000` | CTRL | `[4]=gelu_en` `[3]=irq_en` `[2]=done(RO)` `[1]=busy(RO)` `[0]=start(WO)` |
| `0x1000` | MAT_A | Row i at `i×64B`. INCR burst, AWLEN=7 (8 beats × 4 BF16 per 64-bit word) |
| `0x2000` | MAT_B | Same layout as MAT_A |
| `0x3000` | RESULTS | Row i at `i×128B`. INCR burst, ARLEN=15 (16 beats × 2 FP32 per 64-bit word) |

- Only INCR burst (`AWBURST/ARBURST = 2'b01`) is supported
- `done` is latched; cleared when a new `start` is issued
- `irq` is level-sensitive: `irq = irq_en & done`

---

## Computation Timing

For N=32, MAC_LATENCY=2:

| Phase | Cycles | Description |
|---|---|---|
| CLEAR | 1 | Zero accumulators and skew registers |
| LOAD | 32 | Stream N rows of A and N columns of B through skew controllers |
| DRAIN | 63 | Wait for last data to propagate to PE[31][31] and commit |
| **Total** | **96** | Results valid on `done` pulse |

---

## Validated P&R Results (INT8, ASAP7 7nm)

The following configurations have been taken through the full Genus + Innovus flow. These used INT8 inputs with 32-bit integer accumulators (the original design before BF16 upgrade).

### 8×8 Array — Post-Route (667 MHz, TT 0.7V 25°C)

| Metric | Value |
|---|---|
| Setup WNS | **+21.6 ps** ✓ |
| Hold WNS | Met ✓ |
| Total Power | **14.05 mW** |
| Die Size | 370.4 × 370.4 µm |
| Cell Count | 33,560 |
| Placement Density | 56.6% |

#### Power Breakdown

| Component | Power (mW) | % |
|---|---|---|
| Internal | 9.607 | 68.4% |
| Switching | 4.010 | 28.5% |
| Leakage | 0.434 | 3.1% |
| **Total** | **14.050** | |

| Group | Power (mW) | % |
|---|---|---|
| Combinational | 7.083 | 50.4% |
| Sequential | 5.385 | 38.3% |
| Clock Tree | 1.582 | 11.3% |

#### Synthesis → P&R Comparison

| Metric | Genus | Innovus | Delta |
|---|---|---|---|
| Setup WNS | 0 ps | +21.6 ps | Improved |
| Cell Count | 31,520 | 33,560 | +6.5% |
| Cell Area | 70,544 µm² | 73,016 µm² | +3.5% |
| Total Power | 19.91 mW | 14.05 mW | −29.4% |

> Power reduction after P&R is expected: Genus uses wire-load estimates; Innovus uses extracted parasitics from actual routing.

#### DRC

| Check | Result |
|---|---|
| Geometry DRC | 976 M3 Rect violations (known ASAP7 4× scaled LEF artifact) |
| Cap / Tran / Fanout | 0 violations |
| Antenna / SI | Clean |

---

### 4×4 Array — Post-Route (667 MHz, TT corner)

| Metric | Value |
|---|---|
| Setup WNS | **+21 ps** ✓ |
| Hold WNS | **+79 ps** ✓ |
| Total Power | **4.57 mW** |
| Die Size | 192.8 × 192.8 µm |
| Cell Count | 8,228 |

#### Power Breakdown

| Component | Power (mW) | % |
|---|---|---|
| Internal | 2.238 | 48.9% |
| Switching | 2.230 | 48.8% |
| Leakage | 0.105 | 2.3% |
| **Total** | **4.572** | |

---

### Scaling Analysis: INT8 4×4 → 8×8

| Metric | 4×4 | 8×8 | Ratio |
|---|---|---|---|
| PEs | 16 | 64 | 4.0× |
| Cell Count | 8,228 | 33,560 | 4.08× |
| Die Area | 37,172 µm² | 137,211 µm² | 3.69× |
| Total Power | 4.57 mW | 14.05 mW | 3.07× |
| Peak GOPS | 10.67 | 42.67 | 4.0× |
| GOPS/mW | 2.33 | 3.04 | 1.30× |

Sub-linear power growth with linear throughput scaling — larger arrays are more efficient per operation.

---

## File Structure

```
rtl/
├── tpu_top.sv                  Top-level: AXI4 wrapper + systolic_top
├── tpu_axi4_wrapper.sv         AXI4 slave (MMIO, IRQ, burst read/write FSMs)
├── systolic_top.sv             Systolic array top (matrix reg files, skew, GeLU)
├── systolic_array_32x32.sv     32×32 PE mesh
├── processing_element.sv       PE: systolic pass-through registers + MAC
├── mac_unit.sv                 2-stage BF16×BF16→FP32 pipelined MAC
├── input_skew_controller.sv    Channel-k delay-k stagger registers
├── systolic_controller.sv      FSM controller (IDLE→CLEAR→LOAD→DRAIN→DONE)
├── gelu_unit.sv                Combinational FP32 GeLU, 256-entry LUT
└── gelu_lut.mem                GeLU LUT hex data (generate with script in gelu_unit.sv)
```

---

## Tool Flow

```
   SystemVerilog RTL
        │
        ▼
   ┌──────────┐    Cadence Xcelium
   │ Simulate  │──────────────────── functional verification
   └──────────┘
        │
        ▼
   ┌──────────┐    Cadence Genus 23.14
   │ Synthesis │──────────────────── ASAP7 LVT + SLVT multi-Vt
   └──────────┘                      Target: 667 MHz
        │
        ▼
   ┌──────────┐    Cadence Innovus 23.14
   │   P & R   │──────────────────── Floorplan → Place → CTS → Route
   └──────────┘                      OCV + SI-aware optimization
        │
        ▼
   DEF + Netlist + SDF
```

---

## Design Decisions

1. **Output-Stationary Dataflow** — Each PE accumulates its own C[i][j] in place. Minimizes result data movement; optimal for weight-reuse inference.

2. **BF16 Inputs / FP32 Accumulator** — BF16 (1s + 8e + 7m) matches the exponent range of FP32, making conversion trivial. FP32 accumulator avoids numeric overflow across the dot-product length.

3. **Simplified FP Arithmetic (Option A)** — Flush-to-zero for denormals, truncate rounding, no NaN/Inf propagation. Sufficient for inference; eliminates IEEE 754 edge-case logic.

4. **2-Stage MAC Pipeline** — Stage 1: BF16 multiply → FP32 product. Stage 2: FP32 accumulate. The controller extends DRAIN by `(MAC_LATENCY-1)` cycles to ensure the last product commits before results are read.

5. **Skewed Input Feeding** — Channel k is delayed k cycles before entering the array. Aligns partial products across the diagonal wave without needing data buffering inside the array.

6. **Full AXI4 (not AXI4-Lite)** — INCR burst transfers load an entire matrix row in one transaction (8 beats for 32×32 BF16 on a 64-bit bus), minimizing AXI transaction overhead.

7. **32×32 for RV64** — WORDS_PER_ROW = (32×16)/64 = 8 beats per row. Balanced memory bandwidth vs. compute density on a 64-bit bus. 1024 MACs/cycle for transformer/CNN inference.

8. **ROM-based GeLU** — 256-entry FP32 LUT covers x ∈ [−4, +4] at 1/32 resolution. Address extracted entirely from FP32 bit fields (no floating-point arithmetic in the address path). Purely combinational — zero pipeline latency on the result read path.

---

## Future Work

- Synthesis and P&R of the 32×32 BF16 configuration
- RTL simulation and verification of BF16 datapath
- Multi-batch / streaming operation (continuous matrix feed without re-start)
- Signoff STA with Cadence Tempus (multi-corner multi-mode)
- QRC parasitic extraction for signoff-quality timing
- Resolve M3 DRC violations with tighter routing constraints

---

## Prerequisites

- **Cadence Xcelium** — RTL simulation
- **Cadence Genus** — Logic synthesis
- **Cadence Innovus** — Place & route
- **ASAP7 7nm PDK** — [GitHub](https://github.com/The-OpenROAD-Project/asap7)
- **Python 3** — GeLU LUT generation (script embedded in `gelu_unit.sv`)

---

## Quick Start

```bash
# 1. Generate GeLU LUT (required before simulation/synthesis)
python3 -c "
import struct, math
def fp32_hex(v): return '{:08x}'.format(struct.unpack('>I',struct.pack('>f',v))[0])
with open('gelu_lut.mem','w') as f:
    for i in range(256):
        x = max(-4.0, min(4.0, (i-128)/32.0))
        g = x * 0.5 * (1 + math.tanh(0.7978846*(x + 0.044715*x**3)))
        f.write(fp32_hex(g) + '\n')
"

# 2. Update ASAP7 library paths in scripts/

# 3. RTL simulation
cd scripts && source run_sim.sh

# 4. Synthesis (set ARRAY_SIZE in TCL script)
cd run && genus -f ../scripts/syn_systolic.tcl

# 5. Place & Route
cd PnR/run && innovus -init ../scripts/innovus.tcl
```

---

## License

Released for educational and portfolio purposes.

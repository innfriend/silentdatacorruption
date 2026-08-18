import React, { useState } from 'react';
import { Ieee754Format } from '../types';
import { IEEE754_FORMATS } from '../data/mockCluster';
import {
  BookOpen,
  Atom,
  Binary,
  Layers,
  ShieldAlert,
  HelpCircle,
  TrendingDown,
  RotateCcw,
  Sparkles,
} from 'lucide-react';

export const ScienceModule: React.FC = () => {
  const [selectedFormat, setSelectedFormat] = useState<Ieee754Format>(IEEE754_FORMATS[0]); // BF16

  // Bit representation state: array of 0s and 1s
  // Default to a small normal float ~1.50
  const [bits, setBits] = useState<number[]>([
    0, // Sign (1 bit)
    0, 1, 1, 1, 1, 1, 1, 1, // Exponent (8 bits = 127 = 2^0)
    1, 0, 0, 0, 0, 0, 0, // Mantissa (7 bits = 0.5) => Total = 1.50
  ]);

  const handleFormatChange = (fmt: Ieee754Format) => {
    setSelectedFormat(fmt);
    // Initialize default bits for the format
    if (fmt.name.includes('BF16')) {
      setBits([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); // 1.5
    } else if (fmt.name.includes('FP16')) {
      setBits([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // 1.5 in FP16 (5 exp, 10 mantissa)
    } else if (fmt.name.includes('FP8')) {
      setBits([0, 0, 1, 1, 1, 1, 0, 0]); // 1.5 in FP8 E4M3 (4 exp, 3 mantissa)
    } else {
      // FP32
      const arr = new Array(32).fill(0);
      arr[1] = 0; arr[2] = 1; arr[3] = 1; arr[4] = 1; arr[5] = 1; arr[6] = 1; arr[7] = 1; arr[8] = 1; // Exp 127
      arr[9] = 1; // Mantissa 0.5
      setBits(arr);
    }
  };

  const toggleBit = (index: number) => {
    setBits((prev) => {
      const next = [...prev];
      next[index] = next[index] === 0 ? 1 : 0;
      return next;
    });
  };

  // Decode the current float
  const sign = bits[0] === 1 ? -1 : 1;
  const expBits = bits.slice(1, 1 + selectedFormat.exponentBits);
  const mantissaBits = bits.slice(1 + selectedFormat.exponentBits);

  const rawExponent = expBits.reduce((acc, bit, idx) => acc + bit * Math.pow(2, expBits.length - 1 - idx), 0);
  const exponentUnbiased = rawExponent - selectedFormat.bias;

  let mantissaFraction = 0;
  mantissaBits.forEach((bit, idx) => {
    if (bit === 1) mantissaFraction += Math.pow(2, -(idx + 1));
  });

  const isZero = rawExponent === 0 && mantissaFraction === 0;
  const isSubnormal = rawExponent === 0 && mantissaFraction !== 0;
  const isInfOrNan = rawExponent === Math.pow(2, selectedFormat.exponentBits) - 1;

  let decodedValue = 0;
  let statusTag = 'NORMAL FLOAT';

  if (isZero) {
    decodedValue = sign * 0;
    statusTag = 'ZERO (0.0)';
  } else if (isSubnormal) {
    decodedValue = sign * mantissaFraction * Math.pow(2, 1 - selectedFormat.bias);
    statusTag = 'SUBNORMAL (Underflow Risk)';
  } else if (isInfOrNan) {
    if (mantissaFraction === 0) {
      decodedValue = sign * Infinity;
      statusTag = sign > 0 ? '+INFINITY (Overflow)' : '-INFINITY (Overflow)';
    } else {
      decodedValue = NaN;
      statusTag = 'NaN (Not a Number - Poisoning)';
    }
  } else {
    decodedValue = sign * (1 + mantissaFraction) * Math.pow(2, exponentUnbiased);
  }

  return (
    <div className="space-y-6">
      {/* Interactive IEEE 754 Bit Flipper Simulator Card */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
          <div>
            <div className="flex items-center gap-2">
              <Binary className="w-4 h-4 text-[#4A5D4E]" />
              <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                Interactive IEEE 754 Bit-Flipper Simulator
              </h3>
            </div>
            <p className="text-xs text-[#666] mt-0.5">
              Click any bit to simulate an in-flight cosmic ray SEU or ALU carry-chain failure
            </p>
          </div>

          {/* Format selector buttons */}
          <div className="flex bg-[#F8F7F4] p-1 rounded-[2px] border border-[#D1D0CB] text-xs font-mono">
            {IEEE754_FORMATS.map((fmt) => (
              <button
                key={fmt.name}
                onClick={() => handleFormatChange(fmt)}
                className={`px-2.5 py-1 rounded-[2px] transition-all cursor-pointer ${
                  selectedFormat.name === fmt.name
                    ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                    : 'text-[#666] hover:text-[#2A2A2A]'
                }`}
              >
                {fmt.name.split(' ')[0]}
              </button>
            ))}
          </div>
        </div>

        {/* Live Bit Strip */}
        <div className="bg-[#1C1C1A] text-white p-4 rounded-[2px] border border-[#333330] mb-4">
          <div className="flex flex-wrap gap-1.5 justify-center items-center">
            {bits.map((bit, idx) => {
              const isSign = idx === 0;
              const isExp = idx >= 1 && idx < 1 + selectedFormat.exponentBits;
              const isMant = idx >= 1 + selectedFormat.exponentBits;

              let color = 'bg-blue-700 hover:bg-blue-600';
              if (isSign) color = 'bg-purple-700 hover:bg-purple-600';
              else if (isExp) color = 'bg-amber-700 hover:bg-amber-600';
              else if (isMant) color = 'bg-[#4A5D4E] hover:bg-[#3B4A3E]';

              return (
                <button
                  key={idx}
                  onClick={() => toggleBit(idx)}
                  className={`w-7 h-9 rounded-[2px] text-xs font-mono font-bold flex flex-col items-center justify-center transition-all ${color} shadow-xs cursor-pointer`}
                  title={`Bit ${idx}: ${isSign ? 'Sign' : isExp ? 'Exponent' : 'Mantissa'}`}
                >
                  <span className="text-[9px] opacity-70">b{bits.length - 1 - idx}</span>
                  <span className="text-sm font-bold">{bit}</span>
                </button>
              );
            })}
          </div>

          {/* Bit Labels Legend */}
          <div className="flex justify-center gap-6 mt-3 text-[11px] font-mono">
            <span className="flex items-center gap-1 text-purple-300">
              <span className="w-2.5 h-2.5 rounded-xs bg-purple-700"></span> Sign (1 bit)
            </span>
            <span className="flex items-center gap-1 text-amber-300">
              <span className="w-2.5 h-2.5 rounded-xs bg-amber-700"></span> Exponent ({selectedFormat.exponentBits} bits)
            </span>
            <span className="flex items-center gap-1 text-emerald-300">
              <span className="w-2.5 h-2.5 rounded-xs bg-[#4A5D4E]"></span> Mantissa ({selectedFormat.mantissaBits} bits)
            </span>
          </div>
        </div>

        {/* Real-time Decoded Output Value Card */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 font-mono text-xs">
          <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[#999] text-[10px]">DECODED SCALAR VALUE</div>
            <div className="text-base font-bold text-[#2A2A2A] truncate font-serif">
              {Number.isNaN(decodedValue)
                ? 'NaN'
                : !Number.isFinite(decodedValue)
                ? decodedValue.toString()
                : decodedValue.toExponential(4)}
            </div>
            <div className="text-[10px] text-[#666]">Base float output</div>
          </div>

          <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[#999] text-[10px]">EXPONENT POWER (2^E)</div>
            <div className="text-base font-bold text-[#8C2D2D]">
              2^{exponentUnbiased} ({Math.pow(2, exponentUnbiased).toExponential(2)})
            </div>
            <div className="text-[10px] text-[#666]">Raw Exp = {rawExponent}</div>
          </div>

          <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[#999] text-[10px]">MANTISSA FRACTION (1 + M)</div>
            <div className="text-base font-bold text-[#4A5D4E]">
              {(1 + mantissaFraction).toFixed(4)}
            </div>
            <div className="text-[10px] text-[#666]">Frac = +{mantissaFraction.toFixed(4)}</div>
          </div>

          <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[#999] text-[10px]">IEEE 754 CLASSIFICATION</div>
            <div
              className={`text-xs font-bold truncate ${
                isInfOrNan ? 'text-[#8C2D2D] font-bold' : isSubnormal ? 'text-amber-800' : 'text-[#4A5D4E]'
              }`}
            >
              {statusTag}
            </div>
            <div className="text-[10px] text-[#666]">{selectedFormat.name}</div>
          </div>
        </div>
      </div>

      {/* Technical Whitepaper: Silicon Physics, ALU Failure Modes, and Parity Invariants */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Section 1: The Physics of Silent Hardware Errors */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-3 shadow-xs">
          <div className="flex items-center gap-2">
            <Atom className="w-4 h-4 text-[#4A5D4E]" />
            <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
              1. SILICON PHYSICS & TRANSIENT NOISE MECHANICS
            </h4>
          </div>

          <div className="text-xs text-[#666] leading-relaxed space-y-2">
            <p>
              In leading-edge 4nm/3nm lithography (NVIDIA Hopper H100, Blackwell B200), transistor gate oxide layers are only a few atoms thick (~1.2 nm). Silent Data Corruption (SDC) arises from four primary physical phenomena:
            </p>
            <ul className="list-disc pl-4 space-y-1 text-[#2A2A2A]">
              <li>
                <strong>Atmospheric Neutron Single Event Upsets (SEUs):</strong> Cosmic ray neutrons collide with silicon nuclei, creating secondary electron-hole pairs that flip latch states without causing permanent hardware faults.
              </li>
              <li>
                <strong>Dynamic Voltage Droops (L · di/dt):</strong> When 1,000+ SMs switch from sparse softmax memory fetch to dense GEMM Tensor Core MMA, localized current spikes induce power-rail inductive collapse (Vdd &lt; 0.68V).
              </li>
              <li>
                <strong>Negative Bias Temperature Instability (NBTI):</strong> Trapping of charge carriers in the dielectric over tens of thousands of continuous GPU-hours causes subtle threshold voltage shifts.
              </li>
            </ul>
          </div>
        </div>

        {/* Section 2: Why Hardware ECC Fails on ALUs */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-3 shadow-xs">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-[#8C2D2D]" />
            <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
              2. WHY HARDWARE ECC CANNOT PROTECT ARITHMETIC ALUS
            </h4>
          </div>

          <div className="text-xs text-[#666] leading-relaxed space-y-2">
            <p>
              Modern GPUs incorporate SECDED (Single Error Correction, Double Error Detection) on SRAM register files and HBM3 stacks. However:
            </p>
            <ul className="list-disc pl-4 space-y-1 text-[#2A2A2A]">
              <li>
                <strong>Data at Rest vs. Data in Flight:</strong> ECC parity bits are stripped before feeding operands into the ALU adder tree and MMA accumulators.
              </li>
              <li>
                <strong>Carry-Chain Inversion:</strong> During 16-bit / 8-bit dot product reductions, an arithmetic carry-chain bit drop cannot be checked by SRAM ECC because it occurs inside the active clock-cycle arithmetic logic.
              </li>
              <li>
                <strong>Silent Propagation:</strong> The resulting corrupt float writes back to global memory with valid newly calculated ECC bits, rendering hardware diagnostics blind to the arithmetic corruption.
              </li>
            </ul>
          </div>
        </div>

        {/* Section 3: Mathematical Proof of Stochastic Parity Invariants */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-3 md:col-span-2 shadow-xs">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-[#4A5D4E]" />
            <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
              3. MATHEMATICAL DERIVATION: IN-REGISTER FREIVALDS STOCHASTIC PARITY
            </h4>
          </div>

          <div className="text-xs text-[#666] leading-relaxed space-y-2 font-sans">
            <p>
              Direct matrix multiplication verification requires computing C = A x B twice, incurring <strong>+100% FLOP overhead</strong> (O(N^3) complexity). SilentGuard applies a randomized Freivalds invariant projection:
            </p>

            <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB] font-mono text-[11px] text-[#2A2A2A] space-y-1">
              <div>Let A in R^(M x K), B in R^(K x N), and C in R^(M x N) be the output tile.</div>
              <div>Let r in {"{-1, +1}^M"} be a pseudo-Rademacher stochastic projection vector.</div>
              <div className="font-bold text-[#4A5D4E]">
                Invariant: r^T · C = (r^T · A) · B
              </div>
            </div>

            <p>
              Evaluating (r^T · A) requires O(MK) ops. Multiplying this intermediate 1 x K vector by B requires O(KN) ops.
              Thus, total invariant verification cost is:
              <br />
              <span className="font-mono text-[#4A5D4E] font-bold">O(MK + KN) &lt;&lt; O(MKN)</span>
              <br />
              For an 8192 x 8192 GEMM tile, verification requires only <strong>0.024% of the compute FLOPs</strong> while providing a <strong>&gt;99.999% detection probability</strong> for any non-zero arithmetic corruption.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

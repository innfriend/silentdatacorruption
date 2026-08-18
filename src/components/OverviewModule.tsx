import React from 'react';
import {
  ShieldCheck,
  Server,
  Terminal,
  Search,
  Zap,
  Calculator,
  DollarSign,
  BookOpen,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  TrendingUp,
  Layers,
} from 'lucide-react';
import { ModuleTab } from '../types';

interface OverviewModuleProps {
  setActiveTab: (tab: ModuleTab) => void;
  healthyRanks: number;
  totalRanks: number;
  quarantinedRanks: number;
  corruptedRanks: number;
  recomputedTiles: number;
}

export const OverviewModule: React.FC<OverviewModuleProps> = ({
  setActiveTab,
  healthyRanks,
  totalRanks,
  quarantinedRanks,
  corruptedRanks,
  recomputedTiles,
}) => {
  const launcherCards = [
    {
      id: 'simulator' as ModuleTab,
      title: 'Live Cluster NOC & Fault Injector',
      description:
        'Visualize 128 GPU ranks across 16 DGX H100 nodes. Inject cosmic ray bit-flips, voltage droops, and observe in-flight autonomous self-healing and loss recovery.',
      icon: <Server className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Launch Cluster NOC',
      status: `${healthyRanks}/${totalRanks} Ranks Active`,
    },
    {
      id: 'kernels' as ModuleTab,
      title: 'Developer SDK & Fused Kernels',
      description:
        'Compilable OpenAI Triton, PyTorch FSDP, DeepSpeed, and FlashAttention kernels embedding Freivalds stochastic parity invariants directly into register files with <0.08% overhead.',
      icon: <Terminal className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Generate & Download .py',
      status: 'OpenAI Triton / CUDA v12.4',
    },
    {
      id: 'scanner' as ModuleTab,
      title: 'Checkpoint & Tensor Scanner',
      description:
        'Inspect PyTorch .pt and SafeTensors weights. Automatically detect kurtosis anomalies, silent exponent overflows, and low-order mantissa drift without training interruption.',
      icon: <Search className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Scan Weights & Log Files',
      status: 'Kurtosis & Exponent Audit',
    },
    {
      id: 'diagnostic' as ModuleTab,
      title: 'Gemini AI Root Cause Diagnostic',
      description:
        'Autonomous GPU microarchitecture forensic investigator. Connects to Gemini models to parse backtraces, identify damaged ALU units, and prescribe Slurm drain commands.',
      icon: <Zap className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Run AI Forensics',
      status: 'Powered by Gemini 3.7 Flash',
    },
    {
      id: 'roi' as ModuleTab,
      title: 'Compute Waste & ROI Calculator',
      description:
        'Model the financial impact of silent data errors on 64 to 16,384 GPU clusters. Quantify saved GPU-hours, avoided rollback compute waste, and net dollar return.',
      icon: <Calculator className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Calculate Savings',
      status: '$1.42M+ Typical Savings',
    },
    {
      id: 'science' as ModuleTab,
      title: 'SDC Science & IEEE 754 Explorer',
      description:
        'Deep-dive whitepaper detailing atmospheric neutron flux, transistor NBTI aging, why ECC misses ALU calculations, and an interactive bit-flipper explorer for FP32/BF16/FP8.',
      icon: <BookOpen className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Explore Knowledge Base',
      status: 'Interactive Bit Flipper',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Executive SDC Threat & Mathematical Guarantee Banner */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 sm:p-8 shadow-xs">
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-[2px] bg-[#F8F7F4] text-[#4A5D4E] text-[11px] font-mono mb-3 border border-[#D1D0CB]">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span className="font-semibold tracking-wider">MATHEMATICALLY PROVEN INVARIANT PROTECTION</span>
          </div>
          <h2 className="text-xl sm:text-3xl font-light tracking-tight text-[#2A2A2A] font-serif mb-3">
            Eliminating Silent Data Corruption (SDC) in Frontier LLM Pre-Training
          </h2>
          <p className="text-xs sm:text-sm text-[#555] leading-relaxed mb-5">
            Silent Data Errors occur when cosmic rays, dynamic voltage droops (L · di/dt), or transistor aging flip bits directly inside GPU arithmetic ALUs and Tensor Cores. Traditional SRAM/HBM ECC only guards data at rest. SilentGuard injects in-register stochastic parity projections (<span className="font-mono text-[#2A2A2A]">r^T · C = (r^T · A) · B</span>) into fused Triton kernels, catching corruptions in <strong className="text-[#2A2A2A] font-semibold">&lt;3.8 ms</strong> with <strong className="text-[#4A5D4E] font-semibold">&lt;0.08% compute overhead</strong>.
          </p>
          <div className="flex flex-wrap gap-5 text-xs font-mono text-[#4A5D4E]">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">Zero HBM Roundtrips</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">In-Flight Tile Recomputation</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">Autonomous Slurm Node Quarantine</span>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Executive Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-6 gap-3">
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Cluster Health</div>
          <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif mt-1">
            {healthyRanks === totalRanks ? '100%' : `${((healthyRanks / totalRanks) * 100).toFixed(1)}%`}
          </div>
          <div className="text-[11px] text-[#4A5D4E] font-medium mt-1">
            {healthyRanks} / {totalRanks} H100 Ranks
          </div>
        </div>

        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Detection Speed</div>
          <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif mt-1">
            3.18<span className="text-xs text-[#999] ml-1 font-sans">ms</span>
          </div>
          <div className="text-[11px] text-[#666] mt-1">In-register cache check</div>
        </div>

        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">FLOP Overhead</div>
          <div className="text-2xl sm:text-3xl font-light text-[#4A5D4E] font-serif mt-1">&lt;0.08%</div>
          <div className="text-[11px] text-[#666] mt-1">O(N²) vs O(N³) GEMM</div>
        </div>

        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Protected GPUs</div>
          <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif mt-1">16,384</div>
          <div className="text-[11px] text-[#666] mt-1">Scale validated fleet</div>
        </div>

        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Prevented Spikes</div>
          <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif mt-1">{14 + recomputedTiles}</div>
          <div className="text-[11px] text-[#4A5D4E] font-medium mt-1">0 Lost Iterations</div>
        </div>

        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Saved Pre-train $</div>
          <div className="text-2xl sm:text-3xl font-light text-[#4A5D4E] font-serif mt-1">$1.42M+</div>
          <div className="text-[11px] text-[#666] mt-1">70B model run basis</div>
        </div>
      </div>

      {/* 3x2 Modular Tool Launcher Cards */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-[#2A2A2A] tracking-wider uppercase font-mono">
            Platform Modules & Autonomous Workspaces
          </h3>
          <span className="text-[11px] text-[#666] font-mono">
            6 Specialized Workspaces Active
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {launcherCards.map((card) => (
            <div
              key={card.id}
              className="bg-white border border-[#D1D0CB] hover:border-[#4A5D4E] rounded-[3px] p-5 flex flex-col justify-between transition-all duration-150 group shadow-xs"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="w-8 h-8 rounded-[2px] bg-[#F8F7F4] border border-[#D1D0CB] flex items-center justify-center group-hover:bg-[#EBEAE5] transition-colors">
                    {card.icon}
                  </div>
                  <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-[2px] bg-[#F8F7F4] text-[#666] border border-[#D1D0CB]">
                    {card.status}
                  </span>
                </div>
                <h4 className="text-base font-light font-serif text-[#2A2A2A] mb-1.5 group-hover:text-[#4A5D4E] transition-colors">
                  {card.title}
                </h4>
                <p className="text-xs text-[#666] leading-relaxed mb-4">
                  {card.description}
                </p>
              </div>

              <button
                onClick={() => setActiveTab(card.id)}
                className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 text-xs font-semibold rounded-[2px] bg-[#F8F7F4] hover:bg-[#4A5D4E] text-[#2A2A2A] hover:text-white border border-[#D1D0CB] hover:border-transparent transition-colors cursor-pointer"
              >
                <span>{card.actionText}</span>
                <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Architectural Diagram & Invariant Verification Pipeline */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono mb-4">
          SilentGuard Autonomous Reliability Pipeline
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[11px] font-mono font-bold text-[#4A5D4E] mb-1">01. IN-REGISTER INJECT</div>
            <div className="text-xs font-semibold text-[#2A2A2A] mb-1">Stochastic Vector Generation</div>
            <p className="text-[11px] text-[#666] leading-relaxed">
              Generates random pseudo-rademacher vector r in {"{-1, +1}^M"} in SRAM register files without global memory allocation.
            </p>
          </div>
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[11px] font-mono font-bold text-[#4A5D4E] mb-1">02. FUSED GEMM MMA</div>
            <div className="text-xs font-semibold text-[#2A2A2A] mb-1">Dual-Projection Math</div>
            <p className="text-[11px] text-[#666] leading-relaxed">
              As Tensor Cores compute A x B, registers concurrently evaluate r^T · A in O(MK) and multiply by B in O(KN).
            </p>
          </div>
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[11px] font-mono font-bold text-[#4A5D4E] mb-1">03. INVARIANT CHECK</div>
            <div className="text-xs font-semibold text-[#2A2A2A] mb-1">Instant Delta Comparison</div>
            <p className="text-[11px] text-[#666] leading-relaxed">
              Evaluates ||r^T · C - (r^T · A) · B|| &lt;= epsilon. Any bit-flip triggers an immediate trap in 3.2ms.
            </p>
          </div>
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB]">
            <div className="text-[11px] font-mono font-bold text-[#4A5D4E] mb-1">04. AUTO-HEAL & DRAIN</div>
            <div className="text-xs font-semibold text-[#2A2A2A] mb-1">Zero-Loss Continuity</div>
            <p className="text-[11px] text-[#666] leading-relaxed">
              Corrupted tile is re-dispatched to a clean SM, preventing NaN propagation to optimizer state while Slurm drains the degrading rank.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};


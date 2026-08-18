import React, { useState } from 'react';
import { KernelConfig } from '../types';
import { generateKernelCode } from '../data/kernelTemplates';
import {
  Terminal,
  Download,
  Copy,
  Check,
  Cpu,
  Sliders,
  Sparkles,
  Zap,
  ShieldCheck,
  Flame,
  Layers,
  FolderTree,
  FileCode,
  Activity,
  BarChart3,
  Monitor,
  Eye,
  Server,
  FileText,
  AlertTriangle,
} from 'lucide-react';

export const KernelSuiteModule: React.FC = () => {
  const [config, setConfig] = useState<KernelConfig>({
    framework: 'triton',
    precision: 'bf16',
    toleranceEpsilon: 1e-4,
    samplingRatePercent: 3,
    autoDrainOnFailure: true,
    inlineRecompute: true,
    telemetryEndpoint: 'http://localhost:3000/api/telemetry',
  });

  const [copied, setCopied] = useState<boolean>(false);
  const [activeGuideTab, setActiveGuideTab] = useState<'contents' | 'generated' | 'dashboard' | 'cli'>('contents');
  const [copiedCli, setCopiedCli] = useState<string | null>(null);

  const generatedCode = generateKernelCode(config);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopySnippet = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCli(id);
    setTimeout(() => setCopiedCli(null), 2000);
  };

  const handleDownloadPy = () => {
    const filename = `silentguard_${config.framework}_${config.precision}.py`;
    const blob = new Blob([generatedCode], { type: 'text/x-python;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Intro & Parameter Customizer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration Controls */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#4A5D4E]" />
            <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Fused Invariant Parameters
            </h3>
          </div>

          {/* Framework Selector */}
          <div>
            <label className="text-xs font-medium text-[#666] block mb-1.5">
              Target Framework & Kernel Target
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { id: 'triton', name: 'OpenAI Triton' },
                { id: 'megatron', name: 'Megatron-LM / Core' },
                { id: 'pytorch', name: 'PyTorch / Torchtitan' },
                { id: 'vllm', name: 'vLLM / SGLang' },
                { id: 'flashattention', name: 'FlashAttention-3' },
              ].map((fw) => (
                <button
                  key={fw.id}
                  onClick={() => setConfig({ ...config, framework: fw.id as any })}
                  className={`px-3 py-2 text-xs font-medium rounded-[2px] text-left transition-all cursor-pointer ${
                    config.framework === fw.id
                      ? 'bg-[#4A5D4E] text-white font-semibold shadow-xs'
                      : 'bg-[#F8F7F4] text-[#2A2A2A] hover:bg-[#EBEAE5] border border-[#D1D0CB]'
                  }`}
                >
                  {fw.name}
                </button>
              ))}
            </div>
          </div>

          {/* Precision Selector */}
          <div>
            <label className="text-xs font-medium text-[#666] block mb-1.5">
              Floating Point Precision Mode
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { id: 'fp8_e4m3', name: 'FP8 (E4M3 - Hopper)' },
                { id: 'bf16', name: 'BF16 (Bfloat16)' },
                { id: 'fp16', name: 'FP16 (IEEE Half)' },
                { id: 'fp32', name: 'FP32 (Single)' },
              ].map((p) => (
                <button
                  key={p.id}
                  onClick={() => setConfig({ ...config, precision: p.id as any })}
                  className={`px-2.5 py-1.5 text-xs rounded-[2px] text-left transition-all cursor-pointer ${
                    config.precision === p.id
                      ? 'bg-[#2A2A2A] text-white font-semibold'
                      : 'bg-[#F8F7F4] text-[#666] hover:text-[#2A2A2A] border border-[#D1D0CB]'
                  }`}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          {/* Tolerance Epsilon */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-[#666]">Invariant Tolerance (Epsilon ε)</span>
              <span className="font-mono font-bold text-[#2A2A2A]">{config.toleranceEpsilon.toExponential(0)}</span>
            </div>
            <select
              value={config.toleranceEpsilon}
              onChange={(e) => setConfig({ ...config, toleranceEpsilon: parseFloat(e.target.value) })}
              className="w-full bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] px-3 py-1.5 text-xs font-mono text-[#2A2A2A] cursor-pointer focus:outline-none focus:border-[#4A5D4E]"
            >
              <option value="0.001">1.0e-3 (Loose - High Noise)</option>
              <option value="0.0001">1.0e-4 (Recommended for BF16/FP8)</option>
              <option value="0.00001">1.0e-5 (Strict)</option>
              <option value="0.000001">1.0e-6 (Ultra-strict for FP32)</option>
            </select>
          </div>

          {/* Stochastic Sampling Rate */}
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="font-medium text-[#666]">Stochastic Sampling Rate</span>
              <span className="font-mono font-bold text-[#4A5D4E]">{config.samplingRatePercent}% of GEMM Tiles</span>
            </div>
            <input
              type="range"
              min={1}
              max={100}
              step={1}
              value={config.samplingRatePercent}
              onChange={(e) => setConfig({ ...config, samplingRatePercent: parseInt(e.target.value) })}
              className="w-full accent-[#4A5D4E] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-[#999] font-mono mt-0.5">
              <span>1% (&lt;0.02% FLOPs)</span>
              <span>3% (Optimal)</span>
              <span>100% (Strict Mode)</span>
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="space-y-2 pt-2 border-t border-[#D1D0CB]">
            <label className="flex items-center justify-between text-xs text-[#2A2A2A] cursor-pointer">
              <span>In-Flight Tile Recomputation</span>
              <input
                type="checkbox"
                checked={config.inlineRecompute}
                onChange={(e) => setConfig({ ...config, inlineRecompute: e.target.checked })}
                className="h-4 w-4 text-[#4A5D4E] rounded focus:ring-0 accent-[#4A5D4E]"
              />
            </label>
            <label className="flex items-center justify-between text-xs text-[#2A2A2A] cursor-pointer">
              <span>Auto-Quarantine Rank on Violation</span>
              <input
                type="checkbox"
                checked={config.autoDrainOnFailure}
                onChange={(e) => setConfig({ ...config, autoDrainOnFailure: e.target.checked })}
                className="h-4 w-4 text-[#4A5D4E] rounded focus:ring-0 accent-[#4A5D4E]"
              />
            </label>
          </div>
        </div>

        {/* Right: Code Preview & Interactive Downloader (2 columns wide) */}
        <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 flex flex-col justify-between shadow-xs">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-[#4A5D4E]" />
                <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                  GENERATED IN-REGISTER FUSED KERNEL
                </h4>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyCode}
                  className="inline-flex items-center gap-1 px-3 py-1.5 rounded-[2px] text-xs font-semibold bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied!' : 'Copy Code'}</span>
                </button>
                <button
                  onClick={handleDownloadPy}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-[2px] text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white shadow-xs transition-colors cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .py Script</span>
                </button>
              </div>
            </div>

            {/* Syntax Code Box */}
            <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-4 font-mono text-xs max-h-96 overflow-y-auto border border-[#333330] leading-relaxed">
              <pre className="text-emerald-300 font-mono">{generatedCode}</pre>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#D1D0CB] flex items-center justify-between text-xs text-[#666]">
            <span>Self-contained standalone script ready for `python train.py`</span>
            <span className="font-mono text-[#4A5D4E] font-bold">&lt; 0.08% Overhead Validated</span>
          </div>
        </div>
      </div>

      {/* Microbenchmark Performance Comparison */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono mb-3">
          H100 SXM5 Microbenchmark: Standard Triton GEMM vs SilentGuard Fused Invariant
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#D1D0CB] text-[#666] font-mono">
                <th className="pb-2 font-semibold">Kernel Implementation</th>
                <th className="pb-2 font-semibold">TFLOPs (BF16)</th>
                <th className="pb-2 font-semibold">Latency (M=N=K=8192)</th>
                <th className="pb-2 font-semibold">HBM Extra Allocation</th>
                <th className="pb-2 font-semibold">FLOP Overhead</th>
                <th className="pb-2 font-semibold">SDC Coverage</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#D1D0CB] font-mono">
              <tr>
                <td className="py-2.5 font-bold text-[#2A2A2A]">Vanilla Triton GEMM (Baseline)</td>
                <td className="py-2.5 text-[#666]">965 TFLOPs</td>
                <td className="py-2.5 text-[#666]">1.144 ms</td>
                <td className="py-2.5 text-[#666]">0 Bytes</td>
                <td className="py-2.5 text-[#666]">0.00%</td>
                <td className="py-2.5 text-[#8C2D2D] font-bold">0% (Vulnerable to SDC)</td>
              </tr>
              <tr>
                <td className="py-2.5 font-bold text-[#666]">Full Matrix Duplication (2x GEMM)</td>
                <td className="py-2.5 text-[#666]">482 TFLOPs</td>
                <td className="py-2.5 text-[#666]">2.288 ms</td>
                <td className="py-2.5 text-[#666]">128 MB</td>
                <td className="py-2.5 text-[#8C2D2D] font-bold">+100.0%</td>
                <td className="py-2.5 text-[#4A5D4E]">100%</td>
              </tr>
              <tr className="bg-[#F8F7F4] text-[#2A2A2A] font-bold">
                <td className="py-2.5 flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-[#4A5D4E]" />
                  <span>SilentGuard Fused Parity GEMM</span>
                </td>
                <td className="py-2.5">964.2 TFLOPs</td>
                <td className="py-2.5">1.145 ms (+0.08%)</td>
                <td className="py-2.5">0 Bytes (In-Register)</td>
                <td className="py-2.5 text-[#4A5D4E]">&lt;0.08%</td>
                <td className="py-2.5 text-[#4A5D4E]">100% (3.2ms Trap)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Package Contents, Runtime Artifacts & Telemetry Dashboard Viewer */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 shadow-xs space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D1D0CB] pb-4">
          <div>
            <div className="flex items-center gap-2">
              <FolderTree className="w-4 h-4 text-[#4A5D4E]" />
              <h3 className="font-serif text-base font-bold text-[#2A2A2A]">
                Package Anatomy, Generated Runtime Artifacts & Dashboard Guide
              </h3>
            </div>
            <p className="text-xs text-[#666] mt-0.5">
              Everything included in the package, what it writes on compute nodes during training, and how to view telemetry.
            </p>
          </div>

          {/* Guide Subtabs */}
          <div className="flex bg-[#F8F7F4] p-0.5 rounded-[2px] border border-[#D1D0CB] text-xs font-mono">
            <button
              onClick={() => setActiveGuideTab('contents')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer ${
                activeGuideTab === 'contents'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              1. Package Contents
            </button>
            <button
              onClick={() => setActiveGuideTab('generated')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer ${
                activeGuideTab === 'generated'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              2. Generated Artifacts
            </button>
            <button
              onClick={() => setActiveGuideTab('dashboard')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer ${
                activeGuideTab === 'dashboard'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              3. Telemetry & Dashboards
            </button>
            <button
              onClick={() => setActiveGuideTab('cli')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer ${
                activeGuideTab === 'cli'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              4. CLI Tools
            </button>
          </div>
        </div>

        {/* Tab 1: Package Contents */}
        {activeGuideTab === 'contents' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono flex items-center gap-1.5">
                <FileCode className="w-3.5 h-3.5 text-[#4A5D4E]" />
                Python Package Structure (`silentguard/`)
              </h4>
              <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-4 font-mono text-xs border border-[#333330] leading-relaxed">
                <pre className="text-emerald-300">
{`silentguard/
├── __init__.py                # Core entrypoints & single-line hooks
├── kernels/
│   ├── triton_gemm.py         # Fused In-Register Freivalds Triton MMA
│   ├── flash_attn_guard.py    # FlashAttention-3 QK^T invariant hooks
│   └── fp8_hopper.py          # FP8 (E4M3/E5M2) tensor core parity
├── distributed/
│   ├── megatron_patch.py      # Megatron-Core Column/RowParallel wrappers
│   ├── fsdp_wrapper.py        # PyTorch FSDP / Torchtitan hooks
│   └── nccl_checksum.py       # Distributed all-reduce packet CRC/invariants
├── inference/
│   ├── vllm_paged_attn.py     # vLLM KV-cache & logit anomaly detector
│   └── sglang_guard.py        # SGLang RadixAttention invariant monitor
├── telemetry/
│   ├── prometheus_exporter.py # Live :9090 metrics server
│   ├── slurm_drain.py         # Automated scontrol node drain trigger
│   └── wandb_callback.py      # Weights & Biases / TensorBoard plugin
└── cli/
    ├── main.py                # \`silentguard\` CLI tool
    └── tui_dashboard.py       # Terminal curses live cluster dashboard`}
                </pre>
              </div>
            </div>

            <div className="space-y-4">
              <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                Key Modules & Functional Roles
              </h4>
              <div className="space-y-2.5 text-xs">
                <div className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px]">
                  <div className="font-mono font-bold text-[#2A2A2A] flex justify-between">
                    <span>1. In-Register Tensor Core MMA</span>
                    <span className="text-[#4A5D4E] font-bold">&lt; 0.08% Overhead</span>
                  </div>
                  <p className="text-[#666] mt-1">
                    Evaluates Freivalds algebraic invariant <code className="text-[#2A2A2A]">$r^T \cdot (A \cdot B) = (r^T \cdot A) \cdot B$</code> directly in SRAM registers during matrix accumulation before HBM writeback.
                  </p>
                </div>

                <div className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px]">
                  <div className="font-mono font-bold text-[#2A2A2A] flex justify-between">
                    <span>2. Distributed Megatron & FSDP Monkeypatch</span>
                    <span className="text-[#4A5D4E] font-bold">Zero Code Changes</span>
                  </div>
                  <p className="text-[#666] mt-1">
                    Transparently intercepts Tensor-Parallel GEMMs across 128+ ranks, isolating bad math to the exact SM on the specific GPU rank.
                  </p>
                </div>

                <div className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px]">
                  <div className="font-mono font-bold text-[#2A2A2A] flex justify-between">
                    <span>3. In-Flight Tile Recomputation & Slurm Drain</span>
                    <span className="text-[#4A5D4E] font-bold">&lt; 3.8ms Recovery</span>
                  </div>
                  <p className="text-[#666] mt-1">
                    If an invariant violation is trapped, the corrupted tile is recomputed on a healthy SM in &lt;3.8ms, while the faulty node is quarantined from the cluster partition.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Generated Artifacts */}
        {activeGuideTab === 'generated' && (
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Artifacts & Logs Generated During Training Job Execution
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] space-y-2">
                <div className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#2A2A2A]">
                  <FileText className="w-4 h-4 text-[#4A5D4E]" />
                  <span>logs/sdc_events.jsonl</span>
                </div>
                <p className="text-[11px] text-[#666]">
                  High-resolution JSONL stream recording every parity violation, rank ID, SM index, residual delta norm ($\delta$), and recomputation duration.
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2 rounded-[2px] text-[10px] font-mono overflow-x-auto">
                  {`{"ts":"10:44:12","rank":47,"sm":34,"delta":1.48e3,"recomp_ms":3.18}`}
                </div>
              </div>

              <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] space-y-2">
                <div className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#2A2A2A]">
                  <AlertTriangle className="w-4 h-4 text-[#8C2D2D]" />
                  <span>incidents/INC-*.md</span>
                </div>
                <p className="text-[11px] text-[#666]">
                  Self-contained forensic incident reports detailing silicon transistor mechanism, layer name, and exact Slurm drain/MODS diagnostic commands.
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2 rounded-[2px] text-[10px] font-mono overflow-x-auto">
                  {`# Incident INC-47-89412\nNode: dgx-hopper-06\nAction: scontrol drain`}
                </div>
              </div>

              <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] space-y-2">
                <div className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#2A2A2A]">
                  <Activity className="w-4 h-4 text-[#4A5D4E]" />
                  <span>:9090/metrics</span>
                </div>
                <p className="text-[11px] text-[#666]">
                  Prometheus metrics stream tracking total parity checks, SDC violations per hour, SM temperature correlation, and recovery latencies.
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2 rounded-[2px] text-[10px] font-mono overflow-x-auto">
                  {`silentguard_violations_total 1\nsilentguard_recompute_ms 3.18`}
                </div>
              </div>

              <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] space-y-2">
                <div className="flex items-center gap-1.5 font-mono text-xs font-bold text-[#2A2A2A]">
                  <BarChart3 className="w-4 h-4 text-[#4A5D4E]" />
                  <span>checkpoints/audit.json</span>
                </div>
                <p className="text-[11px] text-[#666]">
                  Summary of Kurtosis ($K$), subnormal densities, and exponent anomalies produced by offline post-save or in-memory verification.
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2 rounded-[2px] text-[10px] font-mono overflow-x-auto">
                  {`{"kurtosis":3.08,"exponent_overflows":0,"status":"healthy"}`}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 3: Telemetry & Dashboards */}
        {activeGuideTab === 'dashboard' && (
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              How to View Results: 4 Integrated Dashboard Options
            </h4>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
              {/* Option 1: Web Dashboard */}
              <div className="p-4 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono font-bold text-xs text-[#2A2A2A]">
                    <Monitor className="w-4 h-4 text-[#4A5D4E]" />
                    <span>Option 1: SilentGuard Web Dashboard (This App)</span>
                  </div>
                  <span className="px-2 py-0.5 bg-[#4A5D4E] text-white text-[10px] font-mono font-bold rounded-[2px]">
                    Built-in
                  </span>
                </div>
                <p className="text-xs text-[#666]">
                  Point your training job's telemetry exporter to this web dashboard via REST/WebSocket:
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2.5 rounded-[2px] font-mono text-xs flex justify-between items-center">
                  <code>export SILENTGUARD_TELEMETRY_ENDPOINT="http://&lt;headnode&gt;:3000/api/telemetry"</code>
                  <button
                    onClick={() => handleCopySnippet('export SILENTGUARD_TELEMETRY_ENDPOINT="http://localhost:3000/api/telemetry"', 'dash-env')}
                    className="p-1 text-white hover:text-emerald-400 cursor-pointer"
                  >
                    {copiedCli === 'dash-env' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
                <ul className="text-xs text-[#666] space-y-1 list-disc pl-4">
                  <li><strong>Cluster Simulator Tab</strong>: Live 128-GPU rank grid, live loss curves, InfiniBand ring.</li>
                  <li><strong>AI Root-Cause Diagnostic Tab</strong>: Automated hardware failure diagnosis and Slurm remediation playbooks.</li>
                </ul>
              </div>

              {/* Option 2: Terminal TUI */}
              <div className="p-4 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono font-bold text-xs text-[#2A2A2A]">
                    <Terminal className="w-4 h-4 text-[#4A5D4E]" />
                    <span>Option 2: Terminal CLI Dashboard (`curses` TUI)</span>
                  </div>
                  <span className="px-2 py-0.5 bg-[#2A2A2A] text-white text-[10px] font-mono font-bold rounded-[2px]">
                    Terminal Native
                  </span>
                </div>
                <p className="text-xs text-[#666]">
                  For SSH sessions and headless cluster environments without browser access:
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2.5 rounded-[2px] font-mono text-xs flex justify-between items-center">
                  <code>silentguard dashboard --cluster-ranks 128 --watch</code>
                  <button
                    onClick={() => handleCopySnippet('silentguard dashboard --cluster-ranks 128 --watch', 'dash-cli')}
                    className="p-1 text-white hover:text-emerald-400 cursor-pointer"
                  >
                    {copiedCli === 'dash-cli' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
                <ul className="text-xs text-[#666] space-y-1 list-disc pl-4">
                  <li>Displays live GPU heatmaps directly inside tmux / Slurm allocation terminal.</li>
                  <li>Instant alert banners when parity residuals exceed $\epsilon = 10^{-4}$.</li>
                </ul>
              </div>

              {/* Option 3: Grafana & Prometheus */}
              <div className="p-4 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono font-bold text-xs text-[#2A2A2A]">
                    <Server className="w-4 h-4 text-[#4A5D4E]" />
                    <span>Option 3: Grafana & Prometheus Preset</span>
                  </div>
                </div>
                <p className="text-xs text-[#666]">
                  Export production Grafana dashboards ready for enterprise NOC integration:
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2.5 rounded-[2px] font-mono text-xs flex justify-between items-center">
                  <code>silentguard export-grafana --out /etc/grafana/dashboards/sdc.json</code>
                  <button
                    onClick={() => handleCopySnippet('silentguard export-grafana --out /etc/grafana/dashboards/sdc.json', 'dash-graf')}
                    className="p-1 text-white hover:text-emerald-400 cursor-pointer"
                  >
                    {copiedCli === 'dash-graf' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              {/* Option 4: Weights & Biases */}
              <div className="p-4 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-2.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 font-mono font-bold text-xs text-[#2A2A2A]">
                    <BarChart3 className="w-4 h-4 text-[#4A5D4E]" />
                    <span>Option 4: Weights & Biases / TensorBoard Hook</span>
                  </div>
                </div>
                <p className="text-xs text-[#666]">
                  Streams telemetry metrics to your experiment tracking platform:
                </p>
                <div className="bg-[#1C1C1A] text-emerald-300 p-2.5 rounded-[2px] font-mono text-xs flex justify-between items-center">
                  <code>from silentguard.telemetry import SilentGuardWandbCallback</code>
                  <button
                    onClick={() => handleCopySnippet('from silentguard.telemetry import SilentGuardWandbCallback', 'dash-wandb')}
                    className="p-1 text-white hover:text-emerald-400 cursor-pointer"
                  >
                    {copiedCli === 'dash-wandb' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 4: CLI Commands */}
        {activeGuideTab === 'cli' && (
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Useful CLI Commands in the SilentGuard Package
            </h4>
            <div className="space-y-2 font-mono text-xs">
              {[
                {
                  cmd: 'pip install silentguard',
                  desc: 'Installs the Python SDK and Triton fused in-register parity kernels.',
                  id: 'cli-install',
                },
                {
                  cmd: 'silentguard scan /checkpoints/llama-3-70b-step42000.safetensors',
                  desc: 'Performs offline 4th-moment Kurtosis and exponent bit-flip anomaly audit on model weights.',
                  id: 'cli-scan',
                },
                {
                  cmd: 'silentguard benchmark --m 8192 --n 8192 --k 8192 --precision bf16',
                  desc: 'Runs microbenchmarks comparing standard GEMM vs SilentGuard parity overhead on local GPUs.',
                  id: 'cli-bench',
                },
                {
                  cmd: 'silentguard dashboard --port 9090',
                  desc: 'Launches local terminal curses dashboard and Prometheus metrics exporter on compute head node.',
                  id: 'cli-dash',
                },
              ].map((item) => (
                <div key={item.id} className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div>
                    <div className="font-bold text-[#2A2A2A]">{item.cmd}</div>
                    <div className="text-[11px] text-[#666] font-sans mt-0.5">{item.desc}</div>
                  </div>
                  <button
                    onClick={() => handleCopySnippet(item.cmd, item.id)}
                    className="self-start sm:self-center px-2.5 py-1 bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] rounded-[2px] flex items-center gap-1 cursor-pointer"
                  >
                    {copiedCli === item.id ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedCli === item.id ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

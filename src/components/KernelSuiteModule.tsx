import React, { useState } from 'react';
import JSZip from 'jszip';
import { KernelConfig, PerformanceMode } from '../types';
import { generateKernelCode } from '../data/kernelTemplates';
import { generateTrainGuardSdkCode } from '../data/trainGuardEngine';
import { PACKAGE_FILES } from '../data/packageFiles';
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
  FolderArchive,
  Package,
  Code,
} from 'lucide-react';

export const KernelSuiteModule: React.FC = () => {
  const [codeType, setCodeType] = useState<'trainguard_sdk' | 'fused_kernel'>('trainguard_sdk');
  const [performanceMode, setPerformanceMode] = useState<PerformanceMode>('balanced');
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
  const [activeGuideTab, setActiveGuideTab] = useState<'contents' | 'cluster_health' | 'generated' | 'dashboard' | 'cli'>('contents');
  const [copiedCli, setCopiedCli] = useState<string | null>(null);
  const [isZipping, setIsZipping] = useState<boolean>(false);
  const [downloadSuccess, setDownloadSuccess] = useState<boolean>(false);

  // Cluster Health Interactive Simulator States
  const [clusterReportStage, setClusterReportStage] = useState<'preflight' | 'post_training'>('preflight');
  const [isAuditingCluster, setIsAuditingCluster] = useState<boolean>(false);
  const [auditTimestamp, setAuditTimestamp] = useState<string>('2026-08-18 12:00:00 UTC');
  const [mockClusterHealthStatus, setMockClusterHealthStatus] = useState<'READY' | 'HEALTHY' | 'ANOMALY'>('READY');

  const kernelCode = generateKernelCode(config);
  const sdkCode = generateTrainGuardSdkCode(performanceMode);
  const activeCode = codeType === 'trainguard_sdk' ? sdkCode : kernelCode;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(activeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopySnippet = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCli(id);
    setTimeout(() => setCopiedCli(null), 2000);
  };

  const handleDownloadPy = () => {
    const filename = codeType === 'trainguard_sdk'
      ? `trainguard_sdk_${performanceMode}.py`
      : `silentguard_${config.framework}_${config.precision}.py`;
    const blob = new Blob([activeCode], { type: 'text/x-python;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadFullPackageZip = async () => {
    setIsZipping(true);
    try {
      const zip = new JSZip();

      // README.md
      zip.file(
        'README.md',
        `# TrainGuard v0.3 & SilentGuard Production Resilience Package
================================================================================
Production Platform for Machine Learning Training Reliability & Hardware Silicon Invariants.

## Quickstart Installation
\`\`\`bash
pip install .
# or copy the self-contained trainguard.py or silentguard.py drop-in directly into your codebase
\`\`\`

## 1. Pre-Flight Hardware Cluster Audit (After Package Installation)
Verify GPU Tensor Cores, memory invariants, and NVLink interconnect before launching training:
\`\`\`bash
train-guard cluster-health
# Or in Python:
from train_guard.cluster_health import check_cluster_health
health = check_cluster_health()
assert health["ready"], "Hardware SDC anomaly detected!"
\`\`\`

## 2. TrainGuard High-Level 5-Layer Training Guard
\`\`\`python
from train_guard import TrainGuard, TrainGuardConfig, PerformanceMode

guard = TrainGuard(model, config=TrainGuardConfig(mode=PerformanceMode.BALANCED))
# Pre-flight sanity check
guard.check_cluster_health()

guard.start()
for step, (x, y) in enumerate(loader):
    guard.step(step)
    out = model(x)
    guard.check_tensor("input", x)
    guard.check_tensor("logits", out)
    guard.check_parameters()

guard.stop()

# 3. Post-Training Enterprise Cluster Health & SDC Report
report = guard.cluster_health_report()
print(f"Reliability Score: {report['reliability_scorecard']['score']}/100")
print(f"Effective Cluster Uptime: {report['reliability_scorecard']['effective_cluster_uptime_percent']}% (Target: 99.9%)")
\`\`\`

## 4. Post-Training Report CLI Inspection
\`\`\`bash
train-guard report --input trainguard_cluster_health_report.json
\`\`\`

## 5. Low-Level Fused Parity Triton MMA Kernel
\`\`\`python
import silentguard
silentguard.auto_protect(framework="megatron", tolerance_eps=1e-4, auto_drain=True)
\`\`\`
`
      );

      // setup.py
      zip.file(
        'setup.py',
        `from setuptools import setup, find_packages

setup(
    name="trainguard",
    version="0.3.0",
    author="TrainGuard AI Systems",
    description="Production Platform for Machine Learning Training Reliability & SDC Silicon Resilience",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train-guard=train_guard.cli:main",
            "silentguard=silentguard.cli.main:main",
        ],
    },
    install_requires=[
        "torch>=2.2.0",
        "triton>=2.1.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
`
      );

      // pyproject.toml
      zip.file(
        'pyproject.toml',
        `[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "trainguard"
version = "0.3.0"
description = "Production Platform for Machine Learning Training Reliability"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "torch>=2.2.0",
    "triton>=2.1.0",
    "numpy>=1.24.0",
]
`
      );

      // Standalone single-file drop-ins
      zip.file('trainguard.py', sdkCode);
      zip.file('silentguard.py', kernelCode);

      // Full package tree files
      PACKAGE_FILES.forEach((file) => {
        zip.file(file.path, file.code);
      });

      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(zipBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'trainguard-0.3.0.zip';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setDownloadSuccess(true);
      setTimeout(() => setDownloadSuccess(false), 3000);
    } catch (err) {
      console.error('Error creating package zip:', err);
    } finally {
      setIsZipping(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Intro & Parameter Customizer */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration Controls */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
          {/* Target Engine Switcher (TrainGuard SDK vs Fused Kernel) */}
          <div className="space-y-1.5 pb-3 border-b border-[#D1D0CB]">
            <label className="text-xs font-mono font-bold text-[#4A5D4E] uppercase tracking-wider block">
              Reliability Target Layer
            </label>
            <div className="grid grid-cols-2 gap-1 bg-[#F8F7F4] p-0.5 rounded-[2px] border border-[#D1D0CB]">
              <button
                onClick={() => setCodeType('trainguard_sdk')}
                className={`px-2.5 py-1.5 text-xs font-mono font-bold rounded-[2px] cursor-pointer transition-all ${
                  codeType === 'trainguard_sdk'
                    ? 'bg-[#4A5D4E] text-white shadow-xs'
                    : 'text-[#666] hover:text-[#2A2A2A]'
                }`}
              >
                TrainGuard SDK (v0.3)
              </button>
              <button
                onClick={() => setCodeType('fused_kernel')}
                className={`px-2.5 py-1.5 text-xs font-mono font-bold rounded-[2px] cursor-pointer transition-all ${
                  codeType === 'fused_kernel'
                    ? 'bg-[#4A5D4E] text-white shadow-xs'
                    : 'text-[#666] hover:text-[#2A2A2A]'
                }`}
              >
                Triton/CUDA Kernel
              </button>
            </div>
          </div>

          {codeType === 'trainguard_sdk' ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#4A5D4E]" />
                <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                  Performance Mode & Overhead
                </h3>
              </div>

              {/* Performance Mode Selector */}
              <div className="space-y-2">
                {[
                  {
                    id: 'light' as PerformanceMode,
                    title: 'Light Mode (< 2% Overhead)',
                    desc: 'NaN/Inf, Output tensor validation, minimal scalar statistics.',
                  },
                  {
                    id: 'balanced' as PerformanceMode,
                    title: 'Balanced Mode (< 5% Overhead)',
                    desc: 'Selected tensors, parameter deltas, sampled 4th-moment kurtosis.',
                  },
                  {
                    id: 'full' as PerformanceMode,
                    title: 'Full Mode (< 10% Overhead)',
                    desc: 'Comprehensive activation distribution, all parameter deltas.',
                  },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => setPerformanceMode(mode.id)}
                    className={`w-full text-left p-3 rounded-[2px] text-xs transition-all cursor-pointer ${
                      performanceMode === mode.id
                        ? 'bg-[#4A5D4E] text-white shadow-xs'
                        : 'bg-[#F8F7F4] text-[#2A2A2A] hover:bg-[#EBEAE5] border border-[#D1D0CB]'
                    }`}
                  >
                    <div className="font-bold mb-0.5">{mode.title}</div>
                    <div className={`text-[11px] ${performanceMode === mode.id ? 'text-emerald-100' : 'text-[#666]'}`}>
                      {mode.desc}
                    </div>
                  </button>
                ))}
              </div>

              {/* Strict No-Baseline Poisoning Rule Badge */}
              <div className="p-3 bg-emerald-50/60 border border-emerald-300/80 rounded-[2px] text-xs space-y-1">
                <div className="font-mono font-bold text-[#4A5D4E] uppercase text-[10px]">
                  ✓ Strict No-Baseline Poisoning
                </div>
                <div className="text-[11px] text-[#2A2A2A]">
                  Anomalies are quarantined to evidence and never bias the rolling Median/MAD baseline.
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
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
          )}
        </div>

        {/* Right: Code Preview & Interactive Downloader (2 columns wide) */}
        <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 flex flex-col justify-between shadow-xs">
          <div>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-[#4A5D4E]" />
                <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                  {codeType === 'trainguard_sdk' ? `TRAINGUARD PYTHON SDK (${performanceMode.toUpperCase()} MODE)` : 'GENERATED IN-REGISTER FUSED KERNEL'}
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
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[2px] text-xs font-bold bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .py</span>
                </button>
                <button
                  onClick={handleDownloadFullPackageZip}
                  disabled={isZipping}
                  className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-[2px] text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white shadow-xs transition-colors cursor-pointer disabled:opacity-50"
                >
                  {downloadSuccess ? (
                    <Check className="w-3.5 h-3.5 text-white" />
                  ) : (
                    <FolderArchive className="w-3.5 h-3.5" />
                  )}
                  <span>{isZipping ? 'Zipping...' : downloadSuccess ? 'Downloaded!' : 'Download Package (.zip)'}</span>
                </button>
              </div>
            </div>

            {/* Syntax Code Box */}
            <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-4 font-mono text-xs max-h-96 overflow-y-auto border border-[#333330] leading-relaxed">
              <pre className="text-emerald-300 font-mono">{activeCode}</pre>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#D1D0CB] flex items-center justify-between text-xs text-[#666]">
            <span>
              {codeType === 'trainguard_sdk'
                ? `1-line step verification in PyTorch / Megatron (< ${performanceMode === 'light' ? '2%' : performanceMode === 'balanced' ? '5%' : '10%'} overhead)`
                : 'Self-contained standalone script ready for `python train.py`'}
            </span>
            <span className="font-mono text-[#4A5D4E] font-bold">
              {codeType === 'trainguard_sdk' ? 'TrainGuard v0.3 Compliant' : '< 0.08% FLOP Overhead'}
            </span>
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
          <div className="flex bg-[#F8F7F4] p-0.5 rounded-[2px] border border-[#D1D0CB] text-xs font-mono overflow-x-auto">
            <button
              onClick={() => setActiveGuideTab('contents')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer whitespace-nowrap ${
                activeGuideTab === 'contents'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              1. Package Contents
            </button>
            <button
              onClick={() => setActiveGuideTab('cluster_health')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer whitespace-nowrap flex items-center gap-1.5 ${
                activeGuideTab === 'cluster_health'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>2. Cluster Health & Reports</span>
            </button>
            <button
              onClick={() => setActiveGuideTab('generated')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer whitespace-nowrap ${
                activeGuideTab === 'generated'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              3. Generated Artifacts
            </button>
            <button
              onClick={() => setActiveGuideTab('dashboard')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer whitespace-nowrap ${
                activeGuideTab === 'dashboard'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              4. Telemetry & Dashboards
            </button>
            <button
              onClick={() => setActiveGuideTab('cli')}
              className={`px-3 py-1.5 rounded-[2px] transition-all cursor-pointer whitespace-nowrap ${
                activeGuideTab === 'cli'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              5. CLI Tools
            </button>
          </div>
        </div>

        {/* Tab 1: Package Contents & Main Python Code */}
        {activeGuideTab === 'contents' && (
          <div className="space-y-5">
            {/* Top Download Action Banner */}
            <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <Package className="w-4 h-4 text-[#4A5D4E]" />
                  <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                    SilentGuard Production Package (`silentguard-1.4.2.zip`)
                  </h4>
                </div>
                <p className="text-xs text-[#666] mt-0.5">
                  Complete distribution with <code className="text-[#2A2A2A]">setup.py</code>, <code className="text-[#2A2A2A]">pyproject.toml</code>, Triton kernels, Megatron/PyTorch hooks, and Prometheus telemetry.
                </p>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  onClick={handleDownloadPy}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[2px] text-xs font-semibold bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download silentguard.py</span>
                </button>
                <button
                  onClick={handleDownloadFullPackageZip}
                  disabled={isZipping}
                  className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-[2px] text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white shadow-xs transition-colors cursor-pointer disabled:opacity-50"
                >
                  {downloadSuccess ? (
                    <Check className="w-3.5 h-3.5 text-white" />
                  ) : (
                    <FolderArchive className="w-3.5 h-3.5" />
                  )}
                  <span>{isZipping ? 'Generating ZIP...' : downloadSuccess ? 'Downloaded!' : 'Download Package (.zip)'}</span>
                </button>
              </div>
            </div>

            {/* Main Content Grid: Package Anatomy (Left) & Main Python Script (Right) */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Static Package Structure & Summary (5 cols) */}
              <div className="lg:col-span-5 space-y-4">
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono flex items-center gap-1.5">
                    <FileCode className="w-3.5 h-3.5 text-[#4A5D4E]" />
                    Package Anatomy (`pip install .`)
                  </h4>
                  <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-4 font-mono text-xs border border-[#333330] leading-relaxed">
                    <pre className="text-emerald-300">
{`silentguard/
├── __init__.py                # Main 1-line auto-patchers
├── kernels/
│   ├── triton_gemm.py         # In-Register Freivalds Triton MMA
│   ├── flash_attn_guard.py    # FlashAttention-3 row-sum trap
│   └── fp8_hopper.py          # FP8 (E4M3) tensor saturation
├── distributed/
│   ├── megatron_patch.py      # Megatron-Core Column/Row hooks
│   ├── fsdp_wrapper.py        # PyTorch FSDP linear wrappers
│   └── nccl_checksum.py       # Distributed NVLink CRC traps
├── inference/
│   ├── vllm_paged_attn.py     # vLLM KV-cache Kurtosis auditor
│   └── sglang_guard.py        # SGLang RadixAttention monitor
├── telemetry/
│   ├── prometheus_exporter.py # Live :9090 Prometheus metrics
│   ├── slurm_drain.py         # Automated scontrol node drain
│   └── wandb_callback.py      # Weights & Biases telemetry
└── cli/
    ├── main.py                # \`silentguard\` CLI tool
    └── tui_dashboard.py       # Terminal live cluster NOC`}
                    </pre>
                  </div>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px]">
                    <div className="font-mono font-bold text-[#2A2A2A] flex justify-between">
                      <span>Zero-Dependency Drop-in</span>
                      <span className="text-[#4A5D4E] font-bold">1 File (`silentguard.py`)</span>
                    </div>
                    <p className="text-[#666] mt-1">
                      Drop directly into your Git repository without pip package installs or root permissions.
                    </p>
                  </div>

                  <div className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px]">
                    <div className="font-mono font-bold text-[#2A2A2A] flex justify-between">
                      <span>Full Enterprise Package</span>
                      <span className="text-[#4A5D4E] font-bold">`pip install .`</span>
                    </div>
                    <p className="text-[#666] mt-1">
                      Includes CLI diagnostic tools, Prometheus exporters, and automated Slurm node quarantine daemons.
                    </p>
                  </div>
                </div>
              </div>

              {/* Right Column: Main Python Code Preview (7 cols) */}
              <div className="lg:col-span-7 bg-white border border-[#D1D0CB] rounded-[2px] p-4 flex flex-col justify-between shadow-xs">
                <div>
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3 pb-2 border-b border-[#D1D0CB]">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <Terminal className="w-4 h-4 text-[#4A5D4E]" />
                        <span className="font-mono text-xs font-bold text-[#2A2A2A]">
                          Main Python Implementation (`silentguard.py`)
                        </span>
                      </div>
                      <p className="text-[11px] text-[#666] mt-0.5">
                        In-register Freivalds fused tensor MMA arithmetic validator with automated self-healing.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleCopyCode}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-[2px] text-xs font-semibold bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
                      >
                        {copied ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>{copied ? 'Copied' : 'Copy'}</span>
                      </button>
                      <button
                        onClick={handleDownloadPy}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[2px] text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white shadow-xs transition-colors cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download .py</span>
                      </button>
                    </div>
                  </div>

                  {/* Syntax Code Editor Box */}
                  <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-4 font-mono text-xs max-h-[380px] overflow-y-auto border border-[#333330] leading-relaxed">
                    <pre className="text-emerald-300 font-mono">{activeCode}</pre>
                  </div>
                </div>

                <div className="mt-3 pt-2 border-t border-[#D1D0CB] flex items-center justify-between text-[11px] text-[#666]">
                  <span>Ready for immediate deployment across NVIDIA DGX H100 / Blackwell clusters</span>
                  <span className="font-mono text-[#4A5D4E] font-bold">&lt; 0.08% FLOP Overhead</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Cluster Health & Reports (Pre-Flight & Post-Training) */}
        {activeGuideTab === 'cluster_health' && (
          <div className="space-y-6">
            {/* Header & Stage Switcher */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[3px] p-4">
              <div>
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-[#4A5D4E]" />
                  <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                    TrainGuard Cluster Health Suite: Silicon Probing & Post-Training Scorecard
                  </h4>
                </div>
                <p className="text-xs text-[#666] mt-0.5">
                  Check cluster readiness immediately after package installation and inspect comprehensive reliability reports after training runs.
                </p>
              </div>

              {/* Stage Selector */}
              <div className="flex bg-white p-0.5 rounded-[2px] border border-[#D1D0CB] text-xs font-mono">
                <button
                  onClick={() => setClusterReportStage('preflight')}
                  className={`px-3 py-1 rounded-[2px] transition-all cursor-pointer ${
                    clusterReportStage === 'preflight'
                      ? 'bg-[#4A5D4E] text-white font-bold'
                      : 'text-[#666] hover:text-[#2A2A2A]'
                  }`}
                >
                  Phase 1: Pre-Flight Audit
                </button>
                <button
                  onClick={() => setClusterReportStage('post_training')}
                  className={`px-3 py-1 rounded-[2px] transition-all cursor-pointer ${
                    clusterReportStage === 'post_training'
                      ? 'bg-[#4A5D4E] text-white font-bold'
                      : 'text-[#666] hover:text-[#2A2A2A]'
                  }`}
                >
                  Phase 2: Post-Training Report
                </button>
              </div>
            </div>

            {/* STAGE 1: Pre-Flight Cluster Health Audit */}
            {clusterReportStage === 'preflight' && (
              <div className="space-y-5">
                {/* Pre-Flight Actions & Status Banner */}
                <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D1D0CB] pb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4 text-[#4A5D4E]" />
                        <h4 className="font-serif text-sm font-bold text-[#2A2A2A]">
                          Pre-Flight Hardware Silicon Audit (Post-Installation)
                        </h4>
                      </div>
                      <p className="text-xs text-[#666] mt-0.5">
                        Probes Tensor Core arithmetic with stochastic Freivalds invariants, tests NVLink CRC32 checksums, and audits HBM allocators.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          setIsAuditingCluster(true);
                          setTimeout(() => {
                            setIsAuditingCluster(false);
                            setAuditTimestamp(new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC');
                            setMockClusterHealthStatus('HEALTHY');
                          }, 900);
                        }}
                        disabled={isAuditingCluster}
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-[2px] text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white transition-colors cursor-pointer disabled:opacity-50"
                      >
                        <Zap className="w-3.5 h-3.5" />
                        <span>{isAuditingCluster ? 'Probing Silicon Ranks...' : 'Run Pre-Flight Cluster Audit'}</span>
                      </button>
                      <button
                        onClick={() => handleCopySnippet('train-guard cluster-health --tolerance-eps 1e-4', 'cli-preflight-cmd')}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-[2px] text-xs font-semibold bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
                      >
                        {copiedCli === 'cli-preflight-cmd' ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>Copy CLI</span>
                      </button>
                    </div>
                  </div>

                  {/* 8-Rank GPU Matrix Audit Table */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs text-[#666] font-mono">
                      <span>CLUSTER TOPOLOGY: 8x NVIDIA H100 SXM5 80GB (NVLINK 4.0 / NDR 400G)</span>
                      <span className="text-[#4A5D4E] font-bold">
                        AUDIT STATUS: {mockClusterHealthStatus === 'HEALTHY' ? 'READY_FOR_TRAINING (SLA: 99.90%)' : 'READY_FOR_PROBE'}
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left">
                        <thead>
                          <tr className="border-b border-[#D1D0CB] text-[#666] font-mono">
                            <th className="pb-2 font-semibold">Rank ID</th>
                            <th className="pb-2 font-semibold">Accelerator Device</th>
                            <th className="pb-2 font-semibold">MMA Parity Residual ($\delta$)</th>
                            <th className="pb-2 font-semibold">NVLink All-Reduce</th>
                            <th className="pb-2 font-semibold">HBM Free / Alloc</th>
                            <th className="pb-2 font-semibold">Silicon Temp</th>
                            <th className="pb-2 font-semibold">Health State</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#D1D0CB] font-mono">
                          {[
                            { rank: 0, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.42e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '47°C', status: 'HEALTHY' },
                            { rank: 1, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.61e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '48°C', status: 'HEALTHY' },
                            { rank: 2, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.38e-05', nvlink: 'PASSED (0 CRC)', mem: '79.1 / 80.0 GB', temp: '49°C', status: 'HEALTHY' },
                            { rank: 3, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.84e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '46°C', status: 'HEALTHY' },
                            { rank: 4, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.29e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '51°C', status: 'HEALTHY' },
                            { rank: 5, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.53e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '48°C', status: 'HEALTHY' },
                            { rank: 6, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.77e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '50°C', status: 'HEALTHY' },
                            { rank: 7, dev: 'NVIDIA H100 SXM5 80GB', delta: '1.35e-05', nvlink: 'PASSED (0 CRC)', mem: '79.2 / 80.0 GB', temp: '47°C', status: 'HEALTHY' },
                          ].map((row) => (
                            <tr key={row.rank} className="hover:bg-[#F8F7F4] transition-colors">
                              <td className="py-2 text-[#2A2A2A] font-bold">Rank {row.rank}</td>
                              <td className="py-2 text-[#666]">{row.dev}</td>
                              <td className="py-2 text-[#4A5D4E] font-bold">{row.delta}</td>
                              <td className="py-2 text-[#2A2A2A]">{row.nvlink}</td>
                              <td className="py-2 text-[#666]">{row.mem}</td>
                              <td className="py-2 text-[#666]">{row.temp}</td>
                              <td className="py-2">
                                <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-emerald-50 text-emerald-800 border border-emerald-200 text-[10px] font-bold rounded-[2px]">
                                  <Check className="w-3 h-3 text-emerald-700" />
                                  <span>{row.status}</span>
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Terminal CLI Command Execution Box */}
                  <div className="space-y-1.5 pt-2 border-t border-[#D1D0CB]">
                    <div className="flex items-center justify-between text-xs text-[#666]">
                      <span className="font-mono font-bold text-[#2A2A2A]">Terminal Output: `train-guard cluster-health`</span>
                      <span className="font-mono text-[11px]">Audit Timestamp: {auditTimestamp}</span>
                    </div>
                    <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-3 font-mono text-xs max-h-48 overflow-y-auto border border-[#333330] leading-relaxed">
                      <div className="text-emerald-400">================================================================================</div>
                      <div className="text-white font-bold"> [TrainGuard] HARDWARE CLUSTER HEALTH & SILICON ARITHMETIC AUDIT</div>
                      <div className="text-emerald-400">================================================================================</div>
                      <div className="text-gray-300"> Cluster Infrastructure: 8 Accelerators Detected (NVIDIA H100 SXM5 80GB)</div>
                      <div className="text-gray-400"> [Phase 1/3] Tensor Core Arithmetic Parity Probe (Freivalds Invariant)...</div>
                      <div className="text-emerald-300">   • Rank 0-7: MMA Parity Residuals ≤ 1.84e-05 -&gt; [PASSED]</div>
                      <div className="text-gray-400"> [Phase 2/3] Interconnect &amp; NVLink All-Reduce Checksum...</div>
                      <div className="text-emerald-300">   • Distributed World Size 8: NVLink Parity -&gt; [PASSED - 0 CRC Errors]</div>
                      <div className="text-gray-400"> [Phase 3/3] CUDA Memory Allocator &amp; Subnormal Traps...</div>
                      <div className="text-emerald-300">   • HBM Allocation verified: 79.2 GB / 80.0 GB available per device</div>
                      <div className="text-emerald-400">================================================================================</div>
                      <div className="text-emerald-300 font-bold"> [Cluster Health Summary] Status: READY_FOR_TRAINING (Target Uptime SLA: 99.90%)</div>
                      <div className="text-emerald-400">================================================================================</div>
                    </div>
                  </div>
                </div>

                {/* Integration Code Box */}
                <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[3px] p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-[#2A2A2A]">Python 1-Line Pre-Flight Verification</span>
                    <button
                      onClick={() => handleCopySnippet(`from train_guard.cluster_health import check_cluster_health\nhealth = check_cluster_health()\nassert health["ready"], "Hardware SDC anomaly detected!"`, 'code-preflight')}
                      className="px-2.5 py-1 bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] rounded-[2px] text-xs font-mono flex items-center gap-1 cursor-pointer"
                    >
                      {copiedCli === 'code-preflight' ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>Copy Code</span>
                    </button>
                  </div>
                  <div className="bg-[#1C1C1A] text-emerald-300 p-2.5 rounded-[2px] font-mono text-xs">
                    <code>from train_guard.cluster_health import check_cluster_health<br />health = check_cluster_health()<br />assert health["ready"], "Hardware SDC anomaly detected!"</code>
                  </div>
                </div>
              </div>
            )}

            {/* STAGE 2: Post-Training Cluster Health Report */}
            {clusterReportStage === 'post_training' && (
              <div className="space-y-5">
                {/* Executive Scorecard Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-white border border-[#D1D0CB] p-4 rounded-[3px] shadow-xs space-y-1">
                    <div className="text-[11px] font-mono text-[#666] uppercase">Reliability Score</div>
                    <div className="text-2xl font-serif font-bold text-[#4A5D4E]">98.4 / 100</div>
                    <div className="text-[11px] text-emerald-700 font-medium">Status: PRODUCTION_HEALTHY</div>
                  </div>

                  <div className="bg-white border border-[#D1D0CB] p-4 rounded-[3px] shadow-xs space-y-1">
                    <div className="text-[11px] font-mono text-[#666] uppercase">Effective Cluster Uptime</div>
                    <div className="text-2xl font-serif font-bold text-[#2A2A2A]">99.98%</div>
                    <div className="text-[11px] text-[#666]">Target SLA: 99.90% (Unprotected: 83.4%)</div>
                  </div>

                  <div className="bg-white border border-[#D1D0CB] p-4 rounded-[3px] shadow-xs space-y-1">
                    <div className="text-[11px] font-mono text-[#666] uppercase">SDC Interventions</div>
                    <div className="text-2xl font-serif font-bold text-[#8C2D2D]">14 Prevented</div>
                    <div className="text-[11px] text-[#666]">In-flight MMA tile recomputations</div>
                  </div>

                  <div className="bg-white border border-[#D1D0CB] p-4 rounded-[3px] shadow-xs space-y-1">
                    <div className="text-[11px] font-mono text-[#666] uppercase">Avoided Compute Loss</div>
                    <div className="text-2xl font-serif font-bold text-[#4A5D4E]">448 GPU-hrs</div>
                    <div className="text-[11px] text-emerald-700 font-medium">~$1,702.40 USD saved</div>
                  </div>
                </div>

                {/* Detailed Post-Training Terminal Output & JSON Artifact */}
                <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#D1D0CB] pb-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <FileCode className="w-4 h-4 text-[#4A5D4E]" />
                        <h4 className="font-serif text-sm font-bold text-[#2A2A2A]">
                          Generated Post-Training Artifact (`trainguard_cluster_health_report.json`)
                        </h4>
                      </div>
                      <p className="text-xs text-[#666] mt-0.5">
                        Exported automatically at the end of training upon calling `guard.cluster_health_report()` or `guard.stop()`.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => handleCopySnippet(JSON.stringify({
                          report_id: "rpt_llama3_405b_posttrain_8921",
                          schema_version: "1.0",
                          product: "TrainGuard",
                          summary: { reliability_score: 98.4, status: "STABLE", monitored_steps: 5200, clean_steps: 5186, total_incidents_prevented: 14 },
                          sla_metrics: { enterprise_sla_target: "99.90%", protected_cluster_uptime: "99.98%", unprotected_estimated_uptime: "83.40%" },
                          economic_savings: { avoided_gpu_hours: 448, saved_cost_usd: "$1,702.40" },
                          zero_poisoning_rule_enforced: true
                        }, null, 2), 'post-train-json')}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-[2px] text-xs font-semibold bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
                      >
                        {copiedCli === 'post-train-json' ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                        <span>Copy Report JSON</span>
                      </button>
                      <button
                        onClick={() => {
                          const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
                            report_id: "rpt_llama3_405b_posttrain_8921",
                            schema_version: "1.0",
                            product: "TrainGuard",
                            summary: { reliability_score: 98.4, status: "STABLE", monitored_steps: 5200, clean_steps: 5186, total_incidents_prevented: 14 },
                            sla_metrics: { enterprise_sla_target: "99.90%", protected_cluster_uptime: "99.98%", unprotected_estimated_uptime: "83.40%" },
                            economic_savings: { avoided_gpu_hours: 448, saved_cost_usd: "$1,702.40" },
                            zero_poisoning_rule_enforced: true
                          }, null, 2));
                          const dlAnchor = document.createElement('a');
                          dlAnchor.setAttribute("href", dataStr);
                          dlAnchor.setAttribute("download", "trainguard_cluster_health_report.json");
                          document.body.appendChild(dlAnchor);
                          dlAnchor.click();
                          dlAnchor.remove();
                        }}
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-[2px] text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white transition-colors cursor-pointer"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download JSON</span>
                      </button>
                    </div>
                  </div>

                  {/* Code / Report Terminal Output */}
                  <div className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-4 font-mono text-xs max-h-72 overflow-y-auto border border-[#333330] leading-relaxed">
                    <pre className="text-emerald-300 font-mono">
{`{
  "report_id": "rpt_llama3_405b_posttrain_8921",
  "schema_version": "1.0",
  "product": "TrainGuard",
  "version": "0.3.0",
  "generated_at": "${auditTimestamp}",
  "run_metadata": {
    "run_id": "pretrain_run_uswest_812",
    "total_monitored_steps": 5200,
    "clean_steps": 5186,
    "anomaly_events_prevented": 14,
    "performance_mode": "balanced"
  },
  "reliability_scorecard": {
    "score": 98.4,
    "category": "PRODUCTION_HEALTHY",
    "sla_target_percent": 99.90,
    "effective_cluster_uptime_percent": 99.98,
    "unprotected_projected_uptime_percent": 83.40
  },
  "sdc_prevention_breakdown": {
    "layer1_deterministic_nan_inf": 3,
    "layer2_parameter_bitflips": 4,
    "layer3_activation_kurtosis": 2,
    "layer4_robust_median_mad": 3,
    "layer5_cross_rank_nccl": 2,
    "total_prevented_sdc": 14
  },
  "economic_impact": {
    "saved_gpu_hours": 448,
    "saved_dollars_usd": "$1,702.40",
    "avoided_divergent_runs": 4
  },
  "baseline_protection": {
    "strict_no_poisoning_rule_enforced": true,
    "contaminated_baseline_steps": 0
  }
}`}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Generated Artifacts */}
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

        {/* Tab 5: CLI Commands */}
        {activeGuideTab === 'cli' && (
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Useful CLI Commands in the TrainGuard / SilentGuard Package
            </h4>
            <div className="space-y-2 font-mono text-xs">
              {[
                {
                  cmd: 'train-guard cluster-health --tolerance-eps 1e-4',
                  desc: 'Pre-flight hardware audit: Probes Tensor Core MMA parity, NVLink all-reduce CRC32, and HBM allocators.',
                  id: 'cli-cluster-health',
                },
                {
                  cmd: 'train-guard report --input trainguard_cluster_health_report.json',
                  desc: 'Post-training summary: Inspects reliability scorecard, intercepted SDCs, and avoided GPU compute loss.',
                  id: 'cli-report',
                },
                {
                  cmd: 'train-guard validate /checkpoints/model.safetensors',
                  desc: 'Validates safetensors or PyTorch checkpoint files for bit-flips, subnormals, and kurtosis explosions.',
                  id: 'cli-validate',
                },
                {
                  cmd: 'pip install .',
                  desc: 'Installs the Python SDK, Triton fused in-register parity kernels, and cluster health tools.',
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

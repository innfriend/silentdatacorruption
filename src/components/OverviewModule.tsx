import React, { useState } from 'react';
import {
  ShieldCheck,
  Server,
  Terminal,
  Search,
  Zap,
  Calculator,
  BookOpen,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  TrendingUp,
  Layers,
  Activity,
  Gauge,
  Sliders,
  Sparkles,
  Lock,
} from 'lucide-react';
import { ModuleTab } from '../types';
import { PIPELINE_LAYERS, INITIAL_TRAINGUARD_EVENTS, calculateReliabilityScore } from '../data/trainGuardEngine';
import { ClusterHealthTrends } from './ClusterHealthTrends';

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
  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number>(0);
  const reliabilityScoreData = calculateReliabilityScore(INITIAL_TRAINGUARD_EVENTS, 5200);

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
      title: 'TrainGuard v0.3 Production SDK',
      description:
        'Official 5-Layer ML reliability SDK with Light/Balanced/Full profiles, exact parameter index localization, and 1-line PyTorch instrumentation.',
      icon: <Terminal className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Download v0.3 SDK (.zip)',
      status: 'TrainGuard v0.3 / PyTorch 2.4+',
    },
    {
      id: 'diagnostic' as ModuleTab,
      title: '7-Question Evidence & Diagnostics',
      description:
        'Standardized 7-question forensic investigator (What, Where, When, Severity, Confidence, Evidence, Next Action) with actionable remediation playbooks.',
      icon: <Zap className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Run AI Diagnostics',
      status: '7-Question Schema v1.0',
    },
    {
      id: 'scanner' as ModuleTab,
      title: 'Checkpoint & Tensor Scanner',
      description:
        'Inspect PyTorch .pt and SafeTensors weights. Automatically detect kurtosis anomalies, silent exponent overflows, and low-order mantissa drift without training interruption.',
      icon: <Search className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Scan Weights & Checkpoints',
      status: 'Kurtosis & Exponent Audit',
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
      title: 'Statistical Engine & SDC Science',
      description:
        'Deep-dive into Median/MAD mathematics, strict no-baseline poisoning safeguards, atmospheric neutron flux, and interactive IEEE 754 bit-flipper.',
      icon: <BookOpen className="w-4 h-4 text-[#4A5D4E]" />,
      actionText: 'Explore Knowledge Base',
      status: 'Median/MAD & IEEE 754',
    },
  ];

  const selectedLayer = PIPELINE_LAYERS[selectedLayerIndex];

  return (
    <div className="space-y-6">
      {/* Executive SDC Threat & Mathematical Guarantee Banner */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 sm:p-8 shadow-xs">
        <div className="max-w-4xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-[2px] bg-[#F8F7F4] text-[#4A5D4E] text-[11px] font-mono mb-3 border border-[#D1D0CB]">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span className="font-semibold tracking-wider">TRAINGUARD v0.3 PRODUCTION PLATFORM SPECIFICATION</span>
          </div>
          <h2 className="text-xl sm:text-3xl font-light tracking-tight text-[#2A2A2A] font-serif mb-3">
            Evidence-Backed ML Reliability, Tensor Integrity & SDC Defense
          </h2>
          <p className="text-xs sm:text-sm text-[#555] leading-relaxed mb-5">
            TrainGuard is a PyTorch ML reliability SDK that monitors model execution and training for deterministic numerical failures, tensor/activation integrity problems, parameter state anomalies, and statistically abnormal runtime behavior. Operating across a <strong className="text-[#2A2A2A] font-semibold">5-layer pipeline</strong> with <strong className="text-[#4A5D4E] font-semibold">strict no-baseline poisoning</strong>, TrainGuard answers the 7 critical incident questions with mathematically backed confidence.
          </p>
          <div className="flex flex-wrap gap-5 text-xs font-mono text-[#4A5D4E]">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">5-Layer Defense Architecture</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">Strict No-Baseline Poisoning</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">0–100 Reliability Score</span>
            </div>
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-[#4A5D4E]" />
              <span className="text-[#2A2A2A]">Privacy: Metadata-Only Storage</span>
            </div>
          </div>
        </div>
      </div>

      {/* TrainGuard 0-100 Reliability Scorecard & Real-time Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Scorecard (5 cols) */}
        <div className="lg:col-span-5 bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#D1D0CB]">
              <div className="flex items-center gap-2">
                <Gauge className="w-4 h-4 text-[#4A5D4E]" />
                <span className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                  TrainGuard Reliability Score
                </span>
              </div>
              <span className="px-2 py-0.5 rounded-[2px] bg-[#F8F7F4] text-[#4A5D4E] border border-[#D1D0CB] text-[10px] font-mono font-bold">
                {reliabilityScoreData.status}
              </span>
            </div>

            <div className="my-4 flex items-baseline gap-3">
              <div className="text-4xl sm:text-5xl font-light font-serif text-[#2A2A2A]">
                {reliabilityScoreData.score}
              </div>
              <span className="text-sm font-mono text-[#666]">/ 100</span>
              <span className="text-xs text-[#4A5D4E] font-medium font-mono ml-auto">
                {reliabilityScoreData.cleanStepsCount} / {reliabilityScoreData.totalMonitoredSteps} Clean Steps
              </span>
            </div>

            {/* Deductions Breakdown */}
            <div className="space-y-2 text-xs font-mono">
              <div className="flex items-center justify-between text-[#666] pb-1 border-b border-[#EBEAE5]">
                <span>Deterministic Failures Penalty</span>
                <span className="text-red-700 font-bold">-{reliabilityScoreData.deterministicDeduction} pts</span>
              </div>
              <div className="flex items-center justify-between text-[#666] pb-1 border-b border-[#EBEAE5]">
                <span>Statistical / Delta Anomalies</span>
                <span className="text-amber-700 font-bold">-{reliabilityScoreData.statisticalDeduction} pts</span>
              </div>
              <div className="flex items-center justify-between text-[#666] pb-1 border-b border-[#EBEAE5]">
                <span>Persistence / Recurrence Penalty</span>
                <span className="text-amber-700 font-bold">-{reliabilityScoreData.persistenceDeduction} pts</span>
              </div>
              <div className="flex items-center justify-between text-[#666]">
                <span>Affected Components Weighting</span>
                <span className="text-[#666] font-bold">-{reliabilityScoreData.affectedComponentsDeduction} pts</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#D1D0CB] flex items-center justify-between text-[11px] text-[#666]">
            <span className="flex items-center gap-1">
              <Lock className="w-3 h-3 text-[#4A5D4E]" />
              Baseline Shield: <strong>100% Unpoisoned</strong>
            </span>
            <span className="font-mono text-[#4A5D4E] font-bold">Weighted Algorithm v1.0</span>
          </div>
        </div>

        {/* Executive Metrics (7 cols) */}
        <div className="lg:col-span-7 grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col justify-between">
            <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Cluster Health</div>
            <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif my-1">
              {healthyRanks === totalRanks ? '100%' : `${((healthyRanks / totalRanks) * 100).toFixed(1)}%`}
            </div>
            <div className="text-[11px] text-[#4A5D4E] font-medium font-mono">
              {healthyRanks}/{totalRanks} H100 Ranks
            </div>
          </div>

          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col justify-between">
            <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Detection Latency</div>
            <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif my-1">
              3.18<span className="text-xs text-[#999] ml-1 font-sans">ms</span>
            </div>
            <div className="text-[11px] text-[#666] font-mono">In-register cache check</div>
          </div>

          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col justify-between">
            <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Overhead (Balanced)</div>
            <div className="text-2xl sm:text-3xl font-light text-[#4A5D4E] font-serif my-1">&lt; 0.08%</div>
            <div className="text-[11px] text-[#666] font-mono">Profile: Balanced Mode</div>
          </div>

          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col justify-between">
            <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Monitored Steps</div>
            <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif my-1">5,200</div>
            <div className="text-[11px] text-[#666] font-mono">Continuous validation</div>
          </div>

          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col justify-between">
            <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Prevented Spikes</div>
            <div className="text-2xl sm:text-3xl font-light text-[#2A2A2A] font-serif my-1">{14 + recomputedTiles}</div>
            <div className="text-[11px] text-[#4A5D4E] font-medium font-mono">0 Lost Iterations</div>
          </div>

          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 flex flex-col justify-between">
            <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">Avoided Waste</div>
            <div className="text-2xl sm:text-3xl font-light text-[#4A5D4E] font-serif my-1">$1.42M+</div>
            <div className="text-[11px] text-[#666] font-mono">70B model run basis</div>
          </div>
        </div>
      </div>

      {/* Enterprise Cluster Health Trends Visualization (Recharts) */}
      <ClusterHealthTrends />

      {/* Interactive 5-Layer ML Reliability Architecture Pipeline */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#D1D0CB] pb-3">
          <div>
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#4A5D4E]" />
              <h3 className="text-sm font-bold text-[#2A2A2A] tracking-wider uppercase font-mono">
                TrainGuard 5-Layer Reliability & Detection Pipeline
              </h3>
            </div>
            <p className="text-xs text-[#666] mt-0.5">
              Click any layer to inspect deterministic integrity checks, parameter tracking, and robust statistical safeguards.
            </p>
          </div>
          <span className="text-[11px] text-[#4A5D4E] font-mono font-bold">
            All 5 Layers Online
          </span>
        </div>

        {/* 5 Layer Navigation Tabs */}
        <div className="grid grid-cols-1 sm:grid-cols-5 gap-2">
          {PIPELINE_LAYERS.map((layer, idx) => {
            const isSelected = selectedLayerIndex === idx;
            return (
              <button
                key={layer.number}
                onClick={() => setSelectedLayerIndex(idx)}
                className={`p-3 rounded-[2px] text-left border transition-all cursor-pointer flex flex-col justify-between ${
                  isSelected
                    ? 'bg-[#4A5D4E] text-white border-[#4A5D4E] shadow-xs'
                    : 'bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border-[#D1D0CB]'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono uppercase font-bold mb-1">
                    <span>L{layer.number}</span>
                    <span className={isSelected ? 'text-emerald-200' : 'text-[#4A5D4E]'}>{layer.overhead}</span>
                  </div>
                  <div className="text-xs font-serif font-bold line-clamp-1">{layer.name.replace(/Layer \d: /, '')}</div>
                </div>
                <div className={`text-[10px] font-mono mt-2 truncate ${isSelected ? 'text-emerald-100' : 'text-[#666]'}`}>
                  {layer.tagline}
                </div>
              </button>
            );
          })}
        </div>

        {/* Selected Layer Inspection Card */}
        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-4 grid grid-cols-1 md:grid-cols-12 gap-5">
          <div className="md:col-span-7 space-y-2">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded-[2px] bg-[#4A5D4E] text-white text-[10px] font-mono font-bold">
                Layer {selectedLayer.number}
              </span>
              <h4 className="text-sm font-serif font-bold text-[#2A2A2A]">{selectedLayer.name}</h4>
            </div>
            <p className="text-xs text-[#555] leading-relaxed">
              {selectedLayer.description}
            </p>
            <div className="pt-2 flex items-center gap-4 text-xs font-mono text-[#666]">
              <span>Warmup: <strong className="text-[#2A2A2A]">{selectedLayer.warmupRequired ? '50-100 Steps' : 'Zero Warmup (Instant)'}</strong></span>
              <span>Overhead: <strong className="text-[#4A5D4E]">{selectedLayer.overhead}</strong></span>
            </div>
          </div>

          <div className="md:col-span-5 bg-white border border-[#D1D0CB] rounded-[2px] p-3 space-y-2">
            <div className="text-[11px] font-mono font-bold text-[#2A2A2A] uppercase tracking-wider">
              Active Checks & Invariants:
            </div>
            <ul className="space-y-1 text-xs font-mono text-[#4A5D4E]">
              {selectedLayer.checks.map((chk, i) => (
                <li key={i} className="flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5 shrink-0 text-[#4A5D4E]" />
                  <span className="text-[#2A2A2A]">{chk}</span>
                </li>
              ))}
            </ul>
          </div>
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
    </div>
  );
};



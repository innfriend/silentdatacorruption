import React from 'react';
import { ChevronRight, Server, Shield, Sparkles, RefreshCw, AlertTriangle } from 'lucide-react';
import { ModuleTab } from '../types';

interface SubHeaderProps {
  activeTab: ModuleTab;
  setActiveTab: (tab: ModuleTab) => void;
  onResetCluster: () => void;
  onQuickFaultInject: () => void;
}

const MODULE_META: Record<ModuleTab, { title: string; subtitle: string; tag: string }> = {
  overview: {
    title: 'Executive Platform Overview',
    subtitle: 'Cluster-wide SDC telemetry, compute overhead analytics, and platform capability launcher',
    tag: 'EXECUTIVE DASHBOARD',
  },
  simulator: {
    title: 'Live Cluster Simulator & Network Operations Center (NOC)',
    subtitle: '128-rank GPU topology grid with in-flight cosmic ray bit-flip injector and live loss curve tracker',
    tag: 'REAL-TIME NOC',
  },
  kernels: {
    title: 'Developer SDK & Fused Kernel Suite',
    subtitle: 'Triton, PyTorch FSDP, DeepSpeed, and FlashAttention stochastic parity kernels with in-register verification',
    tag: 'KERNEL COMPILER',
  },
  scanner: {
    title: 'Checkpoint & Tensor Log Scanner',
    subtitle: 'Offline kurtosis anomaly detector, exponent overflow inspector, and layer-by-layer bit drift analyzer',
    tag: 'FORENSIC SCANNER',
  },
  diagnostic: {
    title: 'Gemini-Powered AI Root Cause Diagnostic',
    subtitle: 'Autonomous GPU microarchitecture investigator for ALU carry-chain faults, voltage droop, and SEU analysis',
    tag: 'AI FORENSICS',
  },
  roi: {
    title: 'Compute Waste & Cluster ROI Calculator',
    subtitle: 'Financial modeler estimating saved GPU-hours, prevented loss spikes, and pre-training dollar savings',
    tag: 'FINANCIAL MODEL',
  },
  commercial: {
    title: 'Enterprise Commercial Suite & Fleet Licensing',
    subtitle: 'B2B subscription tiers, Slurm daemonset manifests, Prometheus telemetry exporters, and licensing specs',
    tag: 'FLEET LICENSING',
  },
  science: {
    title: 'SDC Physics, Silicon Mechanics & IEEE 754 Knowledge Base',
    subtitle: 'Mathematical proof of stochastic parity invariants, why hardware ECC misses ALU SDCs, and bit-flipper simulator',
    tag: 'TECHNICAL WHITEPAPER',
  },
};

export const SubHeader: React.FC<SubHeaderProps> = ({
  activeTab,
  setActiveTab,
  onResetCluster,
  onQuickFaultInject,
}) => {
  const meta = MODULE_META[activeTab];

  return (
    <div className="bg-[#F8F7F4] border-b border-[#D1D0CB] py-3.5 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          {/* Breadcrumb */}
          <div className="flex items-center gap-1.5 text-[11px] text-[#666] font-mono mb-1">
            <span className="hover:text-[#2A2A2A] cursor-pointer" onClick={() => setActiveTab('overview')}>
              SilentGuard
            </span>
            <ChevronRight className="w-3 h-3 text-[#999]" />
            <span className="text-[#4A5D4E] font-semibold tracking-wider">{meta.tag}</span>
          </div>

          <div className="flex items-center gap-3">
            <h1 className="text-lg sm:text-2xl font-light text-[#2A2A2A] tracking-tight font-serif">
              {meta.title}
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-[#666] mt-0.5 max-w-3xl">
            {meta.subtitle}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2 shrink-0">
          {activeTab === 'simulator' && (
            <>
              <button
                onClick={onQuickFaultInject}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#8C2D2D] hover:bg-[#722424] text-white text-xs font-semibold rounded-[2px] transition-colors shadow-xs cursor-pointer"
              >
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Inject Fault</span>
              </button>
              <button
                onClick={onResetCluster}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] text-xs font-medium rounded-[2px] transition-colors cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reset Cluster</span>
              </button>
            </>
          )}

          {activeTab !== 'simulator' && (
            <button
              onClick={() => setActiveTab('simulator')}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white text-xs font-medium rounded-[2px] shadow-xs transition-colors cursor-pointer"
            >
              <Server className="w-3.5 h-3.5" />
              <span>Open Cluster NOC</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

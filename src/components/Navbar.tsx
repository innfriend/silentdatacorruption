import React from 'react';
import { ShieldCheck, Activity, Cpu, Server, Terminal, Search, Zap, BookOpen, Calculator, DollarSign } from 'lucide-react';
import { ModuleTab } from '../types';

interface NavbarProps {
  activeTab: ModuleTab;
  setActiveTab: (tab: ModuleTab) => void;
  healthyCount: number;
  corruptedCount: number;
  quarantinedCount: number;
  recomputingCount: number;
  totalGpus: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  healthyCount,
  corruptedCount,
  quarantinedCount,
  recomputingCount,
  totalGpus,
}) => {
  const isAllNominal = corruptedCount === 0 && quarantinedCount === 0 && recomputingCount === 0;

  const navItems: { id: ModuleTab; label: string; badge?: string }[] = [
    { id: 'overview', label: 'OVERVIEW' },
    { id: 'simulator', label: 'LIVE CLUSTER NOC', badge: corruptedCount > 0 ? `${corruptedCount} SDC` : undefined },
    { id: 'kernels', label: 'SDK & KERNELS' },
    { id: 'scanner', label: 'LOG FORENSICS' },
    { id: 'diagnostic', label: 'AI DIAGNOSTIC', badge: 'GEMINI' },
    { id: 'roi', label: 'ROI CALCULATOR' },
    { id: 'commercial', label: 'FLEET LICENSING' },
    { id: 'science', label: 'SDC SCIENCE' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white shadow-xs">
      {/* Top Editorial Nav Header (56px) */}
      <div className="h-14 border-b border-[#D1D0CB] flex items-center justify-between px-4 sm:px-6 lg:px-8 bg-white">
        {/* Brand & Platform Identity */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setActiveTab('overview')}
            className="flex items-center gap-3 text-left focus:outline-none group cursor-pointer"
          >
            <div className="w-6 h-6 rounded-[3px] bg-[#4A5D4E] flex items-center justify-center text-white shrink-0 shadow-xs">
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-base sm:text-lg text-[#2A2A2A] font-sans">
                SILENTGUARD<span className="text-[#4A5D4E]">ENTERPRISE</span>
              </span>
              <span className="font-mono text-[10px] bg-[#E4E3E0] px-2 py-0.5 rounded-[3px] text-[#666] font-medium tracking-wide">
                v2.4.0-STABLE
              </span>
            </div>
          </button>
        </div>

        {/* Right Status Cluster & Controls */}
        <div className="flex items-center gap-3 sm:gap-4">
          {/* Health Pill */}
          <div className="flex items-center gap-2 text-[11px] font-semibold tracking-wider font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                isAllNominal ? 'bg-[#4A5D4E]' : 'bg-[#F27D26] animate-pulse'
              }`}
            />
            <span className={isAllNominal ? 'text-[#4A5D4E]' : 'text-[#D96B1E]'}>
              {isAllNominal ? 'CLUSTER HEALTH: NOMINAL' : `${corruptedCount + quarantinedCount} ANOMALIES DETECTED`}
            </span>
          </div>

          <div className="hidden sm:block h-5 w-[1px] bg-[#D1D0CB]" />

          {/* Quick Metrics */}
          <div className="hidden lg:flex items-center gap-3 text-[11px] font-mono text-[#666]">
            <span>
              RANKS: <strong className="text-[#2A2A2A] font-semibold">{healthyCount}/{totalGpus}</strong>
            </span>
            <span>
              OVERHEAD: <strong className="text-[#4A5D4E] font-semibold">&lt;0.08%</strong>
            </span>
          </div>

          <div className="hidden sm:block h-5 w-[1px] bg-[#D1D0CB]" />

          {/* Admin / Fault Injector quick trigger */}
          <button
            onClick={() => setActiveTab('simulator')}
            className="bg-transparent border border-[#2A2A2A] hover:bg-[#2A2A2A] hover:text-white px-3 py-1 text-[11px] font-semibold text-[#2A2A2A] transition-colors cursor-pointer rounded-[2px]"
          >
            ADMIN CONSOLE
          </button>
        </div>
      </div>

      {/* Secondary Tab Bar (48px) */}
      <div className="h-12 border-b border-[#D1D0CB] flex items-center px-4 sm:px-6 lg:px-8 gap-4 sm:gap-6 lg:gap-8 bg-[#F8F7F4] overflow-x-auto scrollbar-none">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`h-full flex items-center gap-1.5 text-xs uppercase tracking-wider transition-all whitespace-nowrap cursor-pointer ${
                isActive
                  ? 'font-bold text-[#2A2A2A] border-b-2 border-[#4A5D4E]'
                  : 'font-medium text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              <span>{item.label}</span>
              {item.badge && (
                <span
                  className={`text-[9px] font-mono px-1.5 py-0.5 rounded-[2px] ${
                    isActive
                      ? 'bg-[#4A5D4E] text-white font-bold'
                      : 'bg-[#E4E3E0] text-[#666]'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </header>
  );
};


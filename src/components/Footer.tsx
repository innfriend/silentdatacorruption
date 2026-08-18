import React from 'react';
import { ShieldCheck, Lock, Cpu } from 'lucide-react';
import { ModuleTab } from '../types';

interface FooterProps {
  setActiveTab: (tab: ModuleTab) => void;
}

export const Footer: React.FC<FooterProps> = ({ setActiveTab }) => {
  return (
    <footer className="bg-white text-[#666] border-t border-[#D1D0CB] py-8 mt-12 text-xs font-mono">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 rounded-[2px] bg-[#4A5D4E] flex items-center justify-center text-white">
              <ShieldCheck className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-[#2A2A2A] font-bold tracking-tight font-sans text-xs">
                SILENTGUARD <span className="text-[#4A5D4E]">ENTERPRISE</span>
              </div>
              <div className="text-[10px] text-[#999]">
                Autonomous SDC & GPU Cluster Reliability Platform • v2.4.0-prod
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-4 text-xs font-sans text-[#666]">
            <button onClick={() => setActiveTab('overview')} className="hover:text-[#2A2A2A] transition-colors cursor-pointer">
              Overview
            </button>
            <button onClick={() => setActiveTab('simulator')} className="hover:text-[#2A2A2A] transition-colors cursor-pointer">
              Cluster NOC
            </button>
            <button onClick={() => setActiveTab('kernels')} className="hover:text-[#2A2A2A] transition-colors cursor-pointer">
              SDK & Kernels
            </button>
            <button onClick={() => setActiveTab('diagnostic')} className="hover:text-[#2A2A2A] transition-colors cursor-pointer">
              AI Diagnostic
            </button>
            <button onClick={() => setActiveTab('roi')} className="hover:text-[#2A2A2A] transition-colors cursor-pointer">
              ROI Calculator
            </button>
            <button onClick={() => setActiveTab('science')} className="hover:text-[#2A2A2A] transition-colors cursor-pointer">
              SDC Science
            </button>
          </div>

          <div className="flex items-center gap-3 text-[11px] text-[#888]">
            <span className="flex items-center gap-1">
              <Lock className="w-3 h-3 text-[#4A5D4E]" /> Air-Gapped Ready
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <Cpu className="w-3 h-3 text-[#4A5D4E]" /> Hopper & Blackwell ASICs
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
};


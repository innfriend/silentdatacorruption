import React, { useState } from 'react';
import {
  Calculator,
  TrendingUp,
  DollarSign,
  AlertTriangle,
  ShieldCheck,
  Flame,
  CheckCircle2,
  PieChart,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export const RoiCalculatorModule: React.FC = () => {
  const [clusterGpus, setClusterGpus] = useState<number>(2048);
  const [gpuHourlyRate, setGpuHourlyRate] = useState<number>(3.80);
  const [trainingDays, setTrainingDays] = useState<number>(45);
  const [sdcFrequencyGpuDays, setSdcFrequencyGpuDays] = useState<number>(500); // 1 SDC every 500 GPU-days
  const [rollbackHours, setRollbackHours] = useState<number>(6); // 6 hours lost rewinding to last clean checkpoint

  // Financial calculations
  const totalGpuDays = (clusterGpus * trainingDays);
  const totalGpuHours = clusterGpus * trainingDays * 24;
  const baseTrainingCost = totalGpuHours * gpuHourlyRate;

  // Expected SDC events during training run
  const expectedSdcEvents = Math.max(1, Math.round(totalGpuDays / sdcFrequencyGpuDays));
  
  // Hours lost on unmitigated cluster rewinds
  // When an SDC happens, the entire cluster must rewind to the last clean checkpoint
  const wastedGpuHoursPerEvent = clusterGpus * rollbackHours;
  const totalWastedGpuHours = expectedSdcEvents * wastedGpuHoursPerEvent;
  const totalWastedDollars = totalWastedGpuHours * gpuHourlyRate;

  // SilentGuard Software cost ($0.035 / GPU-hr)
  const silentGuardSoftwareCost = totalGpuHours * 0.035;
  const netSavings = Math.max(0, totalWastedDollars - silentGuardSoftwareCost);
  const roiMultiplier = (netSavings / Math.max(1, silentGuardSoftwareCost)).toFixed(1);

  const chartData = [
    {
      name: 'Unprotected Cluster',
      'Effective Compute Cost': Number((baseTrainingCost / 1e6).toFixed(2)),
      'SDC Rollback Waste': Number((totalWastedDollars / 1e6).toFixed(2)),
      'SilentGuard License': 0,
    },
    {
      name: 'SilentGuard Protected',
      'Effective Compute Cost': Number((baseTrainingCost / 1e6).toFixed(2)),
      'SDC Rollback Waste': 0,
      'SilentGuard License': Number((silentGuardSoftwareCost / 1e6).toFixed(2)),
    },
  ];

  return (
    <div className="space-y-6">
      {/* Top Slider Input Controls & Metrics Banner */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Financial Sliders */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
          <div className="flex items-center gap-2">
            <Calculator className="w-4 h-4 text-[#4A5D4E]" />
            <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Pre-Training Run Parameters
            </h3>
          </div>

          {/* Cluster Size */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1 text-[#666]">
              <span>Active GPUs in Cluster</span>
              <span className="font-mono text-[#4A5D4E] font-bold">{clusterGpus.toLocaleString()} GPUs</span>
            </div>
            <input
              type="range"
              min={64}
              max={16384}
              step={64}
              value={clusterGpus}
              onChange={(e) => setClusterGpus(parseInt(e.target.value))}
              className="w-full accent-[#4A5D4E] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-[#999] mt-0.5">
              <span>64</span>
              <span>2,048</span>
              <span>16,384</span>
            </div>
          </div>

          {/* GPU Rate */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1 text-[#666]">
              <span>GPU Rate ($ / Hour)</span>
              <span className="font-mono text-[#2A2A2A] font-bold">${gpuHourlyRate.toFixed(2)}/hr</span>
            </div>
            <input
              type="range"
              min={1.5}
              max={8.0}
              step={0.1}
              value={gpuHourlyRate}
              onChange={(e) => setGpuHourlyRate(parseFloat(e.target.value))}
              className="w-full accent-[#4A5D4E] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-[#999] mt-0.5">
              <span>$1.50 (Spot)</span>
              <span>$3.80 (H100)</span>
              <span>$8.00 (B200)</span>
            </div>
          </div>

          {/* Training Duration */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1 text-[#666]">
              <span>Pre-Training Duration (Days)</span>
              <span className="font-mono text-[#2A2A2A] font-bold">{trainingDays} Days</span>
            </div>
            <input
              type="range"
              min={7}
              max={180}
              step={1}
              value={trainingDays}
              onChange={(e) => setTrainingDays(parseInt(e.target.value))}
              className="w-full accent-[#4A5D4E] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-[#999] mt-0.5">
              <span>7 Days</span>
              <span>45 Days</span>
              <span>180 Days</span>
            </div>
          </div>

          {/* Checkpoint Rollback Penalty */}
          <div>
            <div className="flex justify-between text-xs font-medium mb-1 text-[#666]">
              <span>Rollback Penalty per SDC</span>
              <span className="font-mono text-[#8C2D2D] font-bold">{rollbackHours} Hours lost</span>
            </div>
            <input
              type="range"
              min={1}
              max={24}
              step={1}
              value={rollbackHours}
              onChange={(e) => setRollbackHours(parseInt(e.target.value))}
              className="w-full accent-[#8C2D2D] cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-[#999] mt-0.5">
              <span>1 hr (Frequent Chkpt)</span>
              <span>6 hrs (Standard)</span>
              <span>24 hrs</span>
            </div>
          </div>
        </div>

        {/* Right: Net Financial Savings & ROI Output (2 Columns) */}
        <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 flex flex-col justify-between shadow-xs">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-xs font-mono text-[#666]">NET FINANCIAL IMPACT:</div>
                <div className="text-2xl sm:text-3xl font-light text-[#4A5D4E] font-serif">
                  ${(netSavings / 1e6).toFixed(2)} Million Saved
                </div>
              </div>
              <div className="bg-[#4A5D4E] text-white px-3.5 py-1.5 rounded-[2px] text-xs font-mono font-bold">
                {roiMultiplier}x ROI MULTIPLIER
              </div>
            </div>

            {/* Financial Breakdown Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs mb-4">
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">EXPECTED SDC FAULTS</div>
                <div className="text-lg font-bold text-[#8C2D2D]">{expectedSdcEvents} Events</div>
                <div className="text-[10px] text-[#666]">1 per {sdcFrequencyGpuDays} GPU-days</div>
              </div>

              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">PREVENTED LOST GPU-HRS</div>
                <div className="text-lg font-bold text-[#2A2A2A]">{totalWastedGpuHours.toLocaleString()} hrs</div>
                <div className="text-[10px] text-[#666]">Rollback avoided</div>
              </div>

              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">TOTAL WASTED $ WITHOUT SG</div>
                <div className="text-lg font-bold text-[#8C2D2D]">${(totalWastedDollars / 1e6).toFixed(2)}M</div>
                <div className="text-[10px] text-[#666]">Burnt compute</div>
              </div>

              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">SILENTGUARD SOFTWARE $</div>
                <div className="text-lg font-bold text-[#4A5D4E]">${(silentGuardSoftwareCost / 1e3).toFixed(1)}k</div>
                <div className="text-[10px] text-[#666]">$0.035 / GPU-hr</div>
              </div>
            </div>
          </div>

          <div className="p-3 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] text-xs text-[#2A2A2A] leading-relaxed">
            <strong>Key Takeaway:</strong> On a {clusterGpus}-GPU cluster over a {trainingDays}-day run, unmitigated SDC events would trigger <strong>{expectedSdcEvents} catastrophic checkpoint rewinds</strong>, burning <strong>${(totalWastedDollars / 1e6).toFixed(2)}M</strong> in compute. SilentGuard traps every violation in &lt;3.8ms with zero lost iterations.
          </div>
        </div>
      </div>

      {/* Financial Comparison Chart: Unprotected vs SilentGuard */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono mb-3">
          Pre-Training Budget Breakdown ($ Millions)
        </h4>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#EBEAE5" />
              <XAxis dataKey="name" stroke="#666" tick={{ fontSize: 11, fontFamily: 'monospace' }} />
              <YAxis stroke="#666" tick={{ fontSize: 11, fontFamily: 'monospace' }} unit="M" />
              <Tooltip
                formatter={(val) => `$${val}M`}
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#D1D0CB',
                  color: '#2A2A2A',
                  fontSize: '12px',
                  fontFamily: 'monospace',
                  borderRadius: '2px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                }}
              />
              <Legend />
              <Bar dataKey="Effective Compute Cost" stackId="a" fill="#4A5D4E" />
              <Bar dataKey="SDC Rollback Waste" stackId="a" fill="#8C2D2D" />
              <Bar dataKey="SilentGuard License" stackId="a" fill="#2A2A2A" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

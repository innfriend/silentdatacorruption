import React, { useState, useEffect, useRef } from 'react';
import {
  GpuRank,
  TelemetryLog,
  LossDataPoint,
  FaultType,
} from '../types';
import {
  Server,
  AlertTriangle,
  Play,
  Pause,
  RefreshCw,
  Zap,
  Activity,
  ShieldCheck,
  Flame,
  Radio,
  Filter,
  CheckCircle2,
  XCircle,
  Sliders,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface SimulatorModuleProps {
  ranks: GpuRank[];
  setRanks: React.Dispatch<React.SetStateAction<GpuRank[]>>;
  logs: TelemetryLog[];
  setLogs: React.Dispatch<React.SetStateAction<TelemetryLog[]>>;
  lossData: LossDataPoint[];
  setLossData: React.Dispatch<React.SetStateAction<LossDataPoint[]>>;
  onResetCluster: () => void;
  recomputedTiles: number;
  setRecomputedTiles: React.Dispatch<React.SetStateAction<number>>;
}

const FAULT_DETAILS: Record<FaultType, { name: string; desc: string; severity: string; bitPattern: string }> = {
  exponent_msb: {
    name: 'MSB Exponent Bit-Flip (FP32/BF16)',
    desc: 'Toggles Bit 14/30 exponent MSB. Multiplies tensor scalar by ~1.84e19, causing instant AdamW gradient explosion.',
    severity: 'critical',
    bitPattern: '0b1100000000000000 -> Exponent Bit toggled',
  },
  mantissa_bit: {
    name: 'Low-Order Mantissa Bit-Drift',
    desc: 'Stuck bit in lower mantissa. Causes slow, silent training loss stagnation without triggering hardware NaN traps.',
    severity: 'warning',
    bitPattern: '0b0011111110000001 -> Low Mantissa bit stuck',
  },
  alu_carry_chain: {
    name: 'ALU Adder Carry-Chain Fault',
    desc: 'Fails to propagate high carry bit during matrix summation, inverting sign of partial dot product.',
    severity: 'critical',
    bitPattern: '0b0111111111111111 -> Carry chain drop',
  },
  subnormal_grad: {
    name: 'Subnormal Gradient NaN Poisoning',
    desc: 'Denormalized float flush-to-zero failure generating Inf / NaN during backprop all-reduce.',
    severity: 'critical',
    bitPattern: '0b0000000000000001 -> Denormal failure',
  },
  voltage_droop: {
    name: 'Dynamic Voltage Droop (L*di/dt)',
    desc: 'Transient Vdd drop from 0.85V to 0.67V in 14ns during FlashAttention GEMM transition.',
    severity: 'warning',
    bitPattern: 'Vdd transient < 0.68V on SM 34',
  },
  bus_drift: {
    name: 'NVLink Bus Signal Degradation',
    desc: 'High-frequency parity drift across NVLink 4.0 lane during all_gather collective.',
    severity: 'warning',
    bitPattern: 'NVLink Port 7 Eye Margin < 12%',
  },
};

export const SimulatorModule: React.FC<SimulatorModuleProps> = ({
  ranks,
  setRanks,
  logs,
  setLogs,
  lossData,
  setLossData,
  onResetCluster,
  recomputedTiles,
  setRecomputedTiles,
}) => {
  const [viewMode, setViewMode] = useState<'grid' | 'nccl'>('grid');
  const [selectedRank, setSelectedRank] = useState<GpuRank | null>(ranks[47] || ranks[0]);
  const [targetRankId, setTargetRankId] = useState<number>(47);
  const [selectedFault, setSelectedFault] = useState<FaultType>('exponent_msb');
  const [autoDrain, setAutoDrain] = useState<boolean>(true);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(42125);
  const [logFilter, setLogFilter] = useState<string>('all');
  const [searchLog, setSearchLog] = useState<string>('');

  // NCCL Ring Simulation State
  const [ncclStep, setNcclStep] = useState<number>(3);
  const [ncclPhase, setNcclPhase] = useState<'scatter-reduce' | 'all-gather'>('scatter-reduce');
  const [ncclCorruptedNode, setNcclCorruptedNode] = useState<number | null>(null);

  const logContainerRef = useRef<HTMLDivElement>(null);

  const handleInjectNcclFault = (nodeIdx: number) => {
    setNcclCorruptedNode(nodeIdx);
    const timeStr = new Date().toTimeString().split(' ')[0] + '.' + Math.floor(Math.random() * 900 + 100);

    const log: TelemetryLog = {
      id: `nccl-${Date.now()}`,
      timestamp: timeStr,
      rank: nodeIdx * 8,
      nodeId: `dgx-hopper-${(nodeIdx + 1).toString().padStart(2, '0')}`,
      severity: 'critical',
      event: 'NCCL_PACKET_CRC_VIOLATION',
      details: `InfiniBand NDR packet bit-flip on Ring chunk [${ncclStep}/8]. Invariant checksum failed on Node ${nodeIdx + 1}.`,
      deltaNorm: 9.42e2,
      durationMs: 1.84,
    };

    setLogs((prev) => [log, ...prev.slice(0, 49)]);

    setTimeout(() => {
      setNcclCorruptedNode(null);
      const healLog: TelemetryLog = {
        id: `nccl-heal-${Date.now()}`,
        timestamp: new Date().toTimeString().split(' ')[0] + '.' + Math.floor(Math.random() * 900 + 100),
        rank: nodeIdx * 8,
        nodeId: `dgx-hopper-${(nodeIdx + 1).toString().padStart(2, '0')}`,
        severity: 'recovery',
        event: 'NCCL_IN_FLIGHT_PACKET_RESEND',
        details: `Clean packet re-sent across InfiniBand QDR/NDR fabric in 1.84ms. Collective all_reduce unharmed.`,
        durationMs: 1.84,
      };
      setLogs((prev) => [healLog, ...prev.slice(0, 49)]);
    }, 1500);
  };

  const currentStepRef = useRef<number>(currentStep);
  currentStepRef.current = currentStep;
  const ranksRef = useRef<GpuRank[]>(ranks);
  ranksRef.current = ranks;
  const autoDrainRef = useRef<boolean>(autoDrain);
  autoDrainRef.current = autoDrain;

  // Auto-simulation interval
  useEffect(() => {
    if (!isSimulating) return;

    const timer = setInterval(() => {
      const nextStep = currentStepRef.current + 5;
      setCurrentStep(nextStep);
      currentStepRef.current = nextStep;

      const baseLoss = 2.11 - (nextStep - 42080) * 0.0004;

      setLossData((prevData) => {
        const lastPoint = prevData[prevData.length - 1];
        const hasActiveCorruption = ranksRef.current.some((r) => r.status === 'corrupted');

        const newUnprotected = hasActiveCorruption
          ? (lastPoint ? lastPoint.unprotectedLoss * 1.85 + 2.5 : 8.4)
          : Math.max(0.85, baseLoss + (Math.random() * 0.004 - 0.002));

        const newProtected = Math.max(0.85, baseLoss + (Math.random() * 0.003 - 0.0015));

        const newPoint: LossDataPoint = {
          step: nextStep,
          unprotectedLoss: Number(newUnprotected.toFixed(4)),
          silentGuardLoss: Number(newProtected.toFixed(4)),
          injected: hasActiveCorruption,
          recovered: false,
        };

        return [...prevData.slice(-18), newPoint];
      });

      // Occasional background cosmic ray (15% chance per tick if no current corruption)
      if (Math.random() < 0.15 && !ranksRef.current.some((r) => r.status === 'corrupted')) {
        const randomRankId = Math.floor(Math.random() * ranksRef.current.length);
        triggerFault(randomRankId, 'exponent_msb', false);
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [isSimulating]);

  // Inject a fault on a rank
  const triggerFault = (rankId: number, fault: FaultType, manual: boolean = true) => {
    const timeStr = new Date().toTimeString().split(' ')[0] + '.' + Math.floor(Math.random() * 900 + 100);
    const faultInfo = FAULT_DETAILS[fault];

    setRanks((prev) =>
      prev.map((r) => {
        if (r.id === rankId) {
          return {
            ...r,
            status: 'corrupted',
            lastFault: fault,
            lastFaultTime: timeStr,
            parityDelta: 1.482e3,
          };
        }
        return r;
      })
    );

    // Add log
    const newLog: TelemetryLog = {
      id: `log-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      timestamp: timeStr,
      rank: rankId,
      nodeId: ranks[rankId]?.nodeId || 'dgx-hopper-06',
      severity: faultInfo.severity as any,
      event: 'PARITY_INVARIANT_VIOLATION',
      details: `${faultInfo.name} detected in Tensor Core. ||r^T*C - (r^T*A)*B||_inf = 1.482e+03 > 1.0e-4 eps.`,
      deltaNorm: 1.482e3,
      durationMs: 3.18,
    };

    setLogs((prev) => [newLog, ...prev.slice(0, 49)]);

    // Update loss curve with an injection marker
    setLossData((prevData) => {
      const lastPoint = prevData[prevData.length - 1] || { step: currentStep, unprotectedLoss: 2.116, silentGuardLoss: 2.116 };
      return [
        ...prevData.slice(-18),
        {
          step: currentStep,
          unprotectedLoss: Number((lastPoint.unprotectedLoss * 2.4).toFixed(4)),
          silentGuardLoss: lastPoint.silentGuardLoss,
          injected: true,
          recovered: false,
        },
      ];
    });

    // Automatically trigger SilentGuard in-flight recomputation and healing
    setTimeout(() => {
      healRank(rankId, autoDrain);
    }, 1200);
  };

  const healRank = (rankId: number, shouldDrain: boolean) => {
    const timeStr = new Date().toTimeString().split(' ')[0] + '.' + Math.floor(Math.random() * 900 + 100);

    setRanks((prev) =>
      prev.map((r) => {
        if (r.id === rankId) {
          return {
            ...r,
            status: shouldDrain ? 'quarantined' : 'healthy',
            parityDelta: 1.2e-6,
            recomputedTiles: r.recomputedTiles + 1,
          };
        }
        return r;
      })
    );

    setRecomputedTiles((prev) => prev + 1);

    const recoveryLog: TelemetryLog = {
      id: `log-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      timestamp: timeStr,
      rank: rankId,
      nodeId: ranks[rankId]?.nodeId || 'dgx-hopper-06',
      severity: 'recovery',
      event: 'IN_FLIGHT_RECOMPUTE_SUCCESS',
      details: `Tile recomputed in 3.18ms on SM 35. Loss curve protected. ${
        shouldDrain ? 'Rank marked Slurm DRAIN.' : 'Rank returned to pool.'
      }`,
      durationMs: 3.18,
    };

    setLogs((prev) => [recoveryLog, ...prev.slice(0, 49)]);
  };

  const filteredLogs = logs.filter((log) => {
    if (logFilter !== 'all' && log.severity !== logFilter) return false;
    if (searchLog) {
      const q = searchLog.toLowerCase();
      return (
        log.nodeId.toLowerCase().includes(q) ||
        log.event.toLowerCase().includes(q) ||
        log.details.toLowerCase().includes(q) ||
        log.rank.toString().includes(q)
      );
    }
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Top NOC Status & Simulation Controller Bar */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 flex flex-col md:flex-row md:items-center md:justify-between gap-4 shadow-xs">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-[2px] bg-[#4A5D4E] text-white flex items-center justify-center font-bold text-xs font-mono">
            NOC
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-serif text-lg font-light text-[#2A2A2A]">
                128-Rank DGX H100 GPU Topology Grid
              </h3>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-[2px] text-[10px] font-mono bg-[#F8F7F4] text-[#4A5D4E] border border-[#D1D0CB]">
                <Radio className="w-3 h-3 text-[#4A5D4E] animate-pulse" />
                <span>TELEMETRY STREAMING (100 Hz)</span>
              </span>
            </div>
            <p className="text-xs text-[#666]">
              16 Nodes × 8 GPUs/Node (1,024 SMs Total) | In-Register Stochastic Parity Invariant Active
            </p>
          </div>
        </div>

        {/* Sim Controls */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Mode Switcher */}
          <div className="flex bg-[#F8F7F4] p-0.5 rounded-[2px] border border-[#D1D0CB] text-xs font-mono">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1 rounded-[2px] transition-all cursor-pointer ${
                viewMode === 'grid'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              128-Rank Topology
            </button>
            <button
              onClick={() => setViewMode('nccl')}
              className={`px-3 py-1 rounded-[2px] transition-all cursor-pointer ${
                viewMode === 'nccl'
                  ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                  : 'text-[#666] hover:text-[#2A2A2A]'
              }`}
            >
              NCCL Ring & InfiniBand
            </button>
          </div>

          <button
            onClick={() => setIsSimulating(!isSimulating)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[2px] text-xs font-semibold shadow-xs transition-colors cursor-pointer ${
              isSimulating
                ? 'bg-[#F27D26] hover:bg-[#D96B1E] text-white'
                : 'bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white'
            }`}
          >
            {isSimulating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isSimulating ? 'Pause Live Sim' : 'Start Live Sim'}</span>
          </button>

          <button
            onClick={onResetCluster}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-[2px] text-xs font-medium bg-white hover:bg-[#F8F7F4] text-[#2A2A2A] border border-[#D1D0CB] transition-colors cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reset 128 Ranks</span>
          </button>
        </div>
      </div>

      {/* Main Grid + Fault Injector + Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Either 128-Rank Grid OR NCCL Ring All-Reduce */}
        {viewMode === 'grid' ? (
          <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                  GPU Rank Map (Click Any Rank to Inspect)
                </h4>
                <p className="text-xs text-[#666]">
                  Each block represents an NVIDIA H100 SXM5 GPU rank with live parity telemetry
                </p>
              </div>
              {/* Status Legend */}
              <div className="flex items-center gap-3 text-[11px] font-mono">
                <span className="flex items-center gap-1 text-[#2A2A2A]">
                  <span className="w-2.5 h-2.5 rounded-[1px] bg-[#4A5D4E]"></span> Healthy
                </span>
                <span className="flex items-center gap-1 text-[#8C2D2D]">
                  <span className="w-2.5 h-2.5 rounded-[1px] bg-[#8C2D2D] animate-pulse"></span> Corrupted (SDC)
                </span>
                <span className="flex items-center gap-1 text-[#F27D26]">
                  <span className="w-2.5 h-2.5 rounded-[1px] bg-[#F27D26]"></span> Quarantined
                </span>
              </div>
            </div>

            {/* 16 Nodes Grid (each node contains 8 GPUs) */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Array.from({ length: 16 }, (_, nodeIdx) => {
                const nodeNum = (nodeIdx + 1).toString().padStart(2, '0');
                const nodeRanks = ranks.slice(nodeIdx * 8, (nodeIdx + 1) * 8);

                return (
                  <div
                    key={nodeIdx}
                    className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB] hover:border-[#4A5D4E] transition-colors"
                  >
                    <div className="text-[10px] font-mono font-bold text-[#666] mb-1.5 flex justify-between">
                      <span>NODE {nodeNum}</span>
                      <span className="text-[#999]">8x H100</span>
                    </div>

                    {/* 8 GPU Ranks Grid */}
                    <div className="grid grid-cols-4 gap-1">
                      {nodeRanks.map((rank) => {
                        const isSelected = selectedRank?.id === rank.id;
                        let bgClass = 'bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white';
                        if (rank.status === 'corrupted') {
                          bgClass = 'bg-[#8C2D2D] text-white animate-pulse';
                        } else if (rank.status === 'quarantined') {
                          bgClass = 'bg-[#F27D26] text-white';
                        } else if (rank.status === 'recomputing') {
                          bgClass = 'bg-blue-600 text-white animate-pulse';
                        }

                        return (
                          <button
                            key={rank.id}
                            onClick={() => {
                              setSelectedRank(rank);
                              setTargetRankId(rank.id);
                            }}
                            title={`Rank ${rank.rank} (${rank.nodeId} GPU ${rank.gpuIndex}) - ${rank.status.toUpperCase()}`}
                            className={`h-7 text-[10px] font-mono font-semibold rounded-[2px] flex items-center justify-center transition-all cursor-pointer ${bgClass} ${
                              isSelected ? 'ring-2 ring-offset-1 ring-[#2A2A2A]' : ''
                            }`}
                          >
                            {rank.rank}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          /* NCCL Ring All-Reduce & InfiniBand Simulator */
          <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                    NCCL Ring All-Reduce & InfiniBand NDR 400G Simulator
                  </h4>
                  <p className="text-xs text-[#666]">
                    Simulating 8-Node Ring Collective | Phase: <span className="font-mono font-bold uppercase text-[#4A5D4E]">{ncclPhase}</span> (Chunk {ncclStep}/8)
                  </p>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setNcclPhase(ncclPhase === 'scatter-reduce' ? 'all-gather' : 'scatter-reduce')}
                    className="px-2.5 py-1 text-xs font-mono bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] rounded-[2px] cursor-pointer"
                  >
                    Toggle Phase
                  </button>
                  <button
                    onClick={() => setNcclStep((s) => (s % 8) + 1)}
                    className="px-2.5 py-1 text-xs font-mono bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white font-bold rounded-[2px] cursor-pointer"
                  >
                    Step Ring (+1)
                  </button>
                </div>
              </div>

              {/* Ring Visualization: 8 Nodes in a Circle/Ring Pattern */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                {Array.from({ length: 8 }, (_, nodeIdx) => {
                  const isCorrupt = ncclCorruptedNode === nodeIdx;
                  return (
                    <div
                      key={nodeIdx}
                      className={`p-3 rounded-[2px] border transition-all text-xs font-mono ${
                        isCorrupt
                          ? 'bg-[#8C2D2D] text-white border-red-800 animate-pulse'
                          : 'bg-[#F8F7F4] text-[#2A2A2A] border-[#D1D0CB]'
                      }`}
                    >
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-bold">NODE {(nodeIdx + 1).toString().padStart(2, '0')}</span>
                        <span className="text-[10px] opacity-80">Ranks {nodeIdx * 8}-{nodeIdx * 8 + 7}</span>
                      </div>
                      <div className="text-[11px] opacity-90 mb-2">
                        NVLink 4.0: 900 GB/s
                        <br />
                        IB Port: NDR 400G
                      </div>
                      <button
                        onClick={() => handleInjectNcclFault(nodeIdx)}
                        className={`w-full py-1 text-[10px] font-bold rounded-[1px] cursor-pointer transition-colors ${
                          isCorrupt
                            ? 'bg-white text-[#8C2D2D]'
                            : 'bg-[#2A2A2A] hover:bg-black text-white'
                        }`}
                      >
                        {isCorrupt ? 'Corrupted!' : 'Inject Packet SDC'}
                      </button>
                    </div>
                  );
                })}
              </div>

              {/* Invariant Transmission Pipeline Banner */}
              <div className="bg-[#1C1C1A] text-white p-3.5 rounded-[2px] border border-[#333330] font-mono text-xs space-y-1.5">
                <div className="flex items-center justify-between text-emerald-400 font-bold">
                  <span>NCCL IN-FLIGHT PACKET INVARIANT: CRC-64 + RADEMACHER CHECKSUM</span>
                  <span className="text-[11px] text-[#888880]">Ring Latency: 4.8 μs/hop</span>
                </div>
                <p className="text-[11px] text-[#D1D0CB] leading-relaxed">
                  Every NCCL gradient buffer packet embeds a 64-bit projection invariant. If an InfiniBand SerDes transceiver or NVLink switch experiences parity drift, the corrupted packet is quarantined and re-requested in <strong className="text-emerald-400">&lt;2ms</strong> without halting the 128-rank distributed training collective.
                </p>
              </div>
            </div>

            <div className="mt-3 pt-2 border-t border-[#D1D0CB] flex justify-between items-center text-xs text-[#666]">
              <span>Scatter-Reduce Bandwidth Efficiency: 2 × (N - 1) / N × Matrix Size</span>
              <span className="text-[#4A5D4E] font-bold font-mono">ZERO NCCL DEADLOCK GUARANTEE</span>
            </div>
          </div>
        )}

        {/* Fault Injector & Selected Rank Inspector */}
        <div className="space-y-4">
          {/* Fault Injector Card */}
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4 text-[#8C2D2D]" />
              <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                Cosmic Ray & SDC Fault Injector
              </h4>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs font-medium text-[#666] block mb-1">
                  Target Rank (0 - 127)
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    min={0}
                    max={127}
                    value={targetRankId}
                    onChange={(e) => {
                      const val = parseInt(e.target.value) || 0;
                      setTargetRankId(Math.max(0, Math.min(127, val)));
                      setSelectedRank(ranks[val] || ranks[0]);
                    }}
                    className="w-full bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] px-3 py-1.5 text-xs font-mono text-[#2A2A2A] focus:outline-none focus:border-[#4A5D4E]"
                  />
                  <button
                    onClick={() => {
                      const r = Math.floor(Math.random() * 128);
                      setTargetRankId(r);
                      setSelectedRank(ranks[r]);
                    }}
                    className="px-2.5 py-1 text-xs font-mono bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] rounded-[2px] cursor-pointer"
                  >
                    Random
                  </button>
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-[#666] block mb-1">
                  SDC / SDE Microarchitecture Fault
                </label>
                <select
                  value={selectedFault}
                  onChange={(e) => setSelectedFault(e.target.value as FaultType)}
                  className="w-full bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] px-3 py-1.5 text-xs text-[#2A2A2A] focus:outline-none focus:border-[#4A5D4E] cursor-pointer"
                >
                  {Object.entries(FAULT_DETAILS).map(([key, val]) => (
                    <option key={key} value={key}>
                      {val.name}
                    </option>
                  ))}
                </select>
                <p className="text-[11px] text-[#666] mt-1 italic leading-relaxed">
                  {FAULT_DETAILS[selectedFault].desc}
                </p>
              </div>

              <div className="flex items-center justify-between pt-1">
                <span className="text-xs text-[#2A2A2A] font-medium">
                  Auto-Drain Slurm on SDC
                </span>
                <input
                  type="checkbox"
                  checked={autoDrain}
                  onChange={(e) => setAutoDrain(e.target.checked)}
                  className="h-4 w-4 text-[#4A5D4E] rounded focus:ring-0 cursor-pointer accent-[#4A5D4E]"
                />
              </div>

              <button
                onClick={() => triggerFault(targetRankId, selectedFault)}
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-[#8C2D2D] hover:bg-[#722424] text-white text-xs font-bold rounded-[2px] shadow-xs transition-colors cursor-pointer"
              >
                <Zap className="w-4 h-4" />
                <span>Inject Bit-Flip into Rank {targetRankId}</span>
              </button>
            </div>
          </div>

          {/* Selected Rank Inspector Card */}
          {selectedRank && (
            <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-[#4A5D4E]" />
                  <h4 className="text-xs font-bold text-[#2A2A2A] font-mono">
                    RANK {selectedRank.rank} INSPECTOR
                  </h4>
                </div>
                <span
                  className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded-[2px] uppercase ${
                    selectedRank.status === 'healthy'
                      ? 'bg-[#F8F7F4] text-[#4A5D4E] border border-[#D1D0CB]'
                      : selectedRank.status === 'corrupted'
                      ? 'bg-[#8C2D2D] text-white animate-pulse'
                      : 'bg-[#F27D26] text-white'
                  }`}
                >
                  {selectedRank.status}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono mb-3">
                <div className="bg-[#F8F7F4] p-2 rounded-[2px] border border-[#D1D0CB]">
                  <div className="text-[#999] text-[10px]">NODE / GPU</div>
                  <div className="font-bold text-[#2A2A2A]">
                    {selectedRank.nodeId} (GPU {selectedRank.gpuIndex})
                  </div>
                </div>
                <div className="bg-[#F8F7F4] p-2 rounded-[2px] border border-[#D1D0CB]">
                  <div className="text-[#999] text-[10px]">TEMP / POWER</div>
                  <div className="font-bold text-[#2A2A2A]">
                    {selectedRank.temperature}°C / {selectedRank.powerWatts}W
                  </div>
                </div>
                <div className="bg-[#F8F7F4] p-2 rounded-[2px] border border-[#D1D0CB]">
                  <div className="text-[#999] text-[10px]">SM UTILIZATION</div>
                  <div className="font-bold text-[#2A2A2A]">{selectedRank.smUtilization}%</div>
                </div>
                <div className="bg-[#F8F7F4] p-2 rounded-[2px] border border-[#D1D0CB]">
                  <div className="text-[#999] text-[10px]">PARITY INVARIANT DELTA</div>
                  <div
                    className={`font-bold ${
                      selectedRank.parityDelta > 1e-4 ? 'text-[#8C2D2D]' : 'text-[#4A5D4E]'
                    }`}
                  >
                    {selectedRank.parityDelta.toExponential(2)}
                  </div>
                </div>
              </div>

              {selectedRank.status !== 'healthy' && (
                <button
                  onClick={() => healRank(selectedRank.id, false)}
                  className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white text-xs font-semibold rounded-[2px] transition-colors cursor-pointer"
                >
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>Restore Rank to Active Pool</span>
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Real-time Loss Curve: Unprotected Divergence vs SilentGuard Zero-Loss Recovery */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
          <div>
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Pre-Training Loss Divergence Comparison
            </h4>
            <p className="text-xs text-[#666]">
              Real-time comparison: Catastrophic loss explosion on unmitigated SDC vs SilentGuard in-flight invariant recompute
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-[#8C2D2D] font-semibold">
              <span className="w-3 h-0.5 bg-[#8C2D2D]"></span> Unprotected (Divergence / NaN)
            </span>
            <span className="flex items-center gap-1.5 text-[#4A5D4E] font-semibold">
              <span className="w-3 h-0.5 bg-[#4A5D4E]"></span> SilentGuard (3.2ms Recovery)
            </span>
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#EBEAE5" />
              <XAxis
                dataKey="step"
                stroke="#666"
                tick={{ fontSize: 11, fontFamily: 'monospace' }}
                tickFormatter={(val) => `Step ${val}`}
              />
              <YAxis
                stroke="#666"
                tick={{ fontSize: 11, fontFamily: 'monospace' }}
                domain={[0, 10]}
              />
              <Tooltip
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
              <Line
                type="monotone"
                dataKey="unprotectedLoss"
                name="Unprotected Loss"
                stroke="#8C2D2D"
                strokeWidth={2}
                dot={{ r: 3, fill: '#8C2D2D' }}
                activeDot={{ r: 5 }}
              />
              <Line
                type="monotone"
                dataKey="silentGuardLoss"
                name="SilentGuard Protected"
                stroke="#4A5D4E"
                strokeWidth={2}
                dot={{ r: 3, fill: '#4A5D4E' }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Millisecond Telemetry Log Stream */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-[#4A5D4E]" />
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Real-Time Telemetry & Quarantine Event Stream
            </h4>
          </div>

          <div className="flex items-center gap-2">
            {/* Filter pills */}
            <div className="flex bg-[#F8F7F4] p-0.5 rounded-[2px] border border-[#D1D0CB] text-[11px] font-mono">
              {['all', 'info', 'warning', 'critical', 'recovery'].map((lvl) => (
                <button
                  key={lvl}
                  onClick={() => setLogFilter(lvl)}
                  className={`px-2 py-0.5 rounded-[2px] uppercase cursor-pointer ${
                    logFilter === lvl
                      ? 'bg-[#4A5D4E] text-white font-bold'
                      : 'text-[#666] hover:text-[#2A2A2A]'
                  }`}
                >
                  {lvl}
                </button>
              ))}
            </div>

            <input
              type="text"
              placeholder="Search logs..."
              value={searchLog}
              onChange={(e) => setSearchLog(e.target.value)}
              className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] px-2.5 py-1 text-xs text-[#2A2A2A] placeholder-[#999] focus:outline-none focus:border-[#4A5D4E]"
            />
          </div>
        </div>

        {/* Logs terminal box */}
        <div
          ref={logContainerRef}
          className="bg-[#1C1C1A] text-[#D1D0CB] rounded-[2px] p-3 font-mono text-xs max-h-60 overflow-y-auto space-y-1.5 border border-[#333330]"
        >
          {filteredLogs.length === 0 ? (
            <div className="text-[#888880] py-4 text-center">No telemetry logs matching filter</div>
          ) : (
            filteredLogs.map((log) => {
              let tagColor = 'text-blue-400 bg-blue-950/40 border-blue-800';
              if (log.severity === 'critical') tagColor = 'text-red-400 bg-red-950/40 border-red-800 font-bold';
              if (log.severity === 'warning') tagColor = 'text-amber-400 bg-amber-950/40 border-amber-800';
              if (log.severity === 'recovery') tagColor = 'text-emerald-400 bg-emerald-950/40 border-emerald-800 font-bold';

              return (
                <div key={log.id} className="flex items-start gap-2 text-[11px] leading-relaxed border-b border-[#2A2A28] pb-1">
                  <span className="text-[#888880] shrink-0">[{log.timestamp}]</span>
                  <span className={`px-1.5 py-0.2 rounded border text-[10px] shrink-0 ${tagColor}`}>
                    {log.event}
                  </span>
                  <span className="text-[#A6A59F] shrink-0">
                    [Rank {log.rank} / {log.nodeId}]
                  </span>
                  <span className="text-[#F2F1ED]">{log.details}</span>
                  {log.durationMs && (
                    <span className="text-[#4A5D4E] text-[10px] ml-auto shrink-0 font-bold">
                      +{log.durationMs}ms
                    </span>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

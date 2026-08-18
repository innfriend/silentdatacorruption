export interface HealthTrendPoint {
  timestamp: string;
  label: string;
  preventedSdcEvents: number;
  uptimePercent: number;
  unprotectedUptimePercent: number;
  savedGpuHours: number;
  savedDollars: number;
  layer1Events: number; // Deterministic NaN/Inf
  layer2Events: number; // Parameter Delta / Bit-Flips
  layer3Events: number; // Activation Kurtosis
  layer4Events: number; // Statistical Median/MAD
  layer5Events: number; // NCCL / Cross-Rank
  recomputedTiles: number;
  quarantinedNodes: number;
}

export interface EnterpriseClusterProfile {
  id: string;
  name: string;
  gpuModel: string;
  gpuCount: number;
  workload: string;
  slaTarget: number;
}

export const CLUSTER_PROFILES: EnterpriseClusterProfile[] = [
  {
    id: 'all',
    name: 'All Production Fleets (Aggregated)',
    gpuModel: 'NVIDIA H100 SXM5 / B200',
    gpuCount: 16384,
    workload: 'Distributed Foundation Model Training & RLHF',
    slaTarget: 99.9,
  },
  {
    id: 'llama405b',
    name: 'Fleet 01: Llama-3 405B Pretraining',
    gpuModel: 'NVIDIA H100 SXM5 80GB',
    gpuCount: 4096,
    workload: 'Megatron-Core 3D Parallelism (TP=8, PP=8, DP=64)',
    slaTarget: 99.95,
  },
  {
    id: 'deepseek_v3',
    name: 'Fleet 02: MoE Pretraining (671B Params)',
    gpuModel: 'NVIDIA H100 SXM5 80GB',
    gpuCount: 2048,
    workload: 'Expert Parallelism (EP=64, FP8 Mixed Precision)',
    slaTarget: 99.9,
  },
  {
    id: 'multimodal',
    name: 'Fleet 03: Multimodal Vision-Language Pretrain',
    gpuModel: 'NVIDIA B200 NVL72',
    gpuCount: 1024,
    workload: 'Torchtitan FSDP2 + FlashAttention-3',
    slaTarget: 99.9,
  },
];

export const HOURLY_24H_DATA: HealthTrendPoint[] = [
  { timestamp: '12:00', label: '12:00', preventedSdcEvents: 2, uptimePercent: 100.0, unprotectedUptimePercent: 91.2, savedGpuHours: 128, savedDollars: 486, layer1Events: 1, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 14, quarantinedNodes: 0 },
  { timestamp: '13:00', label: '13:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 94.5, savedGpuHours: 64, savedDollars: 243, layer1Events: 0, layer2Events: 0, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 8, quarantinedNodes: 0 },
  { timestamp: '14:00', label: '14:00', preventedSdcEvents: 0, uptimePercent: 100.0, unprotectedUptimePercent: 100.0, savedGpuHours: 0, savedDollars: 0, layer1Events: 0, layer2Events: 0, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 0, quarantinedNodes: 0 },
  { timestamp: '15:00', label: '15:00', preventedSdcEvents: 3, uptimePercent: 99.98, unprotectedUptimePercent: 82.1, savedGpuHours: 384, savedDollars: 1459, layer1Events: 1, layer2Events: 1, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 32, quarantinedNodes: 1 },
  { timestamp: '16:00', label: '16:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 96.0, savedGpuHours: 96, savedDollars: 364, layer1Events: 0, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 12, quarantinedNodes: 0 },
  { timestamp: '17:00', label: '17:00', preventedSdcEvents: 0, uptimePercent: 100.0, unprotectedUptimePercent: 100.0, savedGpuHours: 0, savedDollars: 0, layer1Events: 0, layer2Events: 0, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 0, quarantinedNodes: 0 },
  { timestamp: '18:00', label: '18:00', preventedSdcEvents: 2, uptimePercent: 100.0, unprotectedUptimePercent: 88.4, savedGpuHours: 256, savedDollars: 972, layer1Events: 0, layer2Events: 1, layer3Events: 0, layer4Events: 1, layer5Events: 0, recomputedTiles: 18, quarantinedNodes: 0 },
  { timestamp: '19:00', label: '19:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 95.2, savedGpuHours: 80, savedDollars: 304, layer1Events: 0, layer2Events: 0, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 6, quarantinedNodes: 0 },
  { timestamp: '20:00', label: '20:00', preventedSdcEvents: 4, uptimePercent: 99.96, unprotectedUptimePercent: 74.0, savedGpuHours: 512, savedDollars: 1945, layer1Events: 2, layer2Events: 1, layer3Events: 0, layer4Events: 1, layer5Events: 0, recomputedTiles: 45, quarantinedNodes: 1 },
  { timestamp: '21:00', label: '21:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 93.8, savedGpuHours: 112, savedDollars: 425, layer1Events: 0, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 9, quarantinedNodes: 0 },
  { timestamp: '22:00', label: '22:00', preventedSdcEvents: 0, uptimePercent: 100.0, unprotectedUptimePercent: 100.0, savedGpuHours: 0, savedDollars: 0, layer1Events: 0, layer2Events: 0, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 0, quarantinedNodes: 0 },
  { timestamp: '23:00', label: '23:00', preventedSdcEvents: 2, uptimePercent: 100.0, unprotectedUptimePercent: 89.1, savedGpuHours: 220, savedDollars: 836, layer1Events: 1, layer2Events: 0, layer3Events: 0, layer4Events: 1, layer5Events: 0, recomputedTiles: 16, quarantinedNodes: 0 },
  { timestamp: '00:00', label: '00:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 96.2, savedGpuHours: 64, savedDollars: 243, layer1Events: 0, layer2Events: 0, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 7, quarantinedNodes: 0 },
  { timestamp: '01:00', label: '01:00', preventedSdcEvents: 3, uptimePercent: 99.97, unprotectedUptimePercent: 78.5, savedGpuHours: 410, savedDollars: 1558, layer1Events: 1, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 1, recomputedTiles: 28, quarantinedNodes: 1 },
  { timestamp: '02:00', label: '02:00', preventedSdcEvents: 0, uptimePercent: 100.0, unprotectedUptimePercent: 100.0, savedGpuHours: 0, savedDollars: 0, layer1Events: 0, layer2Events: 0, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 0, quarantinedNodes: 0 },
  { timestamp: '03:00', label: '03:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 95.0, savedGpuHours: 72, savedDollars: 273, layer1Events: 0, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 11, quarantinedNodes: 0 },
  { timestamp: '04:00', label: '04:00', preventedSdcEvents: 2, uptimePercent: 100.0, unprotectedUptimePercent: 87.0, savedGpuHours: 240, savedDollars: 912, layer1Events: 1, layer2Events: 0, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 21, quarantinedNodes: 0 },
  { timestamp: '05:00', label: '05:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 94.0, savedGpuHours: 90, savedDollars: 342, layer1Events: 0, layer2Events: 0, layer3Events: 0, layer4Events: 1, layer5Events: 0, recomputedTiles: 5, quarantinedNodes: 0 },
  { timestamp: '06:00', label: '06:00', preventedSdcEvents: 0, uptimePercent: 100.0, unprotectedUptimePercent: 100.0, savedGpuHours: 0, savedDollars: 0, layer1Events: 0, layer2Events: 0, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 0, quarantinedNodes: 0 },
  { timestamp: '07:00', label: '07:00', preventedSdcEvents: 2, uptimePercent: 100.0, unprotectedUptimePercent: 86.5, savedGpuHours: 260, savedDollars: 988, layer1Events: 1, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 19, quarantinedNodes: 0 },
  { timestamp: '08:00', label: '08:00', preventedSdcEvents: 4, uptimePercent: 99.95, unprotectedUptimePercent: 71.2, savedGpuHours: 540, savedDollars: 2052, layer1Events: 1, layer2Events: 2, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 52, quarantinedNodes: 1 },
  { timestamp: '09:00', label: '09:00', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 95.8, savedGpuHours: 85, savedDollars: 323, layer1Events: 0, layer2Events: 0, layer3Events: 1, layer4Events: 0, layer5Events: 0, recomputedTiles: 8, quarantinedNodes: 0 },
  { timestamp: '10:00', label: '10:00', preventedSdcEvents: 2, uptimePercent: 100.0, unprotectedUptimePercent: 88.0, savedGpuHours: 230, savedDollars: 874, layer1Events: 1, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 17, quarantinedNodes: 0 },
  { timestamp: '11:00', label: '11:00 (Now)', preventedSdcEvents: 1, uptimePercent: 100.0, unprotectedUptimePercent: 96.1, savedGpuHours: 68, savedDollars: 258, layer1Events: 0, layer2Events: 1, layer3Events: 0, layer4Events: 0, layer5Events: 0, recomputedTiles: 10, quarantinedNodes: 0 },
];

export const DAILY_7D_DATA: HealthTrendPoint[] = [
  { timestamp: '2026-08-12', label: 'Wed Aug 12', preventedSdcEvents: 18, uptimePercent: 99.98, unprotectedUptimePercent: 84.2, savedGpuHours: 2450, savedDollars: 9310, layer1Events: 6, layer2Events: 7, layer3Events: 3, layer4Events: 2, layer5Events: 0, recomputedTiles: 184, quarantinedNodes: 2 },
  { timestamp: '2026-08-13', label: 'Thu Aug 13', preventedSdcEvents: 22, uptimePercent: 99.99, unprotectedUptimePercent: 81.0, savedGpuHours: 3100, savedDollars: 11780, layer1Events: 8, layer2Events: 9, layer3Events: 3, layer4Events: 1, layer5Events: 1, recomputedTiles: 215, quarantinedNodes: 2 },
  { timestamp: '2026-08-14', label: 'Fri Aug 14', preventedSdcEvents: 15, uptimePercent: 100.0, unprotectedUptimePercent: 87.5, savedGpuHours: 1980, savedDollars: 7524, layer1Events: 5, layer2Events: 5, layer3Events: 4, layer4Events: 1, layer5Events: 0, recomputedTiles: 142, quarantinedNodes: 1 },
  { timestamp: '2026-08-15', label: 'Sat Aug 15', preventedSdcEvents: 12, uptimePercent: 100.0, unprotectedUptimePercent: 89.8, savedGpuHours: 1620, savedDollars: 6156, layer1Events: 3, layer2Events: 4, layer3Events: 3, layer4Events: 2, layer5Events: 0, recomputedTiles: 110, quarantinedNodes: 0 },
  { timestamp: '2026-08-16', label: 'Sun Aug 16', preventedSdcEvents: 19, uptimePercent: 99.97, unprotectedUptimePercent: 83.4, savedGpuHours: 2680, savedDollars: 10184, layer1Events: 7, layer2Events: 8, layer3Events: 2, layer4Events: 1, layer5Events: 1, recomputedTiles: 195, quarantinedNodes: 2 },
  { timestamp: '2026-08-17', label: 'Mon Aug 17', preventedSdcEvents: 26, uptimePercent: 99.96, unprotectedUptimePercent: 77.2, savedGpuHours: 3840, savedDollars: 14592, layer1Events: 10, layer2Events: 10, layer3Events: 4, layer4Events: 1, layer5Events: 1, recomputedTiles: 270, quarantinedNodes: 3 },
  { timestamp: '2026-08-18', label: 'Tue Aug 18 (Today)', preventedSdcEvents: 14, uptimePercent: 100.0, unprotectedUptimePercent: 88.0, savedGpuHours: 1890, savedDollars: 7182, layer1Events: 4, layer2Events: 6, layer3Events: 3, layer4Events: 1, layer5Events: 0, recomputedTiles: 145, quarantinedNodes: 1 },
];

export const DAILY_30D_DATA: HealthTrendPoint[] = Array.from({ length: 30 }, (_, i) => {
  const day = i + 1;
  const dateStr = `2026-${day <= 18 ? '08' : '07'}-${(day <= 18 ? day : day).toString().padStart(2, '0')}`;
  const label = `Day -${30 - i}`;
  const baseSdc = 12 + Math.floor(Math.sin(i * 0.8) * 8 + (i % 4));
  const l1 = Math.floor(baseSdc * 0.35);
  const l2 = Math.floor(baseSdc * 0.38);
  const l3 = Math.floor(baseSdc * 0.18);
  const l4 = Math.max(0, baseSdc - l1 - l2 - l3);
  const l5 = i % 6 === 0 ? 1 : 0;
  const savedGpu = baseSdc * 135;
  const unprotect = 82 + Math.cos(i * 0.5) * 7;

  return {
    timestamp: dateStr,
    label,
    preventedSdcEvents: baseSdc,
    uptimePercent: 99.95 + (Math.sin(i) * 0.04),
    unprotectedUptimePercent: Number(unprotect.toFixed(1)),
    savedGpuHours: savedGpu,
    savedDollars: Math.round(savedGpu * 3.8),
    layer1Events: l1,
    layer2Events: l2,
    layer3Events: l3,
    layer4Events: l4,
    layer5Events: l5,
    recomputedTiles: baseSdc * 11 + Math.floor(Math.random() * 15),
    quarantinedNodes: baseSdc > 18 ? 2 : baseSdc > 12 ? 1 : 0,
  };
});

export const QUARTERLY_90D_DATA: HealthTrendPoint[] = Array.from({ length: 12 }, (_, i) => {
  const weekNum = i + 1;
  const label = `Week ${weekNum}`;
  const baseSdc = 95 + Math.floor(Math.sin(i * 0.6) * 35);
  const savedGpu = baseSdc * 140;
  return {
    timestamp: `2026-W${weekNum.toString().padStart(2, '0')}`,
    label,
    preventedSdcEvents: baseSdc,
    uptimePercent: 99.97 + (Math.sin(i * 0.4) * 0.025),
    unprotectedUptimePercent: 81.5 + (Math.cos(i * 0.4) * 6),
    savedGpuHours: savedGpu,
    savedDollars: Math.round(savedGpu * 3.8),
    layer1Events: Math.floor(baseSdc * 0.36),
    layer2Events: Math.floor(baseSdc * 0.37),
    layer3Events: Math.floor(baseSdc * 0.17),
    layer4Events: Math.floor(baseSdc * 0.08),
    layer5Events: Math.floor(baseSdc * 0.02),
    recomputedTiles: baseSdc * 12,
    quarantinedNodes: Math.floor(baseSdc / 15),
  };
});

export interface IncidentFeedItem {
  id: string;
  time: string;
  cluster: string;
  node: string;
  rank: number;
  layer: string;
  layerNum: number;
  anomalyType: string;
  actionTaken: string;
  savingsGpuHours: number;
  status: 'MITIGATED_IN_FLIGHT' | 'RANK_QUARANTINED' | 'WEIGHT_CLEANSED' | 'SUB_NORMAL_TRAPPED';
}

export const RECENT_INCIDENT_FEED: IncidentFeedItem[] = [
  {
    id: 'INC-8492',
    time: '4 mins ago',
    cluster: 'Llama-3 405B Pretrain',
    node: 'dgx-hopper-06',
    rank: 47,
    layer: 'Layer 2: Parameter Integrity',
    layerNum: 2,
    anomalyType: 'IEEE 754 Bit-Flip on W_qkv.weight[14820] (Exponent bit flipped 0x40 -> 0x7F)',
    actionTaken: 'In-register tile rollback & bit-flip correction within 3.1ms. Loss divergence averted.',
    savingsGpuHours: 24576, // 4096 GPUs * 6h rollback
    status: 'MITIGATED_IN_FLIGHT',
  },
  {
    id: 'INC-8491',
    time: '28 mins ago',
    cluster: 'MoE Pretrain (671B)',
    node: 'dgx-hopper-12',
    rank: 92,
    layer: 'Layer 3: Activation Integrity',
    layerNum: 3,
    anomalyType: 'Heavy-tail activation kurtosis spike (Kurtosis=48.2, Robust Z=6.18)',
    actionTaken: 'Strict no-poisoning baseline isolation + dynamic gradient clipping applied.',
    savingsGpuHours: 12288,
    status: 'WEIGHT_CLEANSED',
  },
  {
    id: 'INC-8490',
    time: '1 hr 14 mins ago',
    cluster: 'Multimodal Vision-Language',
    node: 'dgx-nvl-03',
    rank: 18,
    layer: 'Layer 1: Deterministic Integrity',
    layerNum: 1,
    anomalyType: 'FP8 E4M3 subnormal saturation trap on attention score projection',
    actionTaken: 'Autonomous zero-overhead clamp to max finite FP8 representation (448.0).',
    savingsGpuHours: 6144,
    status: 'SUB_NORMAL_TRAPPED',
  },
  {
    id: 'INC-8489',
    time: '2 hrs 42 mins ago',
    cluster: 'Llama-3 405B Pretrain',
    node: 'dgx-hopper-15',
    rank: 118,
    layer: 'Layer 5: Correlation Engine',
    layerNum: 5,
    anomalyType: 'NVLink All-Reduce packet parity CRC mismatch on Port 9',
    actionTaken: 'Automated Slurm drain signal issued. Rank 118 hot-swapped to standby node in 850ms.',
    savingsGpuHours: 24576,
    status: 'RANK_QUARANTINED',
  },
];

export type ModuleTab =
  | 'overview'
  | 'simulator'
  | 'kernels'
  | 'scanner'
  | 'diagnostic'
  | 'commercial'
  | 'roi'
  | 'science';

export type GpuStatus = 'healthy' | 'corrupted' | 'quarantined' | 'recomputing';

export type FaultType =
  | 'exponent_msb'
  | 'mantissa_bit'
  | 'alu_carry_chain'
  | 'subnormal_grad'
  | 'voltage_droop'
  | 'bus_drift';

export interface GpuRank {
  id: number;
  nodeId: string;
  rank: number;
  gpuIndex: number;
  model: string;
  status: GpuStatus;
  temperature: number;
  powerWatts: number;
  smUtilization: number;
  parityDelta: number;
  lastFault?: FaultType;
  lastFaultTime?: string;
  recomputedTiles: number;
}

export interface TelemetryLog {
  id: string;
  timestamp: string;
  rank: number;
  nodeId: string;
  severity: 'info' | 'warning' | 'critical' | 'recovery';
  event: string;
  details: string;
  deltaNorm?: number;
  durationMs?: number;
}

export interface LossDataPoint {
  step: number;
  unprotectedLoss: number;
  silentGuardLoss: number;
  injected: boolean;
  recovered: boolean;
}

export interface CheckpointLayerScan {
  layerName: string;
  shape: string;
  dtype: string;
  kurtosis: number;
  anomalyScore: number;
  status: 'normal' | 'suspicious' | 'corrupted';
  exponentOverflows: number;
  subnormalCount: number;
  nanCount: number;
  hammingDrift: number;
}

export interface DiagnosticMessage {
  id: string;
  sender: 'user' | 'gemini' | 'system';
  content: string;
  timestamp: string;
  modelUsed?: string;
  isFallback?: boolean;
}

export interface DiagnosticScenario {
  id: string;
  title: string;
  node: string;
  rank: number;
  layer: string;
  description: string;
  logs: string;
}

export interface KernelConfig {
  framework: 'triton' | 'pytorch' | 'deepspeed' | 'flashattention';
  precision: 'fp8_e4m3' | 'bf16' | 'fp16' | 'fp32';
  toleranceEpsilon: number;
  samplingRatePercent: number;
  autoDrainOnFailure: boolean;
  inlineRecompute: boolean;
  telemetryEndpoint: string;
}

export interface Ieee754Format {
  name: string;
  totalBits: number;
  signBits: number;
  exponentBits: number;
  mantissaBits: number;
  bias: number;
  description: string;
}

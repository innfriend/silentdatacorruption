export type ModuleTab =
  | 'overview'
  | 'simulator'
  | 'kernels'
  | 'scanner'
  | 'diagnostic'
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

export type SeverityLevel = 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export type RecommendedAction =
  | 'NONE'
  | 'INSPECT_INPUT'
  | 'INSPECT_ACTIVATION'
  | 'INSPECT_PARAMETER'
  | 'INSPECT_PARAMETER_STATE'
  | 'INSPECT_CHECKPOINT'
  | 'RESTART_RUN'
  | 'ROLLBACK_MODEL'
  | 'CHECK_HARDWARE'
  | 'CHECK_NUMERICAL_PRECISION';

export type PerformanceMode = 'light' | 'balanced' | 'full';

export interface TrainGuardEvent {
  schema_version: string;
  event_id: string;
  timestamp: string;
  model_id: string;
  model_version: string;
  run_id: string;
  step: number;
  layer: 'deterministic_integrity' | 'parameter_integrity' | 'activation_integrity' | 'statistical_engine' | 'correlation_engine';
  layer_number: 1 | 2 | 3 | 4 | 5;
  event_type: string;
  severity: SeverityLevel;
  confidence: number;
  location: {
    module?: string;
    parameter?: string;
    index?: number;
    tensor_name?: string;
    rank?: number;
    node_id?: string;
  };
  observed: {
    value?: number;
    delta?: number;
    finite_ratio?: number;
    nan_count?: number;
    inf_count?: number;
    l2_norm?: number;
    kurtosis?: number;
  };
  baseline: {
    median: number;
    mad: number;
    mean?: number;
    std?: number;
  };
  thresholds: {
    robust_z: number;
    tolerance_eps?: number;
  };
  signals_triggered: string[];
  evidence: {
    persistence: number;
    window: number;
    consecutive_steps?: number;
    baseline_poisoned: boolean;
    explanation: string;
  };
  recommended_action: RecommendedAction;
  action_detail: string;
  detector_version: string;
}

export interface ReliabilityScoreBreakdown {
  score: number; // 0 - 100
  status: 'HEALTHY' | 'MINOR_ANOMALIES' | 'DEGRADED' | 'SERIOUS' | 'CRITICAL';
  deterministicDeduction: number;
  statisticalDeduction: number;
  persistenceDeduction: number;
  affectedComponentsDeduction: number;
  cleanStepsCount: number;
  totalMonitoredSteps: number;
  noPoisoningRate: number;
}

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
  performanceMode?: PerformanceMode;
  historySize?: number;
  robustZThreshold?: number;
  confirmationWindow?: number;
  confirmationCount?: number;
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


import { GpuRank, TelemetryLog, LossDataPoint, DiagnosticScenario, CheckpointLayerScan, Ieee754Format } from '../types';

export const INITIAL_RANKS: GpuRank[] = Array.from({ length: 128 }, (_, i) => {
  const nodeNum = Math.floor(i / 8) + 1;
  const gpuIndex = i % 8;
  const nodeId = `dgx-hopper-${nodeNum.toString().padStart(2, '0')}`;
  return {
    id: i,
    nodeId,
    rank: i,
    gpuIndex,
    model: 'NVIDIA H100 SXM5 80GB',
    status: 'healthy',
    temperature: 58 + Math.floor(Math.sin(i) * 6),
    powerWatts: 610 + Math.floor(Math.cos(i) * 45),
    smUtilization: 94 + (i % 5),
    parityDelta: 1.2e-6 + (Math.random() * 0.5e-6),
    recomputedTiles: 0,
  };
});

export const INITIAL_LOGS: TelemetryLog[] = [
  {
    id: 'log-001',
    timestamp: '08:34:42.108',
    rank: 0,
    nodeId: 'dgx-hopper-01',
    severity: 'info',
    event: 'CLUSTER_INIT_OK',
    details: 'SilentGuard in-register stochastic parity runtime initialized on 128 ranks.',
    durationMs: 1.2,
  },
  {
    id: 'log-002',
    timestamp: '08:34:43.012',
    rank: 47,
    nodeId: 'dgx-hopper-06',
    severity: 'info',
    event: 'PARITY_INVARIANT_SYNC',
    details: '||r^T * C - (r^T * A) * B||_inf = 3.41e-6 <= eps (1.0e-4). Tile nominal.',
    durationMs: 2.1,
  },
  {
    id: 'log-003',
    timestamp: '08:34:44.890',
    rank: 87,
    nodeId: 'dgx-hopper-11',
    severity: 'info',
    event: 'NVLINK_BANDWIDTH_CHECK',
    details: '900 GB/s bidirectional NVLink bandwidth verified across all 18 ports.',
    durationMs: 0.8,
  },
];

export const INITIAL_LOSS_CURVE: LossDataPoint[] = [
  { step: 42080, unprotectedLoss: 2.145, silentGuardLoss: 2.145, injected: false, recovered: false },
  { step: 42085, unprotectedLoss: 2.141, silentGuardLoss: 2.141, injected: false, recovered: false },
  { step: 42090, unprotectedLoss: 2.138, silentGuardLoss: 2.138, injected: false, recovered: false },
  { step: 42095, unprotectedLoss: 2.134, silentGuardLoss: 2.134, injected: false, recovered: false },
  { step: 42100, unprotectedLoss: 2.130, silentGuardLoss: 2.130, injected: false, recovered: false },
  { step: 42105, unprotectedLoss: 2.127, silentGuardLoss: 2.127, injected: false, recovered: false },
  { step: 42110, unprotectedLoss: 2.123, silentGuardLoss: 2.123, injected: false, recovered: false },
  { step: 42115, unprotectedLoss: 2.120, silentGuardLoss: 2.120, injected: false, recovered: false },
  { step: 42120, unprotectedLoss: 2.116, silentGuardLoss: 2.116, injected: false, recovered: false },
];

export const DIAGNOSTIC_SCENARIOS: DiagnosticScenario[] = [
  {
    id: 'scenario-1',
    title: 'Loss divergence on Step 42,100 (Attention QK^T MSB Flip)',
    node: 'dgx-hopper-06',
    rank: 47,
    layer: 'model.layers.31.self_attn.q_proj',
    description: 'During a 70B parameter pre-training run, Rank 47 experienced an MSB exponent bit-flip inside the Tensor Core MMA accumulator during the QK^T matrix multiplication.',
    logs: `[08:34:48.214] WARNING [Rank 47 / dgx-hopper-06 / GPU 7] PARITY_INVARIANT_VIOLATION
Tensor: model.layers.31.self_attn.q_proj.weight (Shape [8192, 8192])
Tolerance eps: 1.00e-04 | Observed Delta: 1.482e+03
Bit signature: 0b1100000000000000 -> Exponent Bit 14 toggled
SM Unit: SM 34 (Tensor Core Slice 2)
Action: Tile auto-recomputed in 3.18 ms on SM 35. Rank 47 marked for Slurm drain.`,
  },
  {
    id: 'scenario-2',
    title: 'Sudden gradient explosion on Rank 87 (DeepSpeed MoE Router ALU Error)',
    node: 'dgx-hopper-11',
    rank: 87,
    layer: 'model.layers.16.mlp.gate',
    description: 'A carry-chain adder fault in the router gate calculation caused an integer overflow, routing 99.8% of tokens to Expert 0 and triggering an unrecoverable gradient blowup.',
    logs: `[08:34:52.901] CRITICAL [Rank 87 / dgx-hopper-11 / GPU 7] ALU_CARRY_CHAIN_FAULT
Subsystem: DeepSpeed MoE Gate Dispatcher (Top-2 Router)
Expected Token Distribution Entropy: 4.82 bits | Observed: 0.04 bits
Router Softmax Exponent Overflow: logits[0] = +inf (NaN generated)
SilentGuard Invariant Guard: Trapped NaN before all_to_all collective.
Action: Token routing fell back to CPU shadow dispatch; Rank 87 isolated.`,
  },
  {
    id: 'scenario-3',
    title: 'Thermal throttling & dynamic voltage droop on Node dgx-06 (SM 34 L*di/dt)',
    node: 'dgx-hopper-06',
    rank: 42,
    layer: 'model.layers.40.mlp.up_proj',
    description: 'Rapid transition between memory-bound attention and compute-dense GEMM caused a transient 180mV voltage drop, triggering multiple single-event upsets across 4 SMs.',
    logs: `[08:35:01.332] WARNING [Rank 42 / dgx-hopper-06 / GPU 2] TRANSIENT_VOLTAGE_DROOP
Vdd Core: Dropped from 0.85V to 0.67V in 14.2 nanoseconds (L*di/dt spike)
Power Step: 380W -> 695W transition during FlashAttention-3 backward pass
Parity Check: 3 of 128 GEMM tiles failed residual invariant.
Remediation: In-register tiles recomputed with clock frequency locked at 1.75 GHz.`,
  },
  {
    id: 'scenario-4',
    title: 'Silent Mantissa Bit-Drift in AdamW First Moment Buffer',
    node: 'dgx-hopper-03',
    rank: 19,
    layer: 'optimizer.state.exp_avg.layer24',
    description: 'A low-order mantissa bit stuck at 1 in SRAM caused exponential variance decay over 5,000 steps without throwing NaN or raising hardware ECC traps.',
    logs: `[08:35:10.744] WARNING [Rank 19 / dgx-hopper-03 / GPU 3] MANTISSA_DRIFT_ACCUMULATOR
Optimizer State: AdamW exp_avg (beta1=0.9, beta2=0.95)
Kurtosis Anomaly: 8.92 (Threshold: 4.5) | Hamming distance drift: +14 bits/hr
Variance Stagnation: Grad norm stalled at 0.0012 while loss stopped descending.
Detection: SilentGuard offline scanner flagged Kurtosis anomaly in checkpoint state.`,
  },
];

export const SAMPLE_CHECKPOINT_LAYERS: CheckpointLayerScan[] = [
  {
    layerName: 'model.layers.0.self_attn.q_proj.weight',
    shape: '[8192, 8192]',
    dtype: 'bfloat16',
    kurtosis: 3.12,
    anomalyScore: 0.05,
    status: 'normal',
    exponentOverflows: 0,
    subnormalCount: 0,
    nanCount: 0,
    hammingDrift: 0.0,
  },
  {
    layerName: 'model.layers.16.mlp.gate.weight (MoE Router)',
    shape: '[64, 8192]',
    dtype: 'float32',
    kurtosis: 6.48,
    anomalyScore: 0.88,
    status: 'suspicious',
    exponentOverflows: 4,
    subnormalCount: 12,
    nanCount: 0,
    hammingDrift: 18.2,
  },
  {
    layerName: 'model.layers.31.self_attn.q_proj.weight',
    shape: '[8192, 8192]',
    dtype: 'bfloat16',
    kurtosis: 14.82,
    anomalyScore: 0.99,
    status: 'corrupted',
    exponentOverflows: 29,
    subnormalCount: 140,
    nanCount: 3,
    hammingDrift: 44.7,
  },
  {
    layerName: 'model.layers.31.self_attn.k_proj.weight',
    shape: '[8192, 1024]',
    dtype: 'bfloat16',
    kurtosis: 3.08,
    anomalyScore: 0.04,
    status: 'normal',
    exponentOverflows: 0,
    subnormalCount: 0,
    nanCount: 0,
    hammingDrift: 0.0,
  },
  {
    layerName: 'model.layers.40.mlp.up_proj.weight',
    shape: '[28672, 8192]',
    dtype: 'bfloat16',
    kurtosis: 3.25,
    anomalyScore: 0.08,
    status: 'normal',
    exponentOverflows: 0,
    subnormalCount: 2,
    nanCount: 0,
    hammingDrift: 0.1,
  },
  {
    layerName: 'model.norm.weight',
    shape: '[8192]',
    dtype: 'float32',
    kurtosis: 2.94,
    anomalyScore: 0.02,
    status: 'normal',
    exponentOverflows: 0,
    subnormalCount: 0,
    nanCount: 0,
    hammingDrift: 0.0,
  },
];

export const IEEE754_FORMATS: Ieee754Format[] = [
  {
    name: 'BF16 (Bfloat16)',
    totalBits: 16,
    signBits: 1,
    exponentBits: 8,
    mantissaBits: 7,
    bias: 127,
    description: 'Same dynamic range as FP32 (~1e-38 to ~3e38). A single bit flip in the exponent MSB multiplies the scalar value by 2^64 (approx 1.84e19), instantly shattering the Adam optimizer.',
  },
  {
    name: 'FP16 (IEEE Half Precision)',
    totalBits: 16,
    signBits: 1,
    exponentBits: 5,
    mantissaBits: 10,
    bias: 15,
    description: 'Narrow dynamic range (~6e-5 to 65504). Exponent flips easily cause immediate overflow to +inf / NaN. Highly susceptible to subnormal underflow corruptions.',
  },
  {
    name: 'FP8 (E4M3 - Hopper / Blackwell)',
    totalBits: 8,
    signBits: 1,
    exponentBits: 4,
    mantissaBits: 3,
    bias: 7,
    description: 'Designed for forward-pass GEMM in H100 and B200. Only 3 mantissa bits means every bit flip causes massive relative error (up to 25-50% divergence per single bit).',
  },
  {
    name: 'FP32 (IEEE Single Precision)',
    totalBits: 32,
    signBits: 1,
    exponentBits: 8,
    mantissaBits: 23,
    bias: 127,
    description: 'Standard master weights and optimizer state format. Mantissa bit flips in lower 12 bits can silently corrupt model reasoning for thousands of steps before detection.',
  },
];

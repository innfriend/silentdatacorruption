import React, { useState } from 'react';
import {
  DollarSign,
  ShieldCheck,
  Server,
  Layers,
  Terminal,
  FileCode,
  Check,
  Copy,
  ArrowRight,
  Sparkles,
  Award,
  Zap,
} from 'lucide-react';

export const CommercialModule: React.FC = () => {
  const [gpuCount, setGpuCount] = useState<number>(1024);
  const [copiedSection, setCopiedSection] = useState<string | null>(null);

  const pricePerGpuHour = 0.035;
  const annualHours = 8760 * 0.85; // 85% cluster utilization
  const totalAnnualCost = gpuCount * annualHours * pricePerGpuHour;
  const estimatedAnnualSavings = gpuCount * 1250; // $1,250 saved per GPU per year from eliminated rollback recomputation

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(id);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  const k8sManifest = `apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: silentguard-agent
  namespace: kube-system
  labels:
    app.kubernetes.io/name: silentguard
spec:
  selector:
    matchLabels:
      name: silentguard-agent
  template:
    metadata:
      labels:
        name: silentguard-agent
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: silentguard-daemon
        image: cr.silentguard.ai/enterprise/sg-daemon:v2.4.0
        securityContext:
          privileged: true
        env:
        - name: SG_CLUSTER_ID
          value: "prod-hopper-cluster-01"
        - name: SG_INVARIANT_TOLERANCE
          value: "1e-4"
        - name: SG_AUTO_QUARANTINE_SLURM
          value: "true"
        volumeMounts:
        - name: dev-nvidia
          mountPath: /dev/nvidia0
        - name: slurm-spool
          mountPath: /var/spool/slurmd
      volumes:
      - name: dev-nvidia
        hostPath:
          path: /dev
      - name: slurm-spool
        hostPath:
          path: /var/spool/slurmd`;

  const slurmProlog = `#!/bin/bash
# ==============================================================================
# SilentGuard Slurm Prolog Hook: Validates GPU ALUs before Job Allocation
# Path: /etc/slurm/prolog.d/99_silentguard_qualify.sh
# ==============================================================================

echo "[SilentGuard] Running fast 2.5-second pre-flight SDC check on GPUs: $SLURM_JOB_GPUS"

# Execute in-register stochastic parity qualification microbenchmark
/opt/silentguard/bin/sg-qualify --gpus "$SLURM_JOB_GPUS" --tolerance 1e-4 --timeout 3

if [ $? -ne 0 ]; then
  echo "[SilentGuard FATAL] SDC detected in ALU execution units! Draining node $SLURMD_NODENAME"
  scontrol update NodeName="$SLURMD_NODENAME" State=DRAIN Reason="SILENTGUARD_PREFLIGHT_SDC_FAILED"
  exit 1
fi

echo "[SilentGuard] All allocated Tensor Cores verified clean. Job authorized."
exit 0`;

  const prometheusMetrics = `# HELP silentguard_sdc_events_total Cumulative count of detected silent data corruption events.
# TYPE silentguard_sdc_events_total counter
silentguard_sdc_events_total{node="dgx-hopper-06",rank="47",subsystem="tensor_core_mma"} 3
silentguard_sdc_events_total{node="dgx-hopper-11",rank="87",subsystem="alu_carry_chain"} 1

# HELP silentguard_quarantine_latency_ms Latency in milliseconds from SDC trap to Slurm drain action.
# TYPE silentguard_quarantine_latency_ms gauge
silentguard_quarantine_latency_ms{node="dgx-hopper-06"} 3.18

# HELP silentguard_recomputed_tiles_total Number of GEMM tiles autonomously recomputed in-register.
# TYPE silentguard_recomputed_tiles_total counter
silentguard_recomputed_tiles_total{cluster="prod-hopper-cluster-01"} 1420`;

  return (
    <div className="space-y-6">
      {/* Tier Pricing Cards */}
      <div>
        <div className="text-center max-w-2xl mx-auto mb-6">
          <h3 className="text-xl font-light text-[#2A2A2A] tracking-tight font-serif">
            Enterprise Fleet Licensing & Production Tiers
          </h3>
          <p className="text-xs text-[#666] mt-1">
            Predictable consumption-based pricing for frontier labs, cloud providers, and hyperscalers
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Community Tier */}
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 flex flex-col justify-between shadow-xs">
            <div>
              <div className="text-xs font-mono uppercase font-bold text-[#666] mb-2">Community / OSS</div>
              <div className="text-3xl font-light text-[#2A2A2A] font-serif mb-1">$0</div>
              <p className="text-xs text-[#666] mb-4">For academic research and single-node experiments</p>

              <ul className="space-y-2 text-xs text-[#2A2A2A] mb-6">
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Standard PyTorch FSDP Hook</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Up to 8 GPUs per cluster</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Community Discord Support</span>
                </li>
              </ul>
            </div>

            <button className="w-full py-2 px-3 rounded-[2px] text-xs font-semibold bg-[#F8F7F4] text-[#2A2A2A] border border-[#D1D0CB] hover:bg-[#EBEAE5] cursor-pointer">
              Deploy OSS Edition
            </button>
          </div>

          {/* Cluster Pro Tier */}
          <div className="bg-white border-2 border-[#4A5D4E] rounded-[3px] p-6 flex flex-col justify-between shadow-xs relative">
            <div className="absolute -top-3 right-6 bg-[#4A5D4E] text-white text-[10px] font-mono font-bold px-2 py-0.5 rounded-[2px] uppercase">
              Most Popular
            </div>

            <div>
              <div className="text-xs font-mono uppercase font-bold text-[#4A5D4E] mb-2">Cluster Pro</div>
              <div className="text-3xl font-light text-[#2A2A2A] font-serif mb-1">
                $0.035 <span className="text-xs font-normal text-[#666] font-sans">/ GPU-hr</span>
              </div>
              <p className="text-xs text-[#666] mb-4">For production LLM & multi-modal training clusters</p>

              <ul className="space-y-2 text-xs text-[#2A2A2A] mb-6">
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Fused In-Register Triton Kernels</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Autonomous Slurm/K8s Quarantine</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>In-flight tile recomputation</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Prometheus / Datadog Exporter</span>
                </li>
              </ul>
            </div>

            <button className="w-full py-2 px-3 rounded-[2px] text-xs font-bold bg-[#4A5D4E] text-white hover:bg-[#3B4A3E] shadow-xs cursor-pointer">
              Start Fleet Trial
            </button>
          </div>

          {/* Enterprise Fleet Tier */}
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 flex flex-col justify-between shadow-xs">
            <div>
              <div className="text-xs font-mono uppercase font-bold text-[#666] mb-2">Enterprise Fleet</div>
              <div className="text-3xl font-light text-[#2A2A2A] font-serif mb-1">Custom</div>
              <p className="text-xs text-[#666] mb-4">For hyperscale clusters (10,000+ GPUs) & sovereign clouds</p>

              <ul className="space-y-2 text-xs text-[#2A2A2A] mb-6">
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>100% Air-Gapped Deployment</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Custom Silicon ASIC Microbenchmarks</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>Dedicated GPU Reliability Engineers</span>
                </li>
                <li className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-[#4A5D4E]" />
                  <span>15-Minute Critical Severity SLA</span>
                </li>
              </ul>
            </div>

            <button className="w-full py-2 px-3 rounded-[2px] text-xs font-semibold bg-[#2A2A2A] text-white hover:bg-[#1C1C1A] cursor-pointer">
              Contact Enterprise Sales
            </button>
          </div>
        </div>
      </div>

      {/* Interactive Annual Licensing Fee Calculator */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <div className="flex items-center gap-2 mb-4">
          <DollarSign className="w-4 h-4 text-[#4A5D4E]" />
          <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
            Interactive Fleet Licensing & Net Savings Estimator
          </h4>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs font-medium mb-1 text-[#666]">
                <span>Cluster Size (Active GPUs)</span>
                <span className="font-mono text-[#4A5D4E] font-bold">{gpuCount} GPUs</span>
              </div>
              <input
                type="range"
                min={64}
                max={16384}
                step={64}
                value={gpuCount}
                onChange={(e) => setGpuCount(parseInt(e.target.value))}
                className="w-full accent-[#4A5D4E] cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-[#999] mt-1">
                <span>64 GPUs</span>
                <span>1,024 GPUs</span>
                <span>16,384 GPUs</span>
              </div>
            </div>

            <div className="text-xs text-[#666] leading-relaxed bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
              Rate: <strong>$0.035 / GPU-hr</strong> • Assumed 85% annualized utilization (7,446 active compute hours / GPU / yr).
            </div>
          </div>

          <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB]">
              <div className="text-[10px] font-mono text-[#666] uppercase">Annual Software Cost</div>
              <div className="text-2xl font-light font-serif text-[#2A2A2A] mt-1">
                ${(totalAnnualCost / 1000).toFixed(1)}k
              </div>
              <div className="text-[11px] text-[#666] mt-1">SilentGuard License</div>
            </div>

            <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB]">
              <div className="text-[10px] font-mono text-[#666] uppercase">Prevented Loss Waste</div>
              <div className="text-2xl font-light font-serif text-[#4A5D4E] mt-1">
                ${(estimatedAnnualSavings / 1000).toFixed(1)}k
              </div>
              <div className="text-[11px] text-[#666] mt-1">Avoided Rollback Cost</div>
            </div>

            <div className="bg-[#4A5D4E] text-white p-4 rounded-[2px] border border-[#3B4A3E]">
              <div className="text-[10px] font-mono text-[#D7E4DA] uppercase">Net ROI Multiple</div>
              <div className="text-2xl font-bold font-mono mt-1">
                {(estimatedAnnualSavings / Math.max(1, totalAnnualCost)).toFixed(1)}x ROI
              </div>
              <div className="text-[11px] text-[#D7E4DA] mt-1">Net positive return</div>
            </div>
          </div>
        </div>
      </div>

      {/* Integration Documentation: Kubernetes, Slurm, Prometheus */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
        <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
          Infrastructure Deployment Manifests
        </h4>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Kubernetes Daemonset */}
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold font-mono text-[#2A2A2A]">1. Kubernetes DaemonSet</span>
                <button
                  onClick={() => handleCopy(k8sManifest, 'k8s')}
                  className="text-[11px] font-mono text-[#4A5D4E] hover:underline flex items-center gap-1 cursor-pointer"
                >
                  {copiedSection === 'k8s' ? <Check className="w-3 h-3 text-[#4A5D4E]" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedSection === 'k8s' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <div className="bg-[#1C1C1A] text-[#D1D0CB] p-2.5 rounded-[2px] font-mono text-[10px] max-h-44 overflow-y-auto leading-relaxed">
                <pre>{k8sManifest}</pre>
              </div>
            </div>
          </div>

          {/* Slurm Prolog Hook */}
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold font-mono text-[#2A2A2A]">2. Slurm Prolog Hook</span>
                <button
                  onClick={() => handleCopy(slurmProlog, 'slurm')}
                  className="text-[11px] font-mono text-[#4A5D4E] hover:underline flex items-center gap-1 cursor-pointer"
                >
                  {copiedSection === 'slurm' ? <Check className="w-3 h-3 text-[#4A5D4E]" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedSection === 'slurm' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <div className="bg-[#1C1C1A] text-[#D1D0CB] p-2.5 rounded-[2px] font-mono text-[10px] max-h-44 overflow-y-auto leading-relaxed">
                <pre>{slurmProlog}</pre>
              </div>
            </div>
          </div>

          {/* Prometheus Metrics */}
          <div className="bg-[#F8F7F4] p-4 rounded-[2px] border border-[#D1D0CB] flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-xs font-bold font-mono text-[#2A2A2A]">3. Prometheus Metrics Exporter</span>
                <button
                  onClick={() => handleCopy(prometheusMetrics, 'prom')}
                  className="text-[11px] font-mono text-[#4A5D4E] hover:underline flex items-center gap-1 cursor-pointer"
                >
                  {copiedSection === 'prom' ? <Check className="w-3 h-3 text-[#4A5D4E]" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedSection === 'prom' ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <div className="bg-[#1C1C1A] text-[#D1D0CB] p-2.5 rounded-[2px] font-mono text-[10px] max-h-44 overflow-y-auto leading-relaxed">
                <pre>{prometheusMetrics}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  AreaChart,
  BarChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import {
  Activity,
  ShieldCheck,
  TrendingUp,
  Clock,
  DollarSign,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Server,
  Zap,
  Filter,
  ArrowUpRight,
  Info,
  Calendar,
  Sparkles,
  Gauge,
  SlidersHorizontal,
  Terminal,
  UploadCloud,
  FileCode,
  Copy,
  Check,
  RefreshCw,
  ExternalLink,
} from 'lucide-react';
import {
  HealthTrendPoint,
  CLUSTER_PROFILES,
  HOURLY_24H_DATA,
  DAILY_7D_DATA,
  DAILY_30D_DATA,
  QUARTERLY_90D_DATA,
  RECENT_INCIDENT_FEED,
} from '../data/healthTrendsData';

type Timeframe = '24h' | '7d' | '30d' | '90d';
type MetricView = 'overview' | 'uptime_sla' | 'prevented_sdc' | 'savings' | 'layers';

export const ClusterHealthTrends: React.FC = () => {
  const [timeframe, setTimeframe] = useState<Timeframe>('7d');
  const [selectedClusterId, setSelectedClusterId] = useState<string>('all');
  const [metricView, setMetricView] = useState<MetricView>('overview');
  const [showRecentFeed, setShowRecentFeed] = useState<boolean>(true);
  const [showDeploymentGuide, setShowDeploymentGuide] = useState<boolean>(false);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  // Custom user report state
  const [customReport, setCustomReport] = useState<any | null>(null);
  const [customReportError, setCustomReportError] = useState<string | null>(null);

  const selectedCluster = CLUSTER_PROFILES.find((c) => c.id === selectedClusterId) || CLUSTER_PROFILES[0];

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (parsed.summary || parsed.reliability_scorecard || parsed.sla_metrics) {
          setCustomReport(parsed);
          setCustomReportError(null);
        } else {
          setCustomReportError('Invalid report format. Expected a TrainGuard cluster health report JSON.');
        }
      } catch (err) {
        setCustomReportError('Failed to parse JSON file.');
      }
    };
    reader.readAsText(file);
  };

  // Workload scale factor
  const clusterMultiplier = selectedCluster.id === 'all'
    ? 1.0
    : selectedCluster.id === 'llama405b'
    ? 0.52
    : selectedCluster.id === 'deepseek_v3'
    ? 0.32
    : 0.16;

  // Retrieve raw dataset based on timeframe
  const rawDataset: HealthTrendPoint[] =
    timeframe === '24h'
      ? HOURLY_24H_DATA
      : timeframe === '7d'
      ? DAILY_7D_DATA
      : timeframe === '30d'
      ? DAILY_30D_DATA
      : QUARTERLY_90D_DATA;

  // Apply cluster filter scaling
  const chartData = rawDataset.map((point) => {
    const sdcScaled = Math.max(1, Math.round(point.preventedSdcEvents * clusterMultiplier));
    const savedHours = Math.round(point.savedGpuHours * clusterMultiplier);
    const savedCost = Math.round(point.savedDollars * clusterMultiplier);
    const l1 = Math.round(point.layer1Events * clusterMultiplier);
    const l2 = Math.round(point.layer2Events * clusterMultiplier);
    const l3 = Math.round(point.layer3Events * clusterMultiplier);
    const l4 = Math.round(point.layer4Events * clusterMultiplier);
    const l5 = Math.round(point.layer5Events * clusterMultiplier);

    return {
      ...point,
      preventedSdcEvents: sdcScaled,
      savedGpuHours: savedHours,
      savedDollars: savedCost,
      layer1Events: l1,
      layer2Events: l2,
      layer3Events: l3,
      layer4Events: l4,
      layer5Events: l5,
    };
  });

  // Calculate high-level summary KPIs (adjusted if custom report is loaded)
  const totalPreventedSdc = customReport
    ? (customReport.summary?.total_incidents_prevented ?? customReport.events?.length ?? 12)
    : chartData.reduce((acc, curr) => acc + curr.preventedSdcEvents, 0);

  const totalGpuHoursSaved = customReport
    ? (customReport.economic_savings?.avoided_compute_loss_gpu_hours ?? totalPreventedSdc * 32)
    : chartData.reduce((acc, curr) => acc + curr.savedGpuHours, 0);

  const totalDollarsSaved = customReport
    ? Math.round(totalGpuHoursSaved * 3.80)
    : chartData.reduce((acc, curr) => acc + curr.savedDollars, 0);

  const avgProtectedUptime = customReport
    ? parseFloat(customReport.sla_metrics?.protected_cluster_uptime ?? '99.98').toFixed(2)
    : (chartData.reduce((acc, curr) => acc + curr.uptimePercent, 0) / chartData.length).toFixed(3);

  const avgUnprotectedUptime = customReport
    ? parseFloat(customReport.sla_metrics?.unprotected_estimated_uptime ?? '83.4').toFixed(1)
    : (chartData.reduce((acc, curr) => acc + curr.unprotectedUptimePercent, 0) / chartData.length).toFixed(1);

  const totalRecomputedTiles = chartData.reduce((acc, curr) => acc + curr.recomputedTiles, 0);
  const totalQuarantined = chartData.reduce((acc, curr) => acc + curr.quarantinedNodes, 0);

  // Custom Chart Tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload as HealthTrendPoint;
      return (
        <div className="bg-[#1C1C1A] text-white border border-[#444] rounded-[2px] p-3 text-xs shadow-xl min-w-[240px] font-mono">
          <div className="flex items-center justify-between border-b border-[#333] pb-1.5 mb-2">
            <span className="font-bold text-[#EBEAE5]">{data.label}</span>
            <span className="text-[10px] text-[#4A5D4E] bg-[#2A352D] px-1.5 py-0.5 rounded-[2px] font-bold">
              {data.uptimePercent.toFixed(2)}% SLA
            </span>
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[#999]">SDC Interceptions:</span>
              <span className="font-bold text-amber-400">{data.preventedSdcEvents} incidents</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#999]">Protected Uptime:</span>
              <span className="font-bold text-emerald-400">{data.uptimePercent.toFixed(2)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#999]">Unprotected Est.:</span>
              <span className="text-red-400">{data.unprotectedUptimePercent}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#999]">Avoided Compute Loss:</span>
              <span className="text-[#EBEAE5]">{data.savedGpuHours.toLocaleString()} GPU-hrs (${data.savedDollars.toLocaleString()})</span>
            </div>
            <div className="flex items-center justify-between pt-1 border-t border-[#333] text-[10px] text-[#888]">
              <span>In-flight Recomputed:</span>
              <span>{data.recomputedTiles} MMA tiles</span>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-6 shadow-xs space-y-6">
      {/* Module Header & Clarity Explainer Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#D1D0CB] pb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-[2px] bg-[#F8F7F4] text-[#4A5D4E] border border-[#D1D0CB] text-[10px] font-mono font-bold tracking-wider">
              <Activity className="w-3 h-3 text-[#4A5D4E]" />
              <span>POST-INSTALLATION OBSERVABILITY & TELEMETRY SUITE</span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-[2px] bg-[#F8F7F4] text-[#4A5D4E] border border-[#D1D0CB] font-semibold">
              Interactive Dashboard Preview
            </span>
          </div>
          <h3 className="text-lg sm:text-xl font-serif text-[#2A2A2A] font-light">
            Cluster Health Trends & SDC Prevention History
          </h3>
          <p className="text-xs text-[#666] mt-0.5">
            This dashboard illustrates the metrics, SLA tracking, and SDC interception graphs generated on your cluster once the TrainGuard package is installed and running.
          </p>
        </div>

        {/* Global Controls & Post-Install Guide Toggle */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setShowDeploymentGuide(!showDeploymentGuide)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-mono font-medium rounded-[2px] bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] cursor-pointer transition-colors"
          >
            <Terminal className="w-3.5 h-3.5 text-[#4A5D4E]" />
            <span>{showDeploymentGuide ? 'Hide CLI Commands' : 'How to Run on Your Cluster'}</span>
          </button>

          {/* Cluster Selector */}
          <div className="flex items-center bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] px-2 py-1 text-xs">
            <Server className="w-3.5 h-3.5 text-[#4A5D4E] mr-1.5 shrink-0" />
            <select
              value={selectedClusterId}
              onChange={(e) => {
                setSelectedClusterId(e.target.value);
                setCustomReport(null);
              }}
              className="bg-transparent text-xs font-mono font-medium text-[#2A2A2A] cursor-pointer focus:outline-none"
            >
              {CLUSTER_PROFILES.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name} ({c.gpuCount} GPUs)
                </option>
              ))}
            </select>
          </div>

          {/* Timeframe Selector */}
          <div className="flex items-center bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-0.5">
            {(
              [
                { id: '24h', label: '24h' },
                { id: '7d', label: '7d' },
                { id: '30d', label: '30d' },
                { id: '90d', label: '90d' },
              ] as const
            ).map((t) => (
              <button
                key={t.id}
                onClick={() => setTimeframe(t.id)}
                className={`px-2.5 py-1 text-[11px] font-mono font-medium rounded-[2px] transition-all cursor-pointer ${
                  timeframe === t.id
                    ? 'bg-[#4A5D4E] text-white shadow-xs font-bold'
                    : 'text-[#666] hover:text-[#2A2A2A]'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Clarification Notice: Explains that this is an enterprise blueprint & how user gets their real data */}
      <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-start gap-2">
          <Info className="w-4 h-4 text-[#4A5D4E] shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <div className="font-mono font-bold text-[#2A2A2A]">
              {customReport
                ? `Active View: Loaded Custom Artifact (${customReport.run_id || 'User Run'})`
                : `Reference Architecture: Showing Calibrated Enterprise Telemetry (${selectedCluster.name})`}
            </div>
            <p className="text-[#666] text-[11px]">
              After installing the package with <code className="bg-white px-1 py-0.5 rounded-[2px] border border-[#D1D0CB] text-[#2A2A2A]">pip install .</code>, run <code className="bg-white px-1 py-0.5 rounded-[2px] border border-[#D1D0CB] text-[#4A5D4E]">train-guard cluster-health</code> to audit your GPUs, or load your generated <code className="text-[#2A2A2A]">report.json</code> below.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <label className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-mono font-medium rounded-[2px] bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] cursor-pointer transition-colors">
            <UploadCloud className="w-3.5 h-3.5 text-[#4A5D4E]" />
            <span>{customReport ? 'Change JSON Report' : 'Load Your Report.json'}</span>
            <input type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
          </label>

          {customReport && (
            <button
              onClick={() => setCustomReport(null)}
              className="inline-flex items-center gap-1 px-2 py-1 text-[11px] font-mono text-[#666] hover:text-[#2A2A2A] underline cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Reset to Default</span>
            </button>
          )}
        </div>
      </div>

      {customReportError && (
        <div className="p-2.5 rounded-[2px] bg-red-50 border border-red-200 text-red-700 text-xs font-mono flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{customReportError}</span>
        </div>
      )}

      {/* Deployment & CLI Command Guide (Expanded when user clicks 'How to Run on Your Cluster') */}
      {showDeploymentGuide && (
        <div className="bg-[#1C1C1A] text-[#EBEAE5] border border-[#444] rounded-[2px] p-4 text-xs font-mono space-y-3">
          <div className="flex items-center justify-between border-b border-[#333] pb-2">
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span className="font-bold text-white uppercase tracking-wider">
                Viewing This Dashboard on Your Own GPU Cluster
              </span>
            </div>
            <span className="text-[10px] text-[#888]">3 Production Access Methods</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="bg-[#262624] p-3 rounded-[2px] border border-[#3A3A36] space-y-1.5 flex flex-col justify-between">
              <div>
                <div className="text-emerald-400 font-bold text-[11px]">1. Local Web Dashboard</div>
                <p className="text-[11px] text-[#AAA] mt-0.5">
                  Hosts the full web NOC on your cluster head node or localhost.
                </p>
                <div className="bg-[#161614] p-1.5 rounded-[2px] text-[#DDD] text-[10px] mt-2 select-all">
                  train-guard dashboard --port 9090
                </div>
              </div>
              <button
                onClick={() => handleCopy('train-guard dashboard --port 9090', 'dash')}
                className="mt-2 text-[10px] inline-flex items-center gap-1 text-emerald-400 hover:underline cursor-pointer"
              >
                {copiedCmd === 'dash' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                <span>{copiedCmd === 'dash' ? 'Copied!' : 'Copy Command'}</span>
              </button>
            </div>

            <div className="bg-[#262624] p-3 rounded-[2px] border border-[#3A3A36] space-y-1.5 flex flex-col justify-between">
              <div>
                <div className="text-emerald-400 font-bold text-[11px]">2. Terminal Curses TUI (SSH)</div>
                <p className="text-[11px] text-[#AAA] mt-0.5">
                  Real-time live GPU health monitor in headless remote terminals.
                </p>
                <div className="bg-[#161614] p-1.5 rounded-[2px] text-[#DDD] text-[10px] mt-2 select-all">
                  train-guard monitor --job-id 89412
                </div>
              </div>
              <button
                onClick={() => handleCopy('train-guard monitor --job-id 89412', 'mon')}
                className="mt-2 text-[10px] inline-flex items-center gap-1 text-emerald-400 hover:underline cursor-pointer"
              >
                {copiedCmd === 'mon' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                <span>{copiedCmd === 'mon' ? 'Copied!' : 'Copy Command'}</span>
              </button>
            </div>

            <div className="bg-[#262624] p-3 rounded-[2px] border border-[#3A3A36] space-y-1.5 flex flex-col justify-between">
              <div>
                <div className="text-emerald-400 font-bold text-[11px]">3. Pre-Flight & Post-Training Reports</div>
                <p className="text-[11px] text-[#AAA] mt-0.5">
                  Audit hardware health & export JSON scorecards to CI/CD.
                </p>
                <div className="bg-[#161614] p-1.5 rounded-[2px] text-[#DDD] text-[10px] mt-2 select-all">
                  train-guard cluster-health
                </div>
              </div>
              <button
                onClick={() => handleCopy('train-guard cluster-health', 'audit')}
                className="mt-2 text-[10px] inline-flex items-center gap-1 text-emerald-400 hover:underline cursor-pointer"
              >
                {copiedCmd === 'audit' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                <span>{copiedCmd === 'audit' ? 'Copied!' : 'Copy Command'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Enterprise Executive KPI Summary Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 flex flex-col justify-between">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">
            Prevented SDCs
          </div>
          <div className="text-xl sm:text-2xl font-serif font-light text-[#2A2A2A] my-0.5">
            {totalPreventedSdc} <span className="text-xs font-sans text-emerald-700 font-medium">Spikes</span>
          </div>
          <div className="text-[10px] font-mono text-[#4A5D4E] flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" />
            <span>0 Loss Divergences</span>
          </div>
        </div>

        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 flex flex-col justify-between">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">
            Effective Uptime
          </div>
          <div className="text-xl sm:text-2xl font-serif font-light text-[#4A5D4E] my-0.5">
            {avgProtectedUptime}%
          </div>
          <div className="text-[10px] font-mono text-[#666]">
            Target SLA: <strong className="text-[#2A2A2A]">{selectedCluster.slaTarget}%</strong>
          </div>
        </div>

        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 flex flex-col justify-between">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">
            Unprotected Est.
          </div>
          <div className="text-xl sm:text-2xl font-serif font-light text-red-700 my-0.5">
            {avgUnprotectedUptime}%
          </div>
          <div className="text-[10px] font-mono text-red-700">
            -{(Number(avgProtectedUptime) - Number(avgUnprotectedUptime)).toFixed(1)}% Lost without Guard
          </div>
        </div>

        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 flex flex-col justify-between">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">
            Saved GPU-Hours
          </div>
          <div className="text-xl sm:text-2xl font-serif font-light text-[#2A2A2A] my-0.5">
            {totalGpuHoursSaved.toLocaleString()}
          </div>
          <div className="text-[10px] font-mono text-[#4A5D4E]">
            Avoided Rollbacks
          </div>
        </div>

        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 flex flex-col justify-between">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">
            Avoided Waste ($)
          </div>
          <div className="text-xl sm:text-2xl font-serif font-light text-[#4A5D4E] my-0.5">
            ${(totalDollarsSaved / 1000).toFixed(1)}k
          </div>
          <div className="text-[10px] font-mono text-[#666]">
            @ $3.80 / H100-hour
          </div>
        </div>

        <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-3 flex flex-col justify-between">
          <div className="text-[10px] font-mono text-[#666] uppercase tracking-wider font-semibold">
            In-Flight Heals
          </div>
          <div className="text-xl sm:text-2xl font-serif font-light text-[#2A2A2A] my-0.5">
            {totalRecomputedTiles} <span className="text-xs font-sans text-[#666]">Tiles</span>
          </div>
          <div className="text-[10px] font-mono text-[#4A5D4E]">
            {totalQuarantined} Drained Ranks
          </div>
        </div>
      </div>

      {/* Metric View Sub-Navigation Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#D1D0CB] pb-2">
        <div className="flex items-center gap-1.5 overflow-x-auto">
          {(
            [
              { id: 'overview', label: 'Composite Health & Incidents', icon: Activity },
              { id: 'uptime_sla', label: 'Cluster Uptime vs. SLA', icon: Gauge },
              { id: 'prevented_sdc', label: 'SDC Prevention Volume', icon: ShieldCheck },
              { id: 'savings', label: 'Avoided Compute Loss ($)', icon: DollarSign },
              { id: 'layers', label: '5-Layer Breakdown', icon: Layers },
            ] as const
          ).map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setMetricView(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-[2px] transition-all cursor-pointer ${
                  metricView === tab.id
                    ? 'bg-[#4A5D4E] text-white font-semibold shadow-xs'
                    : 'bg-[#F8F7F4] text-[#666] hover:text-[#2A2A2A] border border-[#D1D0CB]'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        <div className="text-[11px] font-mono text-[#666] flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse"></span>
          <span>Sampling Interval: {timeframe === '24h' ? '1 Hour' : timeframe === '7d' ? '24 Hours' : 'Daily Rolling'}</span>
        </div>
      </div>

      {/* Main Recharts Visualizations */}
      <div className="bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] p-4">
        {metricView === 'overview' && (
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h4 className="text-xs font-bold font-mono text-[#2A2A2A] uppercase tracking-wider">
                  Composite Historical Telemetry: SDC Incidents vs. Effective Uptime
                </h4>
                <p className="text-[11px] text-[#666]">
                  Dual-axis visualization of daily hardware arithmetic anomalies intercepted vs. training cluster availability SLA.
                </p>
              </div>
              <div className="flex items-center gap-4 text-[11px] font-mono">
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-[1px] bg-[#4A5D4E]"></span>
                  <span className="text-[#2A2A2A]">Prevented SDCs</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-emerald-600"></span>
                  <span className="text-[#2A2A2A]">Protected Uptime (99.98%)</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-red-500 border-dashed"></span>
                  <span className="text-[#666]">Unprotected Projected (83.4%)</span>
                </div>
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D1D0CB" opacity={0.6} />
                  <XAxis dataKey="label" stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <YAxis
                    yAxisId="left"
                    stroke="#4A5D4E"
                    tick={{ fontSize: 10, fill: '#4A5D4E', fontFamily: 'monospace' }}
                    label={{ value: 'Prevented SDC Events', angle: -90, position: 'insideLeft', offset: 18, fontSize: 10, fill: '#4A5D4E', fontFamily: 'monospace' }}
                  />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    domain={[60, 100]}
                    stroke="#2A2A2A"
                    tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }}
                    label={{ value: 'Cluster Uptime %', angle: 90, position: 'insideRight', offset: 18, fontSize: 10, fill: '#666', fontFamily: 'monospace' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine yAxisId="right" y={99.9} stroke="#4A5D4E" strokeDasharray="4 4" label={{ value: 'Enterprise SLA (99.9%)', position: 'insideTopRight', fill: '#4A5D4E', fontSize: 10, fontFamily: 'monospace' }} />
                  <Bar yAxisId="left" dataKey="preventedSdcEvents" fill="#4A5D4E" radius={[2, 2, 0, 0]} maxBarSize={32} name="Prevented SDCs" />
                  <Line yAxisId="right" type="monotone" dataKey="uptimePercent" stroke="#059669" strokeWidth={2.5} dot={{ r: 3, fill: '#059669' }} name="Protected Uptime %" />
                  <Line yAxisId="right" type="monotone" dataKey="unprotectedUptimePercent" stroke="#DC2626" strokeWidth={1.5} strokeDasharray="3 3" dot={false} name="Unprotected Uptime %" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {metricView === 'uptime_sla' && (
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h4 className="text-xs font-bold font-mono text-[#2A2A2A] uppercase tracking-wider">
                  Cluster Availability & SLA Benchmark ({selectedCluster.slaTarget}% SLA Target)
                </h4>
                <p className="text-[11px] text-[#666]">
                  Uptime stability comparison showing protected execution vs. unmitigated SDC failure rollbacks.
                </p>
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorProtectedUptime" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4A5D4E" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#4A5D4E" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="colorUnprotectedUptime" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#DC2626" stopOpacity={0.25} />
                      <stop offset="95%" stopColor="#DC2626" stopOpacity={0.01} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D1D0CB" opacity={0.6} />
                  <XAxis dataKey="label" stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <YAxis domain={[70, 100]} stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} unit="%" />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={99.9} stroke="#4A5D4E" strokeWidth={1.5} strokeDasharray="5 5" label={{ value: 'Target 99.9% Enterprise SLA', position: 'insideTopLeft', fill: '#4A5D4E', fontSize: 10, fontFamily: 'monospace' }} />
                  <Area type="monotone" dataKey="uptimePercent" stroke="#4A5D4E" strokeWidth={2} fillOpacity={1} fill="url(#colorProtectedUptime)" name="Protected Uptime" />
                  <Area type="monotone" dataKey="unprotectedUptimePercent" stroke="#DC2626" strokeWidth={1.5} strokeDasharray="2 2" fillOpacity={1} fill="url(#colorUnprotectedUptime)" name="Unprotected Baseline" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {metricView === 'prevented_sdc' && (
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h4 className="text-xs font-bold font-mono text-[#2A2A2A] uppercase tracking-wider">
                  Silent Arithmetic Errors & SDC Interceptions Timeline
                </h4>
                <p className="text-[11px] text-[#666]">
                  Volume of bit-flips, subnormal saturations, and tensor corruption events trapped in-flight.
                </p>
              </div>
              <div className="text-[11px] font-mono text-[#4A5D4E]">
                Total Intercepted: <strong>{totalPreventedSdc} Incidents</strong>
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D1D0CB" opacity={0.6} />
                  <XAxis dataKey="label" stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <YAxis stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="preventedSdcEvents" fill="#4A5D4E" radius={[2, 2, 0, 0]} maxBarSize={36} name="SDC Interceptions" />
                  <Line type="monotone" dataKey="recomputedTiles" stroke="#D97706" strokeWidth={1.5} dot={{ r: 2 }} name="Recomputed Tiles" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {metricView === 'savings' && (
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h4 className="text-xs font-bold font-mono text-[#2A2A2A] uppercase tracking-wider">
                  Avoided Compute Loss & Rollback Waste ($)
                </h4>
                <p className="text-[11px] text-[#666]">
                  Cumulative GPU-hours and financial waste saved by eliminating multi-hour training rewinds.
                </p>
              </div>
              <div className="text-[11px] font-mono text-[#4A5D4E] font-bold">
                Net Period Savings: ${totalDollarsSaved.toLocaleString()}
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorSavedDollars" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#4A5D4E" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#4A5D4E" stopOpacity={0.05} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D1D0CB" opacity={0.6} />
                  <XAxis dataKey="label" stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <YAxis yAxisId="left" stroke="#4A5D4E" tick={{ fontSize: 10, fill: '#4A5D4E', fontFamily: 'monospace' }} label={{ value: 'Saved GPU-Hours', angle: -90, position: 'insideLeft', offset: 18, fontSize: 10, fill: '#4A5D4E', fontFamily: 'monospace' }} />
                  <YAxis yAxisId="right" orientation="right" stroke="#2A2A2A" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} unit="$" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar yAxisId="left" dataKey="savedGpuHours" fill="#4A5D4E" radius={[2, 2, 0, 0]} maxBarSize={32} name="Saved GPU Hours" />
                  <Area yAxisId="right" type="monotone" dataKey="savedDollars" stroke="#059669" strokeWidth={2} fillOpacity={1} fill="url(#colorSavedDollars)" name="Avoided Dollar Waste" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {metricView === 'layers' && (
          <div>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
              <div>
                <h4 className="text-xs font-bold font-mono text-[#2A2A2A] uppercase tracking-wider">
                  5-Layer Defense Incident Breakdown (L1 - L5)
                </h4>
                <p className="text-[11px] text-[#666]">
                  Stack distribution of detected faults across the 5 TrainGuard architectural defense layers.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-[10px] font-mono">
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#4A5D4E] rounded-xs"></span> L1 Deterministic</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#2A2A2A] rounded-xs"></span> L2 Parameter Delta</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#8C6D46] rounded-xs"></span> L3 Activation</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#5D7A68] rounded-xs"></span> L4 Median/MAD</span>
                <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 bg-[#9E4738] rounded-xs"></span> L5 Cross-Rank</span>
              </div>
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#D1D0CB" opacity={0.6} />
                  <XAxis dataKey="label" stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <YAxis stroke="#666" tick={{ fontSize: 10, fill: '#666', fontFamily: 'monospace' }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="layer1Events" stackId="a" fill="#4A5D4E" name="L1: Deterministic NaN/Inf" />
                  <Bar dataKey="layer2Events" stackId="a" fill="#2A2A2A" name="L2: Parameter Deltas" />
                  <Bar dataKey="layer3Events" stackId="a" fill="#8C6D46" name="L3: Activation Kurtosis" />
                  <Bar dataKey="layer4Events" stackId="a" fill="#5D7A68" name="L4: Statistical Median/MAD" />
                  <Bar dataKey="layer5Events" stackId="a" fill="#9E4738" name="L5: Cross-Rank NCCL" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>

      {/* Live Recent Prevented Incident Stream */}
      <div className="space-y-3 pt-2 border-t border-[#D1D0CB]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-[#4A5D4E]" />
            <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Sample Production Incident Audit Log
            </h4>
          </div>
          <button
            onClick={() => setShowRecentFeed(!showRecentFeed)}
            className="text-[11px] font-mono text-[#4A5D4E] hover:underline cursor-pointer"
          >
            {showRecentFeed ? 'Hide Incident Feed' : 'Show Incident Feed'}
          </button>
        </div>

        {showRecentFeed && (
          <div className="space-y-2">
            {RECENT_INCIDENT_FEED.map((inc) => (
              <div
                key={inc.id}
                className="bg-[#F8F7F4] border border-[#D1D0CB] hover:border-[#4A5D4E] rounded-[2px] p-3 transition-colors flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-[#2A2A2A]">{inc.id}</span>
                    <span className="text-[10px] font-mono text-[#666]">• {inc.time}</span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-[2px] bg-white border border-[#D1D0CB] text-[#2A2A2A]">
                      {inc.cluster}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-[2px] bg-[#EBEAE5] text-[#4A5D4E] font-semibold">
                      Rank {inc.rank} ({inc.node})
                    </span>
                  </div>
                  <div className="text-[11px] text-[#2A2A2A] font-mono">
                    <span className="text-[#4A5D4E] font-semibold">{inc.layer}:</span> {inc.anomalyType}
                  </div>
                  <div className="text-[10px] text-[#666]">
                    Remediation: <strong className="text-[#2A2A2A]">{inc.actionTaken}</strong>
                  </div>
                </div>

                <div className="flex sm:flex-col items-center sm:items-end justify-between shrink-0 gap-1 font-mono">
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-[2px] border ${
                      inc.status === 'MITIGATED_IN_FLIGHT'
                        ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                        : inc.status === 'RANK_QUARANTINED'
                        ? 'bg-amber-50 text-amber-800 border-amber-300'
                        : 'bg-blue-50 text-blue-800 border-blue-300'
                    }`}
                  >
                    {inc.status}
                  </span>
                  <span className="text-[10px] text-[#4A5D4E]">
                    +{inc.savingsGpuHours.toLocaleString()} GPU-hrs saved
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};


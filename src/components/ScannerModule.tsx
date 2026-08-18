import React, { useState } from 'react';
import { CheckpointLayerScan } from '../types';
import { SAMPLE_CHECKPOINT_LAYERS } from '../data/mockCluster';
import {
  Search,
  UploadCloud,
  FileCheck,
  AlertTriangle,
  CheckCircle2,
  Layers,
  BarChart3,
  Activity,
  FileCode,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export const ScannerModule: React.FC = () => {
  const [layers, setLayers] = useState<CheckpointLayerScan[]>(SAMPLE_CHECKPOINT_LAYERS);
  const [selectedFilename, setSelectedFilename] = useState<string>('llama-3-70b-step42000.safetensors');
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [selectedLayer, setSelectedLayer] = useState<CheckpointLayerScan>(SAMPLE_CHECKPOINT_LAYERS[2]);

  const handleScanSample = (filename: string) => {
    setSelectedFilename(filename);
    setIsScanning(true);
    setTimeout(() => {
      setIsScanning(false);
    }, 600);
  };

  const corruptedCount = layers.filter((l) => l.status === 'corrupted').length;
  const suspiciousCount = layers.filter((l) => l.status === 'suspicious').length;

  const histogramData = [
    { range: '[-4.0, -3.0]', count: 12 },
    { range: '[-3.0, -2.0]', count: 142 },
    { range: '[-2.0, -1.0]', count: 1280 },
    { range: '[-1.0, 0.0]', count: 4890 },
    { range: '[0.0, 1.0]', count: 5120 },
    { range: '[1.0, 2.0]', count: 1340 },
    { range: '[2.0, 3.0]', count: 160 },
    { range: '[3.0, 4.0]', count: 24 },
    { range: '[10^18 (SDC)]', count: 3 }, // Corrupted outlier
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner & Dropzone */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Dropzone & Sample Selector */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
          <div className="flex items-center gap-2">
            <Search className="w-4 h-4 text-[#4A5D4E]" />
            <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Checkpoint & Weight Inspector
            </h3>
          </div>

          {/* File Dropzone */}
          <div className="border border-dashed border-[#D1D0CB] hover:border-[#4A5D4E] rounded-[2px] p-6 text-center bg-[#F8F7F4] transition-colors cursor-pointer">
            <UploadCloud className="w-8 h-8 text-[#666] mx-auto mb-2" />
            <div className="text-xs font-semibold text-[#2A2A2A]">
              Drop PyTorch <span className="font-mono text-[#4A5D4E]">.pt</span>, <span className="font-mono text-[#4A5D4E]">.bin</span>, or <span className="font-mono text-[#4A5D4E]">.safetensors</span> here
            </div>
            <p className="text-[11px] text-[#999] mt-1">
              Performs client-side tensor header audit & kurtosis anomaly profiling
            </p>
          </div>

          {/* Sample Checkpoints */}
          <div>
            <label className="text-xs font-medium text-[#666] block mb-2">
              Or Load Sample Checkpoint Audit:
            </label>
            <div className="space-y-1.5">
              {[
                { name: 'llama-3-70b-step42000.safetensors', tag: 'Corrupted (QK^T MSB Flip)' },
                { name: 'deepseek-v3-rank87-grad.pt', tag: 'Suspicious (MoE Carry)' },
                { name: 'qwen-2.5-clean-baseline.safetensors', tag: '100% Clean Nominal' },
              ].map((sample) => (
                <button
                  key={sample.name}
                  onClick={() => handleScanSample(sample.name)}
                  className={`w-full text-left px-3 py-2 rounded-[2px] text-xs font-mono transition-all flex items-center justify-between cursor-pointer ${
                    selectedFilename === sample.name
                      ? 'bg-[#4A5D4E] text-white font-bold shadow-xs'
                      : 'bg-[#F8F7F4] text-[#2A2A2A] hover:bg-[#EBEAE5] border border-[#D1D0CB]'
                  }`}
                >
                  <span className="truncate">{sample.name}</span>
                  <span className="text-[10px] opacity-80 shrink-0">{sample.tag}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Scan Summary & Anomaly Score */}
        <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 flex flex-col justify-between shadow-xs">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-xs font-mono text-[#666]">AUDITED FILE:</div>
                <div className="text-base font-bold text-[#2A2A2A] font-mono">{selectedFilename}</div>
              </div>
              <div className="flex gap-2">
                <span className="px-2.5 py-1 rounded-[2px] text-xs font-mono font-bold bg-[#8C2D2D] text-white">
                  {corruptedCount} SDC Corrupted
                </span>
                <span className="px-2.5 py-1 rounded-[2px] text-xs font-mono font-bold bg-[#F27D26] text-white">
                  {suspiciousCount} Suspicious
                </span>
              </div>
            </div>

            {/* Quick Metrics */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">TENSORS SCANNED</div>
                <div className="text-lg font-bold text-[#2A2A2A]">2,840</div>
              </div>
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">MAX KURTOSIS (K)</div>
                <div className="text-lg font-bold text-[#8C2D2D]">14.82 (Norm: ~3.0)</div>
              </div>
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">EXPONENT OVERFLOWS</div>
                <div className="text-lg font-bold text-[#8C2D2D]">29 Elements</div>
              </div>
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">HAMMING BIT DRIFT</div>
                <div className="text-lg font-bold text-[#F27D26]">+44.7 bits/hr</div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#D1D0CB] text-xs text-[#666] flex justify-between items-center">
            <span>Statistical anomaly detected in `model.layers.31.self_attn.q_proj.weight`</span>
            <span className="text-[#8C2D2D] font-bold font-mono">CRITICAL SDC RISK: 99.4%</span>
          </div>
        </div>
      </div>

      {/* Layer Risk Table */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono mb-3">
          Per-Layer SDC & Kurtosis Anomaly Breakdown
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-[#D1D0CB] text-[#666] font-mono">
                <th className="pb-2 font-semibold">Layer Name</th>
                <th className="pb-2 font-semibold">Shape</th>
                <th className="pb-2 font-semibold">Dtype</th>
                <th className="pb-2 font-semibold">Kurtosis (K)</th>
                <th className="pb-2 font-semibold">Overflows</th>
                <th className="pb-2 font-semibold">Subnormals</th>
                <th className="pb-2 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#D1D0CB] font-mono">
              {layers.map((layer) => {
                const isSelected = selectedLayer.layerName === layer.layerName;
                return (
                  <tr
                    key={layer.layerName}
                    onClick={() => setSelectedLayer(layer)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-[#F8F7F4] font-bold text-[#2A2A2A]' : 'hover:bg-[#F8F7F4]'
                    }`}
                  >
                    <td className="py-2.5 font-sans font-medium text-[#2A2A2A]">
                      {layer.layerName}
                    </td>
                    <td className="py-2.5 text-[#666]">{layer.shape}</td>
                    <td className="py-2.5 text-[#666]">{layer.dtype}</td>
                    <td
                      className={`py-2.5 ${
                        layer.kurtosis > 6 ? 'text-[#8C2D2D] font-bold' : 'text-[#2A2A2A]'
                      }`}
                    >
                      {layer.kurtosis.toFixed(2)}
                    </td>
                    <td className="py-2.5 text-[#666]">{layer.exponentOverflows}</td>
                    <td className="py-2.5 text-[#666]">{layer.subnormalCount}</td>
                    <td className="py-2.5">
                      <span
                        className={`px-2 py-0.5 rounded-[2px] text-[10px] uppercase font-bold ${
                          layer.status === 'normal'
                            ? 'bg-[#F8F7F4] text-[#4A5D4E] border border-[#D1D0CB]'
                            : layer.status === 'corrupted'
                            ? 'bg-[#8C2D2D] text-white animate-pulse'
                            : 'bg-[#F27D26] text-white'
                        }`}
                      >
                        {layer.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected Layer Weight Distribution Chart & Attention Heatmap */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Kurtosis Distribution Histogram */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
          <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider mb-1">
            WEIGHT DISTRIBUTION HISTOGRAM
          </h4>
          <p className="text-xs text-[#666] mb-3">
            Layer: {selectedLayer.layerName} (Showing standard normal vs outlier spikes)
          </p>

          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogramData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#EBEAE5" />
                <XAxis dataKey="range" stroke="#666" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis stroke="#666" tick={{ fontSize: 10, fontFamily: 'monospace' }} />
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
                <Bar dataKey="count">
                  {histogramData.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.range.includes('SDC') ? '#8C2D2D' : '#4A5D4E'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Synthetic Head Heatmap Matrix (16x16 Attention Slice) */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
          <div className="flex justify-between items-center mb-1">
            <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
              ATTENTION HEAD VALUE HEATMAP (16×16 TILE)
            </h4>
            <span className="text-[10px] font-mono text-[#8C2D2D] font-bold">
              3 EXPONENT OVERFLOW TILES
            </span>
          </div>
          <p className="text-xs text-[#666] mb-3">
            Heatmap representation of normalized weight activations
          </p>

          <div className="grid grid-cols-16 gap-1 bg-[#1C1C1A] p-3 rounded-[2px] border border-[#333330]">
            {Array.from({ length: 256 }, (_, i) => {
              const isCorruptTile = i === 47 || i === 94 || i === 188;
              let bg = 'bg-[#4A5D4E]/60';
              if (isCorruptTile) bg = 'bg-[#8C2D2D] animate-pulse';
              else if (i % 7 === 0) bg = 'bg-[#4A5D4E]';
              else if (i % 3 === 0) bg = 'bg-[#3B4A3E]';

              return (
                <div
                  key={i}
                  title={isCorruptTile ? `SDC Outlier at Cell [${Math.floor(i/16)}, ${i%16}]: Val = 1.48e+19` : `Cell [${Math.floor(i/16)}, ${i%16}] Nominal`}
                  className={`h-2.5 w-full rounded-[1px] ${bg}`}
                />
              );
            })}
          </div>

          <div className="flex justify-between items-center text-[10px] font-mono text-[#666] mt-2">
            <span>Tile Coord: [0..15, 0..15]</span>
            <span className="text-[#8C2D2D] font-bold">Red Cells: 10^18 Bit-Flip Anomalies</span>
          </div>
        </div>
      </div>
    </div>
  );
};

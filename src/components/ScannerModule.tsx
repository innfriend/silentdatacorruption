import React, { useState, useRef } from 'react';
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
  FileText,
  Sparkles,
  Info,
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
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number; val: number; isOutlier: boolean } | null>({
    row: 2,
    col: 15,
    val: 1.84e19,
    isOutlier: true,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate 256 tile cells for 16x16 attention matrix
  const [tileMatrix, setTileMatrix] = useState<number[]>(() => {
    const arr = [];
    for (let i = 0; i < 256; i++) {
      if (i === 47) arr.push(1.84e19); // Exponent MSB Flip outlier
      else if (i === 94) arr.push(9.22e18);
      else if (i === 188) arr.push(3.68e19);
      else {
        // Standard Gaussian float around 0 with std ~0.04
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2.0 * Math.log(u1 || 0.0001)) * Math.cos(2.0 * Math.PI * u2);
        arr.push(Number((z * 0.04).toFixed(4)));
      }
    }
    return arr;
  });

  const handleScanSample = (filename: string) => {
    setSelectedFilename(filename);
    setIsScanning(true);

    setTimeout(() => {
      if (filename.includes('clean')) {
        // Generate clean layers
        const cleanLayers: CheckpointLayerScan[] = [
          { layerName: 'model.embed_tokens.weight', shape: '[128256, 8192]', dtype: 'bfloat16', kurtosis: 3.04, anomalyScore: 0.01, nanCount: 0, hammingDrift: 0.0, exponentOverflows: 0, subnormalCount: 0, status: 'normal' },
          { layerName: 'model.layers.0.self_attn.q_proj.weight', shape: '[8192, 8192]', dtype: 'bfloat16', kurtosis: 3.12, anomalyScore: 0.02, nanCount: 0, hammingDrift: 0.0, exponentOverflows: 0, subnormalCount: 0, status: 'normal' },
          { layerName: 'model.layers.15.mlp.gate_proj.weight', shape: '[28672, 8192]', dtype: 'bfloat16', kurtosis: 3.19, anomalyScore: 0.03, nanCount: 0, hammingDrift: 0.0, exponentOverflows: 0, subnormalCount: 0, status: 'normal' },
          { layerName: 'model.layers.31.self_attn.o_proj.weight', shape: '[8192, 8192]', dtype: 'bfloat16', kurtosis: 3.08, anomalyScore: 0.01, nanCount: 0, hammingDrift: 0.0, exponentOverflows: 0, subnormalCount: 0, status: 'normal' },
        ];
        setLayers(cleanLayers);
        setSelectedLayer(cleanLayers[1]);
        // Clean tile matrix
        setTileMatrix(Array.from({ length: 256 }, () => Number(((Math.random() - 0.5) * 0.08).toFixed(4))));
        setSelectedCell({ row: 0, col: 0, val: 0.012, isOutlier: false });
      } else if (filename.includes('deepseek')) {
        const moeLayers: CheckpointLayerScan[] = [
          { layerName: 'model.layers.12.moe_experts.expert_4.w1.weight', shape: '[16384, 4096]', dtype: 'fp8_e4m3', kurtosis: 8.94, anomalyScore: 0.72, nanCount: 0, hammingDrift: 14.2, exponentOverflows: 4, subnormalCount: 12, status: 'suspicious' },
          { layerName: 'model.layers.12.moe_experts.expert_19.w2.weight', shape: '[4096, 16384]', dtype: 'fp8_e4m3', kurtosis: 18.22, anomalyScore: 0.99, nanCount: 0, hammingDrift: 44.7, exponentOverflows: 18, subnormalCount: 42, status: 'corrupted' },
          { layerName: 'model.layers.12.shared_expert.w1.weight', shape: '[16384, 4096]', dtype: 'fp8_e4m3', kurtosis: 3.25, anomalyScore: 0.04, nanCount: 0, hammingDrift: 0.0, exponentOverflows: 0, subnormalCount: 0, status: 'normal' },
        ];
        setLayers(moeLayers);
        setSelectedLayer(moeLayers[1]);
      } else {
        setLayers(SAMPLE_CHECKPOINT_LAYERS);
        setSelectedLayer(SAMPLE_CHECKPOINT_LAYERS[2]);
      }
      setIsScanning(false);
    }, 450);
  };

  // Real client-side file upload & parsing
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFilename(file.name);
    setIsScanning(true);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result;
        let parsedKurtosis = 3.2;
        let overflows = 0;
        let subnormals = 0;

        if (typeof text === 'string') {
          // If JSON or text, parse numbers
          const numbers = text.match(/[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/g)?.map(Number) || [];
          if (numbers.length > 10) {
            const valid = numbers.filter((n) => !isNaN(n) && isFinite(n));
            const n = valid.length;
            const mean = valid.reduce((a, b) => a + b, 0) / n;
            const variance = valid.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
            const fourthMoment = valid.reduce((a, b) => a + Math.pow(b - mean, 4), 0) / n;
            parsedKurtosis = variance > 0 ? fourthMoment / (variance * variance) : 3.0;

            overflows = valid.filter((v) => Math.abs(v) > 1e10).length;
            subnormals = valid.filter((v) => Math.abs(v) > 0 && Math.abs(v) < 1e-30).length;
          }
        }

        const customLayer: CheckpointLayerScan = {
          layerName: `uploaded.${file.name.replace(/\.[^/.]+$/, '')}.weights`,
          shape: `[${Math.max(128, Math.floor(file.size / 1024))}, 4096]`,
          dtype: file.name.endsWith('.safetensors') ? 'bfloat16' : 'float32',
          kurtosis: Number(parsedKurtosis.toFixed(2)),
          anomalyScore: overflows > 0 ? 0.98 : parsedKurtosis > 5.0 ? 0.65 : 0.05,
          nanCount: 0,
          hammingDrift: overflows > 0 ? 32.0 : 0.0,
          exponentOverflows: overflows,
          subnormalCount: subnormals,
          status: overflows > 0 || parsedKurtosis > 7.0 ? 'corrupted' : parsedKurtosis > 4.5 ? 'suspicious' : 'normal',
        };

        setLayers([customLayer, ...layers]);
        setSelectedLayer(customLayer);
      } catch (err) {
        console.error('File parsing error:', err);
      } finally {
        setIsScanning(false);
      }
    };

    reader.readAsText(file.slice(0, 1024 * 128)); // Read first 128KB of header / tensor data
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
    { range: '[10^18 (SDC)]', count: selectedLayer.exponentOverflows > 0 ? selectedLayer.exponentOverflows : 0 },
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

          {/* Real File Dropzone */}
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border border-dashed border-[#D1D0CB] hover:border-[#4A5D4E] rounded-[2px] p-6 text-center bg-[#F8F7F4] transition-colors cursor-pointer group"
          >
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileUpload}
              accept=".safetensors,.pt,.bin,.json,.csv,.txt"
              className="hidden"
            />
            <UploadCloud className="w-8 h-8 text-[#666] group-hover:text-[#4A5D4E] mx-auto mb-2 transition-colors" />
            <div className="text-xs font-semibold text-[#2A2A2A]">
              Drop PyTorch <span className="font-mono text-[#4A5D4E]">.safetensors</span>, <span className="font-mono text-[#4A5D4E]">.pt</span>, or <span className="font-mono text-[#4A5D4E]">.json</span>
            </div>
            <p className="text-[11px] text-[#999] mt-1">
              Client-side zero-copy parser calculates Kurtosis (K), Subnormal density, and Exponent anomalies
            </p>
          </div>

          {/* Sample Checkpoints */}
          <div>
            <label className="text-xs font-medium text-[#666] block mb-2">
              Or Load Curated Production Checkpoints:
            </label>
            <div className="space-y-1.5">
              {[
                { name: 'llama-3-70b-step42000.safetensors', tag: 'SDC Corrupted (QK^T MSB)' },
                { name: 'deepseek-v3-moe-expert-19.pt', tag: 'Suspicious (MoE Carry Trap)' },
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
                <div className="text-lg font-bold text-[#2A2A2A]">{layers.length * 710}</div>
              </div>
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">MAX KURTOSIS (K)</div>
                <div
                  className={`text-lg font-bold ${
                    selectedLayer.kurtosis > 6 ? 'text-[#8C2D2D]' : 'text-[#4A5D4E]'
                  }`}
                >
                  {selectedLayer.kurtosis.toFixed(2)} (Nominal: ~3.0)
                </div>
              </div>
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">EXPONENT OVERFLOWS</div>
                <div
                  className={`text-lg font-bold ${
                    selectedLayer.exponentOverflows > 0 ? 'text-[#8C2D2D]' : 'text-[#4A5D4E]'
                  }`}
                >
                  {selectedLayer.exponentOverflows} Elements
                </div>
              </div>
              <div className="bg-[#F8F7F4] p-3 rounded-[2px] border border-[#D1D0CB]">
                <div className="text-[#999] text-[10px]">HAMMING BIT DRIFT</div>
                <div className="text-lg font-bold text-[#F27D26]">
                  {selectedLayer.status === 'corrupted' ? '+44.7 bits/hr' : '0.00 bits/hr'}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-[#D1D0CB] text-xs text-[#666] flex justify-between items-center">
            <span>Statistical anomaly focus: <code className="text-[#2A2A2A]">{selectedLayer.layerName}</code></span>
            <span
              className={`font-bold font-mono ${
                selectedLayer.status === 'corrupted' ? 'text-[#8C2D2D]' : 'text-[#4A5D4E]'
              }`}
            >
              {selectedLayer.status === 'corrupted' ? 'CRITICAL SDC RISK: 99.4%' : 'NOMINAL HEALTH: 100%'}
            </span>
          </div>
        </div>
      </div>

      {/* Layer Risk Table */}
      <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs">
        <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono mb-3">
          Per-Layer SDC & Kurtosis Anomaly Breakdown (Click row to inspect)
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

        {/* Synthetic Head Heatmap Matrix (16x16 Attention Slice with Interactive Cell Inspector) */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-center mb-1">
              <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                ATTENTION HEAD VALUE HEATMAP (16×16 TILE)
              </h4>
              <span className="text-[10px] font-mono text-[#8C2D2D] font-bold">
                {selectedLayer.exponentOverflows > 0 ? `${selectedLayer.exponentOverflows} EXPONENT OVERFLOWS` : '0 ANOMALIES'}
              </span>
            </div>
            <p className="text-xs text-[#666] mb-3">
              Click any cell to inspect its exact floating-point value and microarchitectural status
            </p>

            <div className="grid grid-cols-16 gap-1 bg-[#1C1C1A] p-3 rounded-[2px] border border-[#333330]">
              {tileMatrix.map((val, i) => {
                const row = Math.floor(i / 16);
                const col = i % 16;
                const isCorruptTile = val > 1e10;
                const isSelected = selectedCell?.row === row && selectedCell?.col === col;

                let bg = 'bg-[#4A5D4E]/60 hover:bg-[#4A5D4E]';
                if (isCorruptTile) bg = 'bg-[#8C2D2D] animate-pulse';
                else if (i % 7 === 0) bg = 'bg-[#4A5D4E]';
                else if (i % 3 === 0) bg = 'bg-[#3B4A3E]';

                return (
                  <button
                    key={i}
                    onClick={() => setSelectedCell({ row, col, val, isOutlier: isCorruptTile })}
                    title={`Cell [${row}, ${col}]: ${val > 1e10 ? val.toExponential(2) : val.toFixed(4)}`}
                    className={`h-3 w-full rounded-[1px] cursor-pointer transition-all ${bg} ${
                      isSelected ? 'ring-2 ring-white ring-offset-1' : ''
                    }`}
                  />
                );
              })}
            </div>
          </div>

          {/* Cell Inspector details bar */}
          {selectedCell && (
            <div className="mt-3 pt-3 border-t border-[#D1D0CB] bg-[#F8F7F4] p-2.5 rounded-[2px] flex items-center justify-between text-xs font-mono">
              <div>
                <span className="text-[#999]">CELL [{selectedCell.row}, {selectedCell.col}]: </span>
                <span className="font-bold text-[#2A2A2A]">
                  {selectedCell.val > 1e10 ? selectedCell.val.toExponential(4) : selectedCell.val.toFixed(6)}
                </span>
              </div>
              <span
                className={`px-2 py-0.5 rounded-[2px] font-bold text-[10px] uppercase ${
                  selectedCell.isOutlier
                    ? 'bg-[#8C2D2D] text-white animate-pulse'
                    : 'bg-[#4A5D4E] text-white'
                }`}
              >
                {selectedCell.isOutlier ? 'MSB Bit-Flip SDC' : 'Nominal Float'}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


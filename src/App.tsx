import React, { useState } from 'react';
import { ModuleTab, GpuRank, TelemetryLog, LossDataPoint } from './types';
import { INITIAL_RANKS, INITIAL_LOGS, INITIAL_LOSS_CURVE } from './data/mockCluster';
import { Navbar } from './components/Navbar';
import { SubHeader } from './components/SubHeader';
import { OverviewModule } from './components/OverviewModule';
import { SimulatorModule } from './components/SimulatorModule';
import { KernelSuiteModule } from './components/KernelSuiteModule';
import { ScannerModule } from './components/ScannerModule';
import { DiagnosticModule } from './components/DiagnosticModule';
import { RoiCalculatorModule } from './components/RoiCalculatorModule';
import { ScienceModule } from './components/ScienceModule';
import { Footer } from './components/Footer';

export default function App() {
  const [activeTab, setActiveTab] = useState<ModuleTab>('overview');
  const [ranks, setRanks] = useState<GpuRank[]>(INITIAL_RANKS);
  const [logs, setLogs] = useState<TelemetryLog[]>(INITIAL_LOGS);
  const [lossData, setLossData] = useState<LossDataPoint[]>(INITIAL_LOSS_CURVE);
  const [recomputedTiles, setRecomputedTiles] = useState<number>(14);

  const healthyCount = ranks.filter((r) => r.status === 'healthy').length;
  const corruptedCount = ranks.filter((r) => r.status === 'corrupted').length;
  const quarantinedCount = ranks.filter((r) => r.status === 'quarantined').length;
  const recomputingCount = ranks.filter((r) => r.status === 'recomputing').length;

  const handleResetCluster = () => {
    setRanks(INITIAL_RANKS);
    setLossData(INITIAL_LOSS_CURVE);
    const resetLog: TelemetryLog = {
      id: `log-${Date.now()}`,
      timestamp: new Date().toTimeString().split(' ')[0] + '.000',
      rank: 0,
      nodeId: 'dgx-hopper-01',
      severity: 'info',
      event: 'CLUSTER_TOPOLOGY_RESET',
      details: 'All 128 GPU ranks restored to 100% nominal state with zero active quarantines.',
      durationMs: 0.5,
    };
    setLogs((prev) => [resetLog, ...prev.slice(0, 49)]);
  };

  const handleQuickFaultInject = () => {
    const randomRank = Math.floor(Math.random() * 128);
    const timeStr = new Date().toTimeString().split(' ')[0] + '.' + Math.floor(Math.random() * 900 + 100);

    setRanks((prev) =>
      prev.map((r) => {
        if (r.id === randomRank) {
          return {
            ...r,
            status: 'corrupted',
            lastFault: 'exponent_msb',
            lastFaultTime: timeStr,
            parityDelta: 1.482e3,
          };
        }
        return r;
      })
    );

    const newLog: TelemetryLog = {
      id: `log-${Date.now()}`,
      timestamp: timeStr,
      rank: randomRank,
      nodeId: ranks[randomRank]?.nodeId || 'dgx-hopper-06',
      severity: 'critical',
      event: 'PARITY_INVARIANT_VIOLATION',
      details: `Cosmic ray bit-flip in Tensor Core MMA. Invariant residual delta = 1.482e+03 > 1.0e-4 eps.`,
      deltaNorm: 1.482e3,
      durationMs: 3.18,
    };
    setLogs((prev) => [newLog, ...prev.slice(0, 49)]);

    setTimeout(() => {
      setRanks((prev) =>
        prev.map((r) => {
          if (r.id === randomRank) {
            return {
              ...r,
              status: 'quarantined',
              parityDelta: 1.2e-6,
              recomputedTiles: r.recomputedTiles + 1,
            };
          }
          return r;
        })
      );
      setRecomputedTiles((prev) => prev + 1);
    }, 1200);
  };

  return (
    <div className="min-h-screen bg-[#F2F1ED] text-[#2A2A2A] flex flex-col font-sans selection:bg-[#4A5D4E] selection:text-white">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthyCount={healthyCount}
        corruptedCount={corruptedCount}
        quarantinedCount={quarantinedCount}
        recomputingCount={recomputingCount}
        totalGpus={ranks.length}
      />

      {/* SubHeader Breadcrumb & Actions */}
      <SubHeader
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        onResetCluster={handleResetCluster}
        onQuickFaultInject={handleQuickFaultInject}
      />

      {/* Main Module Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'overview' && (
          <OverviewModule
            setActiveTab={setActiveTab}
            healthyRanks={healthyCount}
            totalRanks={ranks.length}
            quarantinedRanks={quarantinedCount}
            corruptedRanks={corruptedCount}
            recomputedTiles={recomputedTiles}
          />
        )}

        {activeTab === 'simulator' && (
          <SimulatorModule
            ranks={ranks}
            setRanks={setRanks}
            logs={logs}
            setLogs={setLogs}
            lossData={lossData}
            setLossData={setLossData}
            onResetCluster={handleResetCluster}
            recomputedTiles={recomputedTiles}
            setRecomputedTiles={setRecomputedTiles}
          />
        )}

        {activeTab === 'kernels' && <KernelSuiteModule />}

        {activeTab === 'scanner' && <ScannerModule />}

        {activeTab === 'diagnostic' && <DiagnosticModule />}

        {activeTab === 'roi' && <RoiCalculatorModule />}

        {activeTab === 'science' && <ScienceModule />}
      </main>

      {/* Enterprise Footer */}
      <Footer setActiveTab={setActiveTab} />
    </div>
  );
}

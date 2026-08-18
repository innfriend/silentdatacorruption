import React, { useState } from 'react';
import { DiagnosticMessage, DiagnosticScenario, TrainGuardEvent, RecommendedAction } from '../types';
import { DIAGNOSTIC_SCENARIOS } from '../data/mockCluster';
import { INITIAL_TRAINGUARD_EVENTS } from '../data/trainGuardEngine';
import {
  Zap,
  Send,
  Sparkles,
  Bot,
  User,
  Terminal,
  Cpu,
  AlertOctagon,
  Copy,
  Check,
  RotateCcw,
  ShieldAlert,
  FileText,
  Download,
  X,
  FileCode,
  CheckCircle2,
  HelpCircle,
  MapPin,
  Clock,
  Gauge,
  Activity,
  Layers,
  ArrowRight,
  Sliders,
} from 'lucide-react';

export const DiagnosticModule: React.FC = () => {
  const [events] = useState<TrainGuardEvent[]>(INITIAL_TRAINGUARD_EVENTS);
  const [selectedEventId, setSelectedEventId] = useState<string>(INITIAL_TRAINGUARD_EVENTS[0].event_id);
  const [activeTab, setActiveTab] = useState<'7questions' | 'chat' | 'json'>('7questions');
  const [scenarios] = useState<DiagnosticScenario[]>(DIAGNOSTIC_SCENARIOS);
  const [selectedScenario, setSelectedScenario] = useState<DiagnosticScenario>(DIAGNOSTIC_SCENARIOS[0]);
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [customLogs, setCustomLogs] = useState<string>(DIAGNOSTIC_SCENARIOS[0].logs);
  const [rankContext, setRankContext] = useState<string>('Rank 42 (Node dgx-hopper-06)');
  const [loading, setLoading] = useState<boolean>(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showReportModal, setShowReportModal] = useState<boolean>(false);
  const [copiedReport, setCopiedReport] = useState<boolean>(false);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);

  const selectedEvent = events.find((e) => e.event_id === selectedEventId) || events[0];

  const [messages, setMessages] = useState<DiagnosticMessage[]>([
    {
      id: 'init-1',
      sender: 'gemini',
      content: `### 🔬 TrainGuard v0.3 Forensic Diagnostic Online
I am ready to perform hardware-level and numerical integrity analysis across all 5 monitoring layers.
Evidence synthesis answers the 7 mandatory reliability questions:
1. **What happened?**
2. **Where did it happen?**
3. **When did it start?**
4. **How severe is it?**
5. **How confident are we?**
6. **What evidence supports it?**
7. **What should the engineer do next?**

Select an event or scenario to inspect root causes and execute automated mitigation playbooks.`,
      timestamp: '11:48:22',
      modelUsed: 'gemini-3.7-flash',
    },
  ]);

  const handleSelectScenario = (sc: DiagnosticScenario) => {
    setSelectedScenario(sc);
    setCustomLogs(sc.logs);
    setRankContext(`Rank ${sc.rank} (Node ${sc.node})`);
  };

  const handleRunDiagnosis = async () => {
    setLoading(true);

    const userMessage: DiagnosticMessage = {
      id: `msg-${Date.now()}`,
      sender: 'user',
      content: `Investigate incident on ${rankContext}:\n**Scenario**: ${selectedScenario.title}\n\n**Captured Logs**:\n\`\`\`\n${customLogs}\n\`\`\`\n\n${customPrompt || 'Synthesize 7-question forensic analysis and provide hardware/software mitigation playbook.'}`,
      timestamp: new Date().toTimeString().split(' ')[0],
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch('/api/gemini/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: customPrompt || 'Synthesize 7-question forensic analysis and provide hardware/software mitigation playbook.',
          scenario: selectedScenario.title,
          logs: customLogs,
          rankInfo: rankContext,
        }),
      });

      const data = await response.json();

      const aiMessage: DiagnosticMessage = {
        id: `msg-${Date.now()}-ai`,
        sender: 'gemini',
        content: data.analysis || 'Diagnosis completed.',
        timestamp: new Date().toTimeString().split(' ')[0],
        modelUsed: data.modelUsed,
        isFallback: data.isFallback,
      };

      setMessages((prev) => [...prev, aiMessage]);
      setActiveTab('chat');
    } catch (err: any) {
      console.error('Diagnosis failed:', err);
      const errorMessage: DiagnosticMessage = {
        id: `msg-${Date.now()}-err`,
        sender: 'gemini',
        content: `### ⚠️ Diagnostic Error
Unable to query backend: ${err.message}. Showing local offline forensic synthesis.`,
        timestamp: new Date().toTimeString().split(' ')[0],
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setCustomPrompt('');
    }
  };

  const handleCopyText = (text: string, idx: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleTriggerAction = (action: RecommendedAction) => {
    setActionSuccess(`Action '${action}' executed successfully. Node/parameter state synchronized.`);
    setTimeout(() => setActionSuccess(null), 3500);
  };

  const generateReportMarkdown = () => {
    return `# 🚨 TRAINGUARD v0.3 FORENSIC INCIDENT REPORT
**Schema Version**: 1.0
**Event ID**: ${selectedEvent.event_id}
**Model ID**: ${selectedEvent.model_id} (v${selectedEvent.model_version})
**Run ID**: ${selectedEvent.run_id}
**Step**: ${selectedEvent.step}
**Timestamp**: ${selectedEvent.timestamp}

---

## 1. WHAT HAPPENED?
- **Event Type**: ${selectedEvent.event_type}
- **Layer**: Layer ${selectedEvent.layer_number} (${selectedEvent.layer})

## 2. WHERE DID IT HAPPEN?
- **Location**: ${selectedEvent.location.module || 'Global'} -> ${selectedEvent.location.parameter || selectedEvent.location.tensor_name || 'N/A'}
- **Exact Parameter Index**: ${selectedEvent.location.index !== undefined ? selectedEvent.location.index : 'N/A'}
- **Rank / Node**: Rank ${selectedEvent.location.rank || 42} (${selectedEvent.location.node_id || 'dgx-hopper-06'})

## 3. WHEN DID IT START?
- **Start Step**: ${selectedEvent.step}
- **Window**: Past ${selectedEvent.evidence.window} steps (Persistence: ${selectedEvent.evidence.persistence} consecutive occurrences)

## 4. HOW SEVERE IS IT?
- **Severity**: ${selectedEvent.severity}

## 5. HOW CONFIDENT ARE WE?
- **Confidence**: ${(selectedEvent.confidence * 100).toFixed(1)}% (Bayesian robust z-score confirmation)

## 6. WHAT EVIDENCE SUPPORTS IT?
- **Signals Triggered**: ${selectedEvent.signals_triggered.join(', ')}
- **Observed**: ${JSON.stringify(selectedEvent.observed)}
- **Baseline (Median / MAD)**: Median = ${selectedEvent.baseline.median}, MAD = ${selectedEvent.baseline.mad}
- **Baseline Shielding**: ${selectedEvent.evidence.baseline_poisoned ? 'POISONED' : 'CLEAN (Strict No-Poisoning Enforced)'}
- **Detailed Explanation**: ${selectedEvent.evidence.explanation}

## 7. WHAT SHOULD THE ENGINEER DO NEXT?
- **Recommended Action**: ${selectedEvent.recommended_action}
- **Playbook**: ${selectedEvent.action_detail}

---
*Report compiled automatically by TrainGuard v0.3 Production SDK.*
`;
  };

  const handleDownloadJsonReport = () => {
    const jsonStr = JSON.stringify(selectedEvent, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trainguard_event_${selectedEvent.event_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadReport = () => {
    const md = generateReportMarkdown();
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TrainGuard-Incident-${selectedEvent.event_id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyReport = () => {
    navigator.clipboard.writeText(generateReportMarkdown());
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Top Action Notification */}
      {actionSuccess && (
        <div className="bg-[#4A5D4E] text-white px-4 py-2.5 rounded-[2px] text-xs font-mono flex items-center justify-between shadow-xs">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-200" />
            <span>{actionSuccess}</span>
          </div>
          <button onClick={() => setActionSuccess(null)} className="text-white hover:text-gray-200 cursor-pointer">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Main Grid: Left Event & Scenario Selector (4 cols) | Right 7-Question / Chat Panel (8 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Events & Scenarios (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          {/* Active Events Selector */}
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-[#D1D0CB]">
              <div className="flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 text-[#4A5D4E]" />
                <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                  Monitored Reliability Events
                </h3>
              </div>
              <span className="text-[10px] font-mono text-[#666] bg-[#F8F7F4] px-2 py-0.5 rounded-[2px] border border-[#D1D0CB]">
                {events.length} Events
              </span>
            </div>

            <div className="space-y-2">
              {events.map((evt) => {
                const isSelected = evt.event_id === selectedEventId;
                const badgeColor =
                  evt.severity === 'CRITICAL'
                    ? 'bg-red-100 text-red-800 border-red-200'
                    : evt.severity === 'ERROR'
                    ? 'bg-amber-100 text-amber-800 border-amber-200'
                    : 'bg-blue-100 text-blue-800 border-blue-200';

                return (
                  <button
                    key={evt.event_id}
                    onClick={() => {
                      setSelectedEventId(evt.event_id);
                      setRankContext(`Rank ${evt.location.rank || 42} (Node ${evt.location.node_id || 'dgx-06'})`);
                    }}
                    className={`w-full text-left p-3 rounded-[2px] text-xs transition-all cursor-pointer flex flex-col justify-between ${
                      isSelected
                        ? 'bg-[#4A5D4E] text-white shadow-xs'
                        : 'bg-[#F8F7F4] text-[#2A2A2A] hover:bg-[#EBEAE5] border border-[#D1D0CB]'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-[10px] uppercase font-bold tracking-wider">
                        {evt.event_id} • Step {evt.step}
                      </span>
                      <span className={`px-1.5 py-0.2 text-[9px] font-mono font-bold rounded-[2px] border ${badgeColor}`}>
                        {evt.severity}
                      </span>
                    </div>
                    <div className="font-bold text-xs line-clamp-1 mb-1">{evt.event_type.replace(/_/g, ' ')}</div>
                    <div className={`text-[10px] font-mono truncate ${isSelected ? 'text-emerald-100' : 'text-[#666]'}`}>
                      {evt.location.parameter || evt.location.tensor_name} • {(evt.confidence * 100).toFixed(0)}% Conf
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Quick Hardware Backtrace & Scenarios */}
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-4 shadow-xs space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-[#D1D0CB]">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-[#4A5D4E]" />
                <h4 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                  Raw Hardware Logs
                </h4>
              </div>
              <span className="text-[10px] font-mono text-[#666]">NVLink / DCGM</span>
            </div>

            <textarea
              rows={4}
              value={customLogs}
              onChange={(e) => setCustomLogs(e.target.value)}
              className="w-full bg-[#1C1C1A] text-emerald-300 font-mono text-[11px] p-2.5 rounded-[2px] border border-[#333330] focus:outline-none focus:border-[#4A5D4E]"
            />

            <button
              onClick={handleRunDiagnosis}
              disabled={loading}
              className="w-full inline-flex items-center justify-center gap-2 px-3 py-2 bg-[#4A5D4E] hover:bg-[#3B4A3E] disabled:opacity-50 text-white text-xs font-bold rounded-[2px] shadow-xs transition-colors cursor-pointer"
            >
              <Zap className="w-4 h-4" />
              <span>{loading ? 'AI Analyzing...' : 'Run Gemini 3.7 Diagnostic'}</span>
            </button>
          </div>
        </div>

        {/* Right Column: 7-Question Evidence Inspector / Chat / JSON (8 cols) */}
        <div className="lg:col-span-8 bg-white border border-[#D1D0CB] rounded-[3px] p-5 shadow-xs flex flex-col justify-between">
          <div>
            {/* Header with 3 Tabs */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#D1D0CB] mb-4">
              <div>
                <div className="flex items-center gap-2">
                  <div className="w-7 h-7 rounded-[2px] bg-[#4A5D4E] text-white flex items-center justify-center">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                      TrainGuard Evidence Inspector ({selectedEvent.event_id})
                    </h4>
                    <div className="text-[11px] text-[#666]">
                      Layer {selectedEvent.layer_number} • {selectedEvent.layer.replace(/_/g, ' ')}
                    </div>
                  </div>
                </div>
              </div>

              {/* Tab Switcher & Export */}
              <div className="flex items-center gap-2">
                <div className="flex bg-[#F8F7F4] border border-[#D1D0CB] p-0.5 rounded-[2px]">
                  <button
                    onClick={() => setActiveTab('7questions')}
                    className={`px-2.5 py-1 text-xs font-mono font-bold rounded-[2px] cursor-pointer transition-colors ${
                      activeTab === '7questions' ? 'bg-[#4A5D4E] text-white shadow-xs' : 'text-[#666] hover:text-[#2A2A2A]'
                    }`}
                  >
                    7 Questions
                  </button>
                  <button
                    onClick={() => setActiveTab('chat')}
                    className={`px-2.5 py-1 text-xs font-mono font-bold rounded-[2px] cursor-pointer transition-colors ${
                      activeTab === 'chat' ? 'bg-[#4A5D4E] text-white shadow-xs' : 'text-[#666] hover:text-[#2A2A2A]'
                    }`}
                  >
                    Gemini Chat
                  </button>
                  <button
                    onClick={() => setActiveTab('json')}
                    className={`px-2.5 py-1 text-xs font-mono font-bold rounded-[2px] cursor-pointer transition-colors ${
                      activeTab === 'json' ? 'bg-[#4A5D4E] text-white shadow-xs' : 'text-[#666] hover:text-[#2A2A2A]'
                    }`}
                  >
                    Schema JSON
                  </button>
                </div>

                <button
                  onClick={() => setShowReportModal(true)}
                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-[2px] bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] text-xs font-mono font-bold cursor-pointer"
                >
                  <FileText className="w-3.5 h-3.5 text-[#4A5D4E]" />
                  <span>Report</span>
                </button>
              </div>
            </div>

            {/* TAB 1: 7-QUESTION EVIDENCE INSPECTOR */}
            {activeTab === '7questions' && (
              <div className="space-y-4">
                {/* 7 Questions Cards Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  {/* Q1: What happened? */}
                  <div className="p-3.5 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-mono font-bold text-[#4A5D4E] uppercase">
                      <span>1. What happened?</span>
                      <span className="text-[10px] text-[#666]">Layer {selectedEvent.layer_number}</span>
                    </div>
                    <div className="font-serif font-bold text-sm text-[#2A2A2A]">
                      {selectedEvent.event_type.replace(/_/g, ' ')}
                    </div>
                    <p className="text-[11px] text-[#666]">
                      {selectedEvent.evidence.explanation}
                    </p>
                  </div>

                  {/* Q2: Where did it happen? */}
                  <div className="p-3.5 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-mono font-bold text-[#4A5D4E] uppercase">
                      <span>2. Where did it happen?</span>
                      <MapPin className="w-3.5 h-3.5 text-[#4A5D4E]" />
                    </div>
                    <div className="font-mono font-bold text-xs text-[#2A2A2A] truncate">
                      {selectedEvent.location.module ? `${selectedEvent.location.module} -> ` : ''}
                      {selectedEvent.location.parameter || selectedEvent.location.tensor_name}
                      {selectedEvent.location.index !== undefined ? `[${selectedEvent.location.index}]` : ''}
                    </div>
                    <div className="text-[11px] text-[#666] font-mono">
                      Rank {selectedEvent.location.rank || 42} • Node {selectedEvent.location.node_id || 'dgx-06'}
                    </div>
                  </div>

                  {/* Q3: When did it start? */}
                  <div className="p-3.5 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-mono font-bold text-[#4A5D4E] uppercase">
                      <span>3. When did it start?</span>
                      <Clock className="w-3.5 h-3.5 text-[#4A5D4E]" />
                    </div>
                    <div className="font-mono font-bold text-xs text-[#2A2A2A]">
                      Training Step {selectedEvent.step}
                    </div>
                    <div className="text-[11px] text-[#666] font-mono">
                      Timestamp: {selectedEvent.timestamp} (Window: {selectedEvent.evidence.window} steps)
                    </div>
                  </div>

                  {/* Q4 & Q5: Severity & Confidence */}
                  <div className="p-3.5 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-1.5">
                    <div className="flex items-center justify-between text-[11px] font-mono font-bold text-[#4A5D4E] uppercase">
                      <span>4 & 5. Severity & Confidence</span>
                      <Gauge className="w-3.5 h-3.5 text-[#4A5D4E]" />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className={`px-2 py-0.5 text-xs font-mono font-bold rounded-[2px] ${
                        selectedEvent.severity === 'CRITICAL'
                          ? 'bg-red-100 text-red-800 border border-red-200'
                          : selectedEvent.severity === 'ERROR'
                          ? 'bg-amber-100 text-amber-800 border border-amber-200'
                          : 'bg-blue-100 text-blue-800 border border-blue-200'
                      }`}>
                        {selectedEvent.severity}
                      </span>
                      <span className="font-mono text-xs font-bold text-[#2A2A2A]">
                        {(selectedEvent.confidence * 100).toFixed(1)}% Confidence
                      </span>
                    </div>
                    <div className="w-full bg-[#D1D0CB] h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-[#4A5D4E] h-full rounded-full"
                        style={{ width: `${selectedEvent.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Q6: What evidence supports it? (Full Width) */}
                <div className="p-4 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] space-y-2 text-xs">
                  <div className="flex items-center justify-between text-[11px] font-mono font-bold text-[#4A5D4E] uppercase">
                    <span>6. What evidence supports it?</span>
                    <span className="text-[10px] text-[#4A5D4E] font-bold">
                      Strict No-Baseline Poisoning: ACTIVE
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 font-mono text-[11px]">
                    <div className="p-2 bg-white border border-[#D1D0CB] rounded-[2px]">
                      <div className="text-[#666]">Observed Delta / Val</div>
                      <div className="font-bold text-[#2A2A2A]">
                        {selectedEvent.observed.delta !== undefined ? selectedEvent.observed.delta.toFixed(4) : selectedEvent.observed.value?.toFixed(4) || 'N/A'}
                      </div>
                    </div>
                    <div className="p-2 bg-white border border-[#D1D0CB] rounded-[2px]">
                      <div className="text-[#666]">Baseline Median (MAD)</div>
                      <div className="font-bold text-[#2A2A2A]">
                        {selectedEvent.baseline.median.toFixed(4)} (±{selectedEvent.baseline.mad.toFixed(4)})
                      </div>
                    </div>
                    <div className="p-2 bg-white border border-[#D1D0CB] rounded-[2px]">
                      <div className="text-[#666]">Signals Triggered</div>
                      <div className="font-bold text-[#4A5D4E] truncate">
                        {selectedEvent.signals_triggered.join(', ')}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Q7: What should the engineer do next? (Action Playbook Card) */}
                <div className="p-4 bg-emerald-50/50 border border-emerald-300/80 rounded-[2px] space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[11px] font-mono font-bold text-[#4A5D4E] uppercase">
                        7. Recommended Action & Playbook
                      </span>
                      <span className="px-2 py-0.5 bg-[#4A5D4E] text-white text-[10px] font-mono font-bold rounded-[2px]">
                        {selectedEvent.recommended_action}
                      </span>
                    </div>
                  </div>

                  <p className="text-xs text-[#2A2A2A] leading-relaxed">
                    {selectedEvent.action_detail}
                  </p>

                  <div className="flex flex-wrap gap-2 pt-1">
                    <button
                      onClick={() => handleTriggerAction(selectedEvent.recommended_action)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white text-xs font-bold rounded-[2px] shadow-xs cursor-pointer"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>Execute {selectedEvent.recommended_action}</span>
                    </button>
                    <button
                      onClick={() => handleTriggerAction('ROLLBACK_MODEL')}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] text-xs font-medium rounded-[2px] cursor-pointer"
                    >
                      <span>Rollback to Step {Math.max(0, selectedEvent.step - 200)}</span>
                    </button>
                    <button
                      onClick={() => handleTriggerAction('CHECK_HARDWARE')}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] text-xs font-medium rounded-[2px] cursor-pointer"
                    >
                      <span>Run DCGM Hardware Diag</span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: GEMINI FORENSIC INVESTIGATOR CHAT */}
            {activeTab === 'chat' && (
              <div className="space-y-4">
                <div className="space-y-4 max-h-[420px] overflow-y-auto pr-1">
                  {messages.map((msg, idx) => {
                    const isAI = msg.sender === 'gemini';
                    return (
                      <div
                        key={msg.id}
                        className={`flex gap-3 text-xs ${
                          isAI ? 'bg-[#F8F7F4] border border-[#D1D0CB]' : 'bg-[#F2F1ED] border border-[#D1D0CB]'
                        } p-4 rounded-[2px] relative group`}
                      >
                        <div className="w-6 h-6 rounded-[2px] bg-[#4A5D4E] text-white flex items-center justify-center shrink-0 mt-0.5">
                          {isAI ? <Bot className="w-3.5 h-3.5" /> : <User className="w-3.5 h-3.5" />}
                        </div>

                        <div className="flex-1 space-y-2 text-[#2A2A2A] leading-relaxed">
                          <div className="flex items-center justify-between text-[10px] font-mono text-[#666]">
                            <span className="font-bold uppercase">
                              {isAI ? `TrainGuard AI Diagnostic (${msg.modelUsed || 'gemini-3.7-flash'})` : 'Cluster Engineer'}
                            </span>
                            <span>{msg.timestamp}</span>
                          </div>

                          <div className="whitespace-pre-wrap font-sans text-xs">{msg.content}</div>
                        </div>

                        <button
                          onClick={() => handleCopyText(msg.content, idx)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 bg-white hover:bg-[#EBEAE5] text-[#666] border border-[#D1D0CB] rounded-[2px] absolute top-2 right-2 cursor-pointer"
                          title="Copy message"
                        >
                          {copiedIndex === idx ? <Check className="w-3 h-3 text-[#4A5D4E]" /> : <Copy className="w-3 h-3" />}
                        </button>
                      </div>
                    );
                  })}
                </div>

                {/* Chat Input Bar */}
                <div className="flex gap-2 pt-2 border-t border-[#D1D0CB]">
                  <input
                    type="text"
                    value={customPrompt}
                    onChange={(e) => setCustomPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleRunDiagnosis()}
                    placeholder="Ask Gemini to analyze parameter gradients, Slurm drain rules, or IEEE 754 bit-flips..."
                    className="flex-1 px-3 py-2 text-xs bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] focus:outline-none focus:border-[#4A5D4E]"
                  />
                  <button
                    onClick={handleRunDiagnosis}
                    disabled={loading}
                    className="px-4 py-2 bg-[#4A5D4E] hover:bg-[#3B4A3E] disabled:opacity-50 text-white text-xs font-bold rounded-[2px] flex items-center gap-1.5 shadow-xs cursor-pointer"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span>Send</span>
                  </button>
                </div>
              </div>
            )}

            {/* TAB 3: STANDARDIZED SCHEMA JSON (SECTION 11 SPEC) */}
            {activeTab === 'json' && (
              <div className="space-y-3">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-[#666]">TrainGuard Event Schema v1.0 (Section 11 Contract)</span>
                  <button
                    onClick={handleDownloadJsonReport}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-[2px] bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white text-xs font-mono font-bold cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download JSON Event</span>
                  </button>
                </div>

                <div className="bg-[#1C1C1A] text-emerald-300 font-mono text-xs p-4 rounded-[2px] border border-[#333330] max-h-[400px] overflow-y-auto leading-relaxed">
                  <pre>{JSON.stringify(selectedEvent, null, 2)}</pre>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Forensic Report Modal */}
      {showReportModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white border border-[#D1D0CB] rounded-[3px] max-w-3xl w-full max-h-[85vh] flex flex-col shadow-lg">
            <div className="p-4 border-b border-[#D1D0CB] flex items-center justify-between bg-[#F8F7F4]">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-[#4A5D4E]" />
                <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
                  TrainGuard Forensic Reliability Incident Report
                </h3>
              </div>
              <button
                onClick={() => setShowReportModal(false)}
                className="p-1 rounded-[2px] hover:bg-[#EBEAE5] text-[#666] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 overflow-y-auto flex-1 font-mono text-xs bg-[#1C1C1A] text-[#D1D0CB] whitespace-pre-wrap leading-relaxed">
              {generateReportMarkdown()}
            </div>

            <div className="p-4 border-t border-[#D1D0CB] flex items-center justify-between bg-[#F8F7F4]">
              <span className="text-[11px] font-mono text-[#666]">
                Incident {selectedEvent.event_id} • Status: CONFIRMED
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleCopyReport}
                  className="px-3 py-1.5 text-xs font-semibold bg-white hover:bg-[#EBEAE5] text-[#2A2A2A] border border-[#D1D0CB] rounded-[2px] flex items-center gap-1.5 cursor-pointer"
                >
                  {copiedReport ? <Check className="w-3.5 h-3.5 text-[#4A5D4E]" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedReport ? 'Copied' : 'Copy Markdown'}</span>
                </button>
                <button
                  onClick={handleDownloadReport}
                  className="px-4 py-1.5 text-xs font-bold bg-[#4A5D4E] hover:bg-[#3B4A3E] text-white rounded-[2px] flex items-center gap-1.5 shadow-xs cursor-pointer"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download .md</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

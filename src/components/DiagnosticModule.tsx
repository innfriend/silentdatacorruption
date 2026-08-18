import React, { useState } from 'react';
import { DiagnosticMessage, DiagnosticScenario } from '../types';
import { DIAGNOSTIC_SCENARIOS } from '../data/mockCluster';
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
} from 'lucide-react';

export const DiagnosticModule: React.FC = () => {
  const [scenarios] = useState<DiagnosticScenario[]>(DIAGNOSTIC_SCENARIOS);
  const [selectedScenario, setSelectedScenario] = useState<DiagnosticScenario>(DIAGNOSTIC_SCENARIOS[0]);
  const [customPrompt, setCustomPrompt] = useState<string>('');
  const [customLogs, setCustomLogs] = useState<string>(DIAGNOSTIC_SCENARIOS[0].logs);
  const [rankContext, setRankContext] = useState<string>('Rank 47 (Node dgx-hopper-06)');
  const [loading, setLoading] = useState<boolean>(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const [messages, setMessages] = useState<DiagnosticMessage[]>([
    {
      id: 'init-1',
      sender: 'gemini',
      content: `### 🔬 SilentGuard AI Root Cause Diagnostic Online
I am ready to perform hardware-level root cause diagnosis on Silent Data Corruption (SDC), ALU adder faults, dynamic voltage droops (L · di/dt), and Tensor Core anomalies.

Select a pre-configured cluster incident from the left or paste your custom hardware backtrace below to initiate silicon microarchitecture analysis.`,
      timestamp: '08:34:48',
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
      content: `Investigate incident on ${rankContext}:\n**Scenario**: ${selectedScenario.title}\n\n**Captured Logs**:\n\`\`\`\n${customLogs}\n\`\`\`\n\n${customPrompt || 'Analyze root cause and provide hardware mitigation steps.'}`,
      timestamp: new Date().toTimeString().split(' ')[0],
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const response = await fetch('/api/gemini/diagnose', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: customPrompt || 'Analyze root cause and provide hardware mitigation steps.',
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
    } catch (err: any) {
      console.error('Diagnosis failed:', err);
      const errorMessage: DiagnosticMessage = {
        id: `msg-${Date.now()}-err`,
        sender: 'gemini',
        content: `### ⚠️ Diagnostic Error
Unable to query backend: ${err.message}. Please verify local API connectivity.`,
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

  return (
    <div className="space-y-6">
      {/* Top Scenarios & Live Investigator Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Pre-configured Scenarios & Log Editor */}
        <div className="bg-white border border-[#D1D0CB] rounded-[3px] p-5 space-y-4 shadow-xs">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[#4A5D4E]" />
            <h3 className="text-xs font-bold text-[#2A2A2A] uppercase tracking-wider font-mono">
              Diagnostic Scenarios
            </h3>
          </div>

          {/* Scenario Selector */}
          <div className="space-y-2">
            {scenarios.map((sc) => {
              const isSelected = selectedScenario.id === sc.id;
              return (
                <button
                  key={sc.id}
                  onClick={() => handleSelectScenario(sc)}
                  className={`w-full text-left p-3 rounded-[2px] text-xs transition-all cursor-pointer ${
                    isSelected
                      ? 'bg-[#4A5D4E] text-white shadow-xs'
                      : 'bg-[#F8F7F4] text-[#2A2A2A] hover:bg-[#EBEAE5] border border-[#D1D0CB]'
                  }`}
                >
                  <div className="font-bold mb-1">{sc.title}</div>
                  <div className={`text-[11px] ${isSelected ? 'text-[#D7E4DA]' : 'text-[#666]'}`}>
                    Rank {sc.rank} • Node {sc.node} • Layer: {sc.layer}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Captured Logs Box */}
          <div>
            <label className="text-xs font-medium text-[#666] block mb-1">
              Hardware Backtrace & Telemetry Logs
            </label>
            <textarea
              rows={5}
              value={customLogs}
              onChange={(e) => setCustomLogs(e.target.value)}
              className="w-full bg-[#1C1C1A] text-emerald-300 font-mono text-[11px] p-2.5 rounded-[2px] border border-[#333330] focus:outline-none focus:border-[#4A5D4E]"
            />
          </div>

          {/* Quick Trigger Button */}
          <button
            onClick={handleRunDiagnosis}
            disabled={loading}
            className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 bg-[#4A5D4E] hover:bg-[#3B4A3E] disabled:opacity-50 text-white text-xs font-bold rounded-[2px] shadow-xs transition-colors cursor-pointer"
          >
            <Zap className="w-4 h-4" />
            <span>{loading ? 'AI Analyzing Silicon...' : 'Run Gemini Root Cause Diagnosis'}</span>
          </button>
        </div>

        {/* Right: AI Investigator Chat & Analysis Stream (2 Columns) */}
        <div className="lg:col-span-2 bg-white border border-[#D1D0CB] rounded-[3px] p-5 flex flex-col justify-between min-h-[500px] shadow-xs">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-[#D1D0CB] mb-4">
              <div className="flex items-center gap-2">
                <div className="w-7 h-7 rounded-[2px] bg-[#4A5D4E] text-white flex items-center justify-center">
                  <Sparkles className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-xs font-bold text-[#2A2A2A] font-mono uppercase tracking-wider">
                    GEMINI SDC MICROARCHITECTURE INVESTIGATOR
                  </h4>
                  <div className="text-[11px] text-[#666]">
                    Model: <span className="font-mono text-[#4A5D4E] font-semibold">gemini-3.7-flash</span> • Server-Side GenAI
                  </div>
                </div>
              </div>

              <button
                onClick={() =>
                  setMessages([
                    {
                      id: 'reset',
                      sender: 'gemini',
                      content: 'Conversation history reset. Ready for new hardware incident telemetry.',
                      timestamp: new Date().toTimeString().split(' ')[0],
                    },
                  ])
                }
                className="p-1.5 rounded-[2px] bg-[#F8F7F4] hover:bg-[#EBEAE5] text-[#666] border border-[#D1D0CB] cursor-pointer"
                title="Clear Chat"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Chat message stream */}
            <div className="space-y-4 max-h-[460px] overflow-y-auto pr-1">
              {messages.map((msg, idx) => {
                const isAI = msg.sender === 'gemini';
                return (
                  <div
                    key={msg.id}
                    className={`flex gap-3 text-xs ${
                      isAI ? 'bg-[#F8F7F4] border border-[#D1D0CB]' : 'bg-[#F2F1ED] border border-[#D1D0CB]'
                    } p-4 rounded-[2px] relative group`}
                  >
                    <div className="shrink-0 mt-0.5">
                      {isAI ? (
                        <div className="w-6 h-6 rounded-[2px] bg-[#4A5D4E] text-white flex items-center justify-center">
                          <Bot className="w-3.5 h-3.5" />
                        </div>
                      ) : (
                        <div className="w-6 h-6 rounded-[2px] bg-[#2A2A2A] text-white flex items-center justify-center">
                          <User className="w-3.5 h-3.5" />
                        </div>
                      )}
                    </div>

                    <div className="flex-1 space-y-2 overflow-x-auto">
                      <div className="flex items-center justify-between text-[10px] text-[#999] font-mono">
                        <span className="font-bold text-[#2A2A2A]">
                          {isAI ? 'SilentGuard Forensic Agent' : 'Cluster Operator'}
                        </span>
                        <span>{msg.timestamp}</span>
                      </div>

                      <div className="prose prose-xs max-w-none text-[#2A2A2A] leading-relaxed whitespace-pre-wrap font-sans">
                        {msg.content}
                      </div>

                      {msg.modelUsed && (
                        <div className="text-[10px] font-mono text-[#666] pt-2 border-t border-[#D1D0CB] flex justify-between items-center">
                          <span>Diagnostic Engine: {msg.modelUsed}</span>
                          <button
                            onClick={() => handleCopyText(msg.content, idx)}
                            className="inline-flex items-center gap-1 text-[#4A5D4E] hover:underline cursor-pointer"
                          >
                            {copiedIndex === idx ? (
                              <Check className="w-3 h-3 text-[#4A5D4E]" />
                            ) : (
                              <Copy className="w-3 h-3" />
                            )}
                            <span>{copiedIndex === idx ? 'Copied' : 'Copy Diagnosis'}</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {loading && (
                <div className="flex gap-3 text-xs bg-[#F8F7F4] border border-[#D1D0CB] p-4 rounded-[2px] animate-pulse">
                  <div className="w-6 h-6 rounded-[2px] bg-[#4A5D4E] text-white flex items-center justify-center">
                    <Sparkles className="w-3.5 h-3.5 animate-spin" />
                  </div>
                  <div className="space-y-2 flex-1">
                    <div className="h-3 bg-[#D1D0CB] rounded w-1/3"></div>
                    <div className="h-3 bg-[#D1D0CB] rounded w-3/4"></div>
                    <div className="h-3 bg-[#D1D0CB] rounded w-1/2"></div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* User Prompt Input Box */}
          <div className="mt-4 pt-3 border-t border-[#D1D0CB] flex gap-2">
            <input
              type="text"
              placeholder="Ask custom question (e.g. 'What is the transistor failure mechanism on SM 34?')..."
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !loading) handleRunDiagnosis();
              }}
              className="flex-1 bg-[#F8F7F4] border border-[#D1D0CB] rounded-[2px] px-3.5 py-2 text-xs text-[#2A2A2A] placeholder-[#999] focus:outline-none focus:border-[#4A5D4E]"
            />
            <button
              onClick={handleRunDiagnosis}
              disabled={loading}
              className="px-4 py-2 bg-[#4A5D4E] hover:bg-[#3B4A3E] disabled:opacity-50 text-white rounded-[2px] text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Send className="w-3.5 h-3.5" />
              <span>Send</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

import { useState, useEffect } from 'react';
import {
  Brain,
  History,
  CheckCircle2,
  XCircle,
  Zap,
  Eye,
  MessageSquare,
  ShieldCheck,
  RefreshCw,
  Database,
  Upload,
  FileJson,
  ArrowRight
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// --- Constants ---
const ARC_COLORS = [
  '#000000', // 0: black
  '#0074D9', // 1: blue
  '#FF4136', // 2: red
  '#2ECC40', // 3: green
  '#FFDC00', // 4: yellow
  '#AAAAAA', // 5: gray
  '#F012BE', // 6: magenta
  '#FF851B', // 7: orange
  '#7FDBFF', // 8: azure
  '#870C25'  // 9: maroon
];

// --- Utility ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface Rule {
  text: string;
  weight: number;
  success_count: number;
  last_used: string;
}

interface Hypothesis {
  hypothesis_id: string;
  content: string;
  score: number;
  evidence: string[];
}

// --- Components ---

const GridVisualizer = ({ grid, onPixelClick }: { grid: number[][], onPixelClick?: (r: number, c: number) => void }) => {
  if (!grid || !grid.length) return null;
  const rows = grid.length;
  const cols = grid[0].length;

  return (
    <div
      className={cn(
        "w-full h-full flex items-center justify-center p-4",
        onPixelClick && "cursor-crosshair"
      )}
    >
      <div
        className="grid gap-[1px] bg-white/10 border border-white/20 shadow-2xl"
        style={{
          gridTemplateColumns: `repeat(${cols}, minmax(1px, 1fr))`,
          aspectRatio: `${cols}/${rows}`,
          width: 'min(100%, 80vh * ' + (cols / rows) + ')',
          maxHeight: '70vh'
        }}
      >
        {grid.map((row, r) =>
          row.map((colorIdx, c) => (
            <div
              key={`${r}-${c}`}
              onClick={() => onPixelClick?.(r, c)}
              className={cn(
                "w-full aspect-square relative group",
                onPixelClick && "hover:brightness-150 transition-all cursor-pointer"
              )}
              style={{
                backgroundColor: ARC_COLORS[colorIdx] || '#000',
              }}
            >
              {onPixelClick && (
                <div className="absolute inset-x-0 inset-y-0 opacity-0 group-hover:opacity-20 bg-white transition-opacity pointer-events-none" />
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

const StatusBadge = ({ label, type }: { label: string, type: 'success' | 'warning' | 'info' }) => (
  <span className={cn(
    "px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
    type === 'success' && "bg-accent/10 text-accent border border-accent/20",
    type === 'warning' && "bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",
    type === 'info' && "bg-primary/10 text-primary border border-primary/20",
  )}>
    {label}
  </span>
);

const PredictionCard = ({ title, grid, imageUrl, onPixelClick }: { title: string, grid?: number[][], imageUrl?: string, onPixelClick?: (r: number, c: number) => void }) => (
  <div className="bg-[#1a1a1a] rounded-2xl border border-white/5 p-6 flex flex-col gap-4 group hover:border-white/10 transition-all shadow-2xl h-full">
    <div className="flex justify-between items-center shrink-0">
      <div className="flex flex-col gap-1">
        <h3 className="text-base font-bold text-white/90">{title}</h3>
        {grid && (
          <span className="text-[11px] text-primary/60 font-mono font-bold uppercase tracking-widest">
            {grid[0].length} × {grid.length} RESOLUTION
          </span>
        )}
      </div>
      <Eye className="w-5 h-5 text-muted-foreground/50 group-hover:text-primary transition-colors cursor-pointer" />
    </div>
    <div className="flex-1 w-full rounded-xl bg-black/60 border border-white/5 flex items-center justify-center relative shadow-inner overflow-hidden">
      {grid ? (
        <GridVisualizer grid={grid} onPixelClick={onPixelClick} />
      ) : imageUrl ? (
        <img src={`http://localhost:8000${imageUrl}`} alt={title} className="w-full h-full object-contain" />
      ) : (
        <div className="flex flex-col items-center gap-2 text-muted-foreground/30">
          <RefreshCw className="w-10 h-10 animate-spin-slow" />
          <span className="text-xs font-mono">Awaiting Matrix...</span>
        </div>
      )}
    </div>
  </div>
);

export default function App() {
  const [activeTab, setActiveTab] = useState<'reasoning' | 'memory'>('reasoning');
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [rules, setRules] = useState<Rule[]>([]);
  const [knowledge, setKnowledge] = useState('');
  const [jsonInput, setJsonInput] = useState('');
  const [puzzleList, setPuzzleList] = useState<string[]>([]);
  const [selectedPuzzle, setSelectedPuzzle] = useState('');
  const [selectedColor, setSelectedColor] = useState(1);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [zoomedExample, setZoomedExample] = useState<any>(null);
  const [humanGrid, setHumanGrid] = useState<number[][] | null>(null);

  const [reasoningState, setReasoningState] = useState<any>(null);

  // Data fetching
  useEffect(() => {
    const fetchData = async () => {
      try {
        const memRes = await fetch('http://localhost:8000/api/memory');
        const memData = await memRes.json();
        setRules(memData.rules || []);

        const stateRes = await fetch('http://localhost:8000/api/state');
        const stateData = await stateRes.json();
        setHypotheses(stateData.hypotheses || []);
        setReasoningState(stateData);

        const puzzlesRes = await fetch('http://localhost:8000/api/list_puzzles');
        const puzzlesData = await puzzlesRes.json();
        setPuzzleList(puzzlesData.puzzles || []);
      } catch (err) {
        console.error("Failed to fetch backend data", err);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1000); // Poll every 1s for snappier updates
    return () => clearInterval(interval);
  }, []);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsModalOpen(false);
        setZoomedExample(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handlePuzzleSelect = async (filename: string) => {
    setSelectedPuzzle(filename);
    if (!filename) return;
    try {
      const res = await fetch(`http://localhost:8000/api/puzzle/${filename}`);
      const data = await res.json();
      setJsonInput(JSON.stringify(data, null, 2));

      // Auto-upload when selecting from dropdown
      await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      console.log("Auto-loaded puzzle:", filename);
    } catch (err) {
      console.error("Failed to fetch or upload puzzle content", err);
    }
  };

  const handlePredict = async () => {
    try {
      await fetch('http://localhost:8000/api/predict', { method: 'POST' });
    } catch (err) { console.error(err); }
  };

  const handleUpload = async () => {
    try {
      const task = JSON.parse(jsonInput);
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(task)
      });
      if (res.ok) {
        // alert("Task uploaded successfully!");
        setActiveTab('reasoning');
      }
    } catch (err) {
      alert("Invalid JSON format. Please check the ARC task structure.");
    }
  };

  const setStep = async (s: number) => {
    try {
      await fetch(`http://localhost:8000/api/set_step?step=${s}`, { method: 'POST' });
    } catch (err) { console.error(err); }
  };

  const handlePixelClick = async (r: number, c: number) => {
    // Allow editing at any time after task is loaded
    if (reasoningState.current_step < 2) return;
    try {
      await fetch('http://localhost:8000/api/update_test_grid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ r, c, color: selectedColor })
      });
    } catch (err) { console.error(err); }
  };

  const handleInject = async () => {
    try {
      await fetch('http://localhost:8000/api/inject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: knowledge, human_grid: humanGrid })
      });
      setKnowledge('');
      alert("Knowledge Injected! Swarm will prioritize this in the next run.");
    } catch (err) {
      alert("Injection failed. Is the server running?");
    }
  };

  return (
    <div className="min-h-screen w-full flex bg-background text-foreground overflow-hidden font-sans selection:bg-primary/30">

      {/* Sidebar */}
      <nav className="w-64 border-r border-white/5 flex flex-col p-6 gap-8 bg-[#0d0d0d]">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">Brain<span className="text-primary italic">v3</span></span>
        </div>

        <div className="flex flex-col gap-1">
          <button
            onClick={() => setActiveTab('reasoning')}
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition-all",
              activeTab === 'reasoning' ? "bg-primary/10 text-primary font-medium" : "hover:bg-white/5 text-muted-foreground"
            )}
          >
            <Zap className="w-4 h-4" /> Reasoning Engine
          </button>
          <button
            onClick={() => setActiveTab('memory')}
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-xl text-sm transition-all",
              activeTab === 'memory' ? "bg-primary/10 text-primary font-medium" : "hover:bg-white/5 text-muted-foreground"
            )}
          >
            <Database className="w-4 h-4" /> Rule Memory
          </button>
        </div>

        {activeTab === 'reasoning' && (
          <div className="mt-4 flex flex-col gap-4">
            <div className="px-4 py-1 flex items-center gap-2">
              <div className="w-1 h-3 bg-primary rounded-full" />
              <span className="text-[10px] font-black uppercase tracking-widest text-white/40">Mutation Tools</span>
            </div>
            <div className="px-4">
              <div className="grid grid-cols-5 gap-2 bg-black/40 p-3 rounded-2xl border border-white/5">
                {ARC_COLORS.map((c, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedColor(i)}
                    className={cn(
                      "w-full aspect-square rounded-md border-2 transition-all",
                      selectedColor === i ? "border-white scale-110 shadow-lg" : "border-transparent opacity-40 hover:opacity-100"
                    )}
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
              <div className="mt-3 p-3 bg-primary/5 border border-primary/20 rounded-xl">
                <p className="text-[9px] text-primary/80 leading-tight">
                  Select color and click on Challenge Grid pixels to mutate.
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="mt-auto border-t border-white/5 pt-6">
          <div className="bg-secondary/50 rounded-xl p-4 border border-white/5">
            <div className="flex items-center gap-2 mb-2">
              <ShieldCheck className="w-3 h-3 text-accent" />
              <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-widest">System Status</span>
            </div>
            <div className="text-xs font-mono text-muted-foreground/80 leading-relaxed">
              Consensus: <span className="text-accent underline">Active</span><br />
              Agents: 32 Parallel<br />
              Bias: Memory Driven
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-screen relative">
        <header className="h-20 border-b border-white/5 px-8 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div className="flex flex-col">
              <h1 className="text-xl font-bold">HITL Dashboard</h1>
              <span className="text-[10px] text-muted-foreground uppercase font-black tracking-widest">ARC Phase 03</span>
            </div>

            <div className="flex items-center gap-3 bg-white/5 p-1.5 rounded-xl border border-white/5">
              <select
                value={selectedPuzzle}
                onChange={(e) => {
                  handlePuzzleSelect(e.target.value);
                }}
                className="bg-transparent text-sm focus:outline-none px-3 py-1 cursor-pointer font-medium"
              >
                <option value="" className="bg-black">Select Puzzle...</option>
                {puzzleList.map(p => (
                  <option key={p} value={p} className="bg-black">{p}</option>
                ))}
              </select>
              <button
                onClick={() => setIsModalOpen(true)}
                className="px-3 py-1.5 bg-white/5 hover:bg-white/10 text-muted-foreground hover:text-white rounded-lg text-xs font-bold transition-all flex items-center gap-2"
              >
                <FileJson className="w-3 h-3" /> Paste JSON
              </button>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className={`px-4 py-1.5 rounded-full border text-[10px] font-black uppercase tracking-widest ${reasoningState?.current_step >= 3 ? 'border-accent/20 bg-accent/5 text-accent' : 'border-white/10 text-white/20'
              }`}>
              Step {reasoningState?.current_step || 1} / 5
            </div>
          </div>
        </header>

        <div className="flex-1 p-8 overflow-y-auto">
          {activeTab === 'reasoning' && (
            <div className="flex flex-col gap-6 h-full min-h-0 overflow-hidden">
              {/* Top Bar: Demos (Compact) */}
              <section className="shrink-0 bg-black/40 border border-white/5 rounded-2xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <h2 className="text-xs font-black uppercase tracking-[0.2em] text-primary flex items-center gap-2">
                    <History className="w-4 h-4" /> Training Set
                  </h2>
                  <div className="flex items-center gap-4 text-[10px] text-muted-foreground font-bold">
                    <span>{reasoningState?.train?.length || 0} EXAMPLES</span>
                    <span className="h-3 w-[1px] bg-white/10" />
                    <span>SCROLL HORIZONTALLY →</span>
                  </div>
                </div>
                <div className="flex gap-4 overflow-x-auto pb-2 scrollbar-hide">
                  {reasoningState?.train?.map((pair: any, idx: number) => (
                    <div
                      key={idx}
                      onClick={() => setZoomedExample({ ...pair, title: `Training Example ${idx + 1}` })}
                      className="flex items-center gap-3 p-2 rounded-xl bg-white/5 border border-white/5 hover:bg-white/10 transition-all shrink-0 cursor-zoom-in group"
                    >
                      <div className="w-24 h-24 bg-black/40 rounded flex items-center justify-center p-1 group-hover:bg-black/60 transition-all">
                        <GridVisualizer grid={pair.input} />
                      </div>
                      <ArrowRight className="w-3 h-3 text-muted-foreground" />
                      <div className="w-24 h-24 bg-black/40 rounded flex items-center justify-center p-1 group-hover:bg-black/60 transition-all">
                        <GridVisualizer grid={pair.output} />
                      </div>
                      <div className="px-2 border-l border-white/5">
                        <span className="text-[10px] font-black text-white/30 uppercase vertical-text">EX {idx + 1}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Middle: Main Reasoning & Action Area */}
              <div className="flex-1 grid grid-cols-12 gap-6 min-h-0 overflow-hidden">
                <section className="col-span-12 lg:col-span-9 flex flex-col gap-6 min-h-0">
                  <div className="flex-1 grid grid-cols-2 gap-6 min-h-0">
                    <PredictionCard
                      title="Challenge Input (Editable)"
                      grid={reasoningState?.test_input}
                      onPixelClick={reasoningState?.current_step >= 2 ? handlePixelClick : undefined}
                    />
                    {reasoningState?.current_step >= 3 ? (
                      <PredictionCard
                        title="Swarm Logic Prediction"
                        grid={reasoningState?.predicted_grid}
                      />
                    ) : (
                      <div className="relative flex flex-col items-center justify-center bg-[#0a0a0a] rounded-2xl border-2 border-dashed border-white/5 group transition-all hover:border-primary/20">
                        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                        <div className="w-20 h-20 rounded-full bg-primary/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                          <Zap className="w-10 h-10 text-primary/20 group-hover:text-primary transition-colors" />
                        </div>
                        <span className="text-sm font-black uppercase tracking-[0.3em] text-white/10 group-hover:text-white/30 transition-colors">
                          Awaiting Prediction
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Context-Aware Action Button */}
                  <div className="shrink-0">
                    {reasoningState?.current_step === 2 && (
                      <button
                        onClick={handlePredict}
                        className="w-full py-6 bg-primary text-white rounded-2xl font-black flex items-center justify-center gap-4 hover:scale-[1.01] active:scale-95 transition-all shadow-[0_0_50px_-12px_rgba(var(--primary),0.5)] text-xl uppercase tracking-[0.2em] group"
                      >
                        <Zap className="w-8 h-8 group-hover:animate-pulse" /> Invoke Swarm Reasoning
                      </button>
                    )}

                    {reasoningState?.current_step >= 3 && (
                      <div className="flex gap-4">
                        <button
                          onClick={handlePredict}
                          className="flex-1 py-4 bg-primary/20 text-primary border-2 border-primary rounded-xl font-black flex items-center justify-center gap-2 hover:bg-primary hover:text-white transition-all text-sm uppercase tracking-[0.15em]"
                        >
                          <Zap className="w-5 h-5" /> Re-Run Prediction
                        </button>
                        <button
                          onClick={() => setStep(4)}
                          className="flex-1 py-4 bg-white text-black rounded-xl font-black flex items-center justify-center gap-2 hover:scale-[1.01] active:scale-95 transition-all text-sm uppercase tracking-[0.15em]"
                        >
                          Human Feedback
                        </button>
                      </div>
                    )}

                    {reasoningState?.current_step === 5 && (
                      <div className="p-6 rounded-2xl bg-[#151515] border border-white/10 shadow-2xl flex items-center gap-10 animate-in slide-in-from-bottom-2">
                        <div className="flex flex-col gap-4">
                          <span className="text-[10px] font-black uppercase tracking-widest text-primary italic">Active Brush</span>
                          <div className="grid grid-cols-5 gap-2">
                            {ARC_COLORS.map((c, i) => (
                              <button
                                key={i}
                                onClick={() => setSelectedColor(i)}
                                className={cn(
                                  "w-10 h-10 rounded-xl border-4 transition-all relative overflow-hidden",
                                  selectedColor === i ? "border-white scale-110 shadow-[0_0_20px_rgba(255,255,255,0.3)]" : "border-transparent hover:scale-105 opacity-50 hover:opacity-100"
                                )}
                                style={{ backgroundColor: c }}
                              >
                                {selectedColor === i && <div className="absolute inset-0 flex items-center justify-center"><CheckCircle2 className="w-5 h-5 text-white mix-blend-difference" /></div>}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="h-20 w-[1px] bg-white/10" />
                        <div className="flex-1">
                          <h3 className="text-lg font-black text-white uppercase tracking-tighter mb-1">Verify Persistence</h3>
                          <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                            Manual grid mutation active. Modify the test input to verify if the Swarm's rule set remains robust under drift.
                          </p>
                          <button
                            onClick={handlePredict}
                            className="px-8 py-3 bg-accent text-accent-foreground rounded-xl font-black uppercase tracking-widest hover:brightness-110 transition-all"
                          >
                            Push Verification Re-run
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </section>

                {/* Right: Heuristics & Hypotheses */}
                <section className="col-span-12 lg:col-span-3 flex flex-col gap-6 min-h-0">
                  <div className="bg-[#151515] rounded-2xl border border-white/5 p-6 flex flex-col gap-5 shrink-0 shadow-2xl">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-accent/20 flex items-center justify-center">
                        <MessageSquare className="w-5 h-5 text-accent" />
                      </div>
                      <h2 className="text-sm font-black uppercase tracking-widest">Feedback</h2>
                    </div>

                    <textarea
                      value={knowledge}
                      disabled={reasoningState?.current_step < 4}
                      onChange={(e) => setKnowledge(e.target.value)}
                      placeholder={reasoningState?.current_step < 4 ? "LOCKED: Await Step 4" : "Describe the transformation rule..."}
                      className={cn(
                        "w-full h-40 bg-black/60 border border-white/10 rounded-xl p-4 text-xs font-mono focus:outline-none focus:border-primary/50 transition-all resize-none shadow-inner leading-relaxed",
                        reasoningState?.current_step < 4 ? "opacity-20 cursor-not-allowed grayscale" : "opacity-100 border-primary/20 ring-1 ring-primary/10"
                      )}
                    />

                    {/* Step 4: Human Visual Solution Input */}
                    {reasoningState?.current_step === 4 && (
                      <div className="flex flex-col gap-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-black uppercase tracking-widest text-primary italic">Draw Correct Solution (Optional)</span>
                          <button
                            onClick={() => setHumanGrid(JSON.parse(JSON.stringify(reasoningState?.test_input || [])))}
                            className="text-[9px] text-muted-foreground underline hover:text-white"
                          >
                            Reset to Input
                          </button>
                        </div>
                        <div className="h-48 bg-black/40 rounded-xl border border-white/10 overflow-hidden relative">
                          <GridVisualizer
                            grid={humanGrid || reasoningState?.test_input || []}
                            onPixelClick={(r, c) => {
                              if (!humanGrid && reasoningState?.test_input) {
                                // Initialize on first click
                                const newGrid = JSON.parse(JSON.stringify(reasoningState.test_input));
                                newGrid[r][c] = selectedColor;
                                setHumanGrid(newGrid);
                              } else if (humanGrid) {
                                const newGrid = [...humanGrid];
                                newGrid[r] = [...newGrid[r]];
                                newGrid[r][c] = selectedColor;
                                setHumanGrid(newGrid);
                              }
                            }}
                          />
                        </div>
                      </div>
                    )}

                    <div className="flex gap-3">
                      <button
                        onClick={async () => {
                          await handleInject();
                          await setStep(5);
                        }}
                        disabled={reasoningState?.current_step !== 4}
                        className={cn(
                          "flex-1 py-4 bg-accent text-accent-foreground rounded-xl text-[10px] font-black uppercase tracking-[0.2em] transition-all shadow-xl active:scale-95",
                          reasoningState?.current_step !== 4 && "opacity-20 grayscale"
                        )}
                      >
                        Commit Advice
                      </button>
                    </div>
                  </div>

                  <div className="flex-1 bg-[#151515] rounded-2xl border border-white/5 p-5 flex flex-col gap-5 overflow-hidden shadow-2xl">
                    <div className="flex items-center justify-between shrink-0">
                      <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-white/40 flex items-center gap-2">
                        <History className="w-3 h-3" /> Live Hypotheses
                      </h3>
                      <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                    </div>
                    <div className="flex-1 flex flex-col gap-3 overflow-y-auto pr-2 scrollbar-thin">
                      {hypotheses.filter(h => !!h).map((h, i) => (
                        <div key={h.hypothesis_id || `h-${i}`} className="p-4 rounded-xl bg-black/40 border border-white/5 flex flex-col gap-3 hover:border-white/20 transition-all">
                          <p className="text-[11px] font-medium text-white/70 leading-relaxed font-mono">{h.content}</p>
                          <div className="flex justify-between items-center pt-2 border-t border-white/5">
                            <span className="text-[9px] font-black text-white/20 uppercase tracking-widest">{h.hypothesis_id}</span>
                            <div className="flex items-center gap-2">
                              <div className="w-12 h-1 bg-white/5 rounded-full overflow-hidden">
                                <div className="h-full bg-accent" style={{ width: `${h.score * 100}%` }} />
                              </div>
                              <span className="text-[10px] font-black text-accent">{(h.score * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              </div>
            </div>
          )}

          {activeTab === 'memory' && (
            <div className="max-w-6xl mx-auto flex flex-col gap-8 fade-in h-full overflow-y-auto pb-20">
              <div className="flex flex-col gap-2">
                <h2 className="text-3xl font-black uppercase tracking-tighter">Global Rule Repository</h2>
                <p className="text-muted-foreground font-medium">Verified cross-task abstractions persisted in long-term weights.</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {rules.map((rule, i) => (
                  <div key={rule.text || i} className="p-6 rounded-2xl bg-[#1a1a1a] border border-white/5 flex flex-col gap-4 hover:border-primary/30 transition-all group">
                    <p className="text-sm font-bold text-white/90 leading-relaxed">{rule.text}</p>
                    <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-widest pt-4 border-t border-white/5">
                      <div className="flex flex-col gap-1">
                        <span className="text-muted-foreground/50">Success Count</span>
                        <span className="text-white">{rule.success_count}</span>
                      </div>
                      <div className="flex flex-col gap-1 text-right">
                        <span className="text-accent/50 group-hover:text-accent">Weight Bias</span>
                        <span className="text-accent text-lg">{(rule.weight * 10).toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* JSON Upload Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-black/80 backdrop-blur-sm animate-in fade-in">
          <div className="w-full max-w-4xl bg-[#1a1a1a] border border-white/10 rounded-3xl p-8 flex flex-col gap-6 shadow-2xl">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
                  <FileJson className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h2 className="text-xl font-black uppercase tracking-tighter">Native ARC JSON Load</h2>
                  <p className="text-xs text-muted-foreground">Inject raw dataset objects for instantaneous reasoning.</p>
                </div>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="p-2 hover:bg-white/5 rounded-full transition-all"
              >
                <XCircle className="w-6 h-6 text-muted-foreground" />
              </button>
            </div>

            <textarea
              value={jsonInput}
              onChange={(e) => setJsonInput(e.target.value)}
              placeholder='Paste ARC-style JSON task...'
              className="w-full h-96 bg-black/60 border border-white/10 rounded-2xl p-6 text-xs font-mono focus:outline-none focus:border-primary/50 transition-all resize-none shadow-inner"
            />

            <div className="flex gap-4">
              <button
                onClick={() => setIsModalOpen(false)}
                className="flex-1 py-4 bg-white/5 hover:bg-white/10 text-white rounded-xl font-bold transition-all"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  await handleUpload();
                  setIsModalOpen(false);
                }}
                className="flex-3 py-4 bg-primary text-white rounded-xl font-black uppercase tracking-[0.2em] transition-all shadow-xl shadow-primary/20"
              >
                Bootstrap Task
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Zoom Modal */}
      {zoomedExample && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-12 bg-black/95 backdrop-blur-md animate-in zoom-in-95 duration-200">
          <div className="w-full max-w-7xl h-full flex flex-col gap-8">
            <div className="flex justify-between items-center bg-white/5 p-6 rounded-3xl border border-white/10 shrink-0">
              <div className="flex flex-col gap-1">
                <h2 className="text-2xl font-black uppercase tracking-tighter text-primary">{zoomedExample.title}</h2>
                <span className="text-xs text-muted-foreground uppercase font-bold tracking-[0.3em]">Comparison View</span>
              </div>
              <button
                onClick={() => setZoomedExample(null)}
                className="p-3 bg-white/5 hover:bg-white/10 rounded-full transition-all border border-white/10 group"
              >
                <XCircle className="w-8 h-8 text-muted-foreground group-hover:text-white" />
              </button>
            </div>

            <div className="flex-1 grid grid-cols-2 gap-12 min-h-0">
              <div className="flex flex-col gap-4 min-h-0 uppercase">
                <span className="text-xs font-black tracking-widest text-white/40 italic">Demonstration Input</span>
                <div className="flex-1 bg-black/40 rounded-3xl border border-white/5 overflow-hidden shadow-inner">
                  <GridVisualizer grid={zoomedExample.input} />
                </div>
              </div>
              <div className="flex flex-col gap-4 min-h-0 uppercase">
                <span className="text-xs font-black tracking-widest text-primary italic">Demonstration Output</span>
                <div className="flex-1 bg-black/40 rounded-3xl border border-white/5 overflow-hidden shadow-inner">
                  <GridVisualizer grid={zoomedExample.output} />
                </div>
              </div>
            </div>

            <div className="shrink-0 flex justify-center pb-4">
              <button
                onClick={() => setZoomedExample(null)}
                className="px-12 py-4 bg-white text-black rounded-2xl font-black uppercase tracking-widest hover:scale-105 active:scale-95 transition-all shadow-2xl"
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

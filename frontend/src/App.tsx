import { useEffect, useState } from "react";
import { ThemeProvider } from "next-themes";
import { ThemeToggle } from "./components/ThemeToggle";
import { OverviewTab } from "./components/OverviewTab";
import { ConversationsTab } from "./components/ConversationsTab";
import { fetchVersion } from "./lib/api";
import { LayoutDashboard, MessageSquare } from "lucide-react";

type Tab = "overview" | "conversations";

function Dashboard() {
  const [tab, setTab] = useState<Tab>("overview");
  const [version, setVersion] = useState("");

  useEffect(() => {
    fetchVersion().then(setVersion);
  }, []);

  return (
    <div className="mx-auto min-h-screen max-w-7xl px-4 py-6">
      <header className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <img src="/logo.png" alt="GPT-RAG" className="h-10 w-10" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">GPT-RAG Orchestrator</h1>
            {version && (
              <span className="text-xs text-muted-foreground">v{version}</span>
            )}
          </div>
        </div>
        <ThemeToggle />
      </header>

      <nav className="mb-4 flex gap-1 border-b">
        <button
          onClick={() => setTab("overview")}
          className={`flex items-center gap-1.5 border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
            tab === "overview"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          <LayoutDashboard className="h-4 w-4" />
          Overview
        </button>
        <button
          onClick={() => setTab("conversations")}
          className={`flex items-center gap-1.5 border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
            tab === "conversations"
              ? "border-primary text-foreground"
              : "border-transparent text-muted-foreground hover:text-foreground"
          }`}
        >
          <MessageSquare className="h-4 w-4" />
          Conversations
        </button>
      </nav>

      {tab === "overview" ? <OverviewTab /> : <ConversationsTab />}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem>
      <Dashboard />
    </ThemeProvider>
  );
}

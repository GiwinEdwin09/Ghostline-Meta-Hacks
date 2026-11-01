"use client";

import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";

type Status = "checking" | "online" | "offline";

const LABELS: Record<Status, string> = {
  checking: "Checking the back end…",
  online: "Back end online",
  offline: "Back end offline",
};

export function ApiStatus() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    getHealth().then(
      () => setStatus("online"),
      () => setStatus("offline"),
    );
  }, []);

  return (
    <p className="eyebrow inline-flex items-center gap-2" role="status">
      <span
        aria-hidden
        className={`inline-block size-2 rounded-full ${
          status === "online" ? "bg-emerald-400" : status === "offline" ? "bg-red-400" : "bg-muted"
        }`}
        style={status === "checking" ? { animation: "ghost-pulse 1.4s ease-in-out infinite" } : undefined}
      />
      {LABELS[status]}
    </p>
  );
}

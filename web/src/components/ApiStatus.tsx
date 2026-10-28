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
    <p className="text-sm text-zinc-400">
      <span
        className={`mr-2 inline-block size-2 rounded-full ${
          status === "online" ? "bg-cyan-400" : status === "offline" ? "bg-red-500" : "bg-zinc-500"
        }`}
      />
      {LABELS[status]}
    </p>
  );
}

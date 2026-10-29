"use client";

import { useCallback, useEffect, useEffectEvent, useReducer, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { Experiment, ExperimentStatus, Job, NotebookEvent } from "@/lib/types";

/* ------------------------------------------------------------------------------------ */
/* A small fetch-and-poll hook                                                           */
/* ------------------------------------------------------------------------------------ */

export interface Resource<T> {
  data: T | undefined;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

interface ResourceState<T> {
  key: string;
  data: T | undefined;
  error: unknown;
  settled: boolean;
}

/**
 * Load a resource and, optionally, keep it fresh. `key` identifies what is loaded: when it
 * changes the old data is dropped immediately instead of being shown against the new key.
 * Polling uses a chained timeout (never two requests at once), pauses while the tab is in
 * the background and refreshes the moment it comes back, which is what mobile Safari needs.
 */
export function useResource<T>(
  key: string,
  load: (signal: AbortSignal) => Promise<T>,
  options: { intervalMs?: number | null; enabled?: boolean } = {},
): Resource<T> {
  const { intervalMs = null, enabled = true } = options;
  const [state, setState] = useState<ResourceState<T>>({ key, data: undefined, error: null, settled: false });
  const [nonce, setNonce] = useState(0);
  const run = useEffectEvent(load);

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();

    const tick = async () => {
      if (!alive) return;
      if (typeof document !== "undefined" && document.visibilityState === "hidden") {
        // Try again when the tab is visible; the visibilitychange handler below does it.
        return;
      }
      try {
        const data = await run(controller.signal);
        if (alive) setState({ key, data, error: null, settled: true });
      } catch (error) {
        if (alive && !(error instanceof DOMException && error.name === "AbortError")) {
          setState((s) => ({ key, data: s.key === key ? s.data : undefined, error, settled: true }));
        }
      }
      if (alive && intervalMs) timer = setTimeout(tick, intervalMs);
    };

    const onVisible = () => {
      if (document.visibilityState === "visible") {
        if (timer) clearTimeout(timer);
        void tick();
      }
    };

    void tick();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      controller.abort();
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [key, nonce, enabled, intervalMs]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const current = state.key === key ? state : null;
  return {
    data: current?.data,
    error: current?.error ?? null,
    loading: enabled && !(current?.settled ?? false),
    reload,
  };
}

/* ------------------------------------------------------------------------------------ */
/* A live experiment: the record, its notebook events and its running job                */
/* ------------------------------------------------------------------------------------ */

export interface LiveExperiment {
  experiment: Experiment | undefined;
  events: NotebookEvent[];
  job: Job | null;
  status: ExperimentStatus | undefined;
  error: unknown;
  loading: boolean;
  /** True while a job is queued or running (polling every second). */
  active: boolean;
  refresh: () => void;
}

interface LiveState {
  id: string;
  experiment: Experiment | undefined;
  events: NotebookEvent[];
  job: Job | null;
  status: ExperimentStatus | undefined;
  error: unknown;
  settled: boolean;
}

type LiveAction =
  | { type: "loaded"; id: string; experiment: Experiment; events: NotebookEvent[] }
  | { type: "events"; id: string; events: NotebookEvent[]; job: Job | null; status: ExperimentStatus }
  | { type: "experiment"; id: string; experiment: Experiment }
  | { type: "error"; id: string; error: unknown };

function emptyLive(id: string): LiveState {
  return { id, experiment: undefined, events: [], job: null, status: undefined, error: null, settled: false };
}

function mergeEvents(existing: NotebookEvent[], incoming: NotebookEvent[]): NotebookEvent[] {
  if (incoming.length === 0) return existing;
  const seen = new Set(existing.map((e) => e.seq));
  const added = incoming.filter((e) => !seen.has(e.seq));
  if (added.length === 0) return existing;
  return [...existing, ...added].sort((a, b) => a.seq - b.seq);
}

function liveReducer(state: LiveState, action: LiveAction): LiveState {
  // A fresh load always wins; anything else for a different experiment is stale.
  if (action.type !== "loaded" && action.id !== state.id) return state;
  switch (action.type) {
    case "loaded":
      return {
        ...state,
        id: action.id,
        experiment: action.experiment,
        events: action.events,
        job: action.experiment.job,
        status: action.experiment.status,
        error: null,
        settled: true,
      };
    case "events":
      return {
        ...state,
        events: mergeEvents(state.events, action.events),
        job: action.job,
        status: action.status,
        error: null,
      };
    case "experiment":
      return {
        ...state,
        experiment: action.experiment,
        job: action.experiment.job,
        status: action.experiment.status,
        error: null,
      };
    case "error":
      return { ...state, error: action.error, settled: true };
  }
}

export function isJobActive(job: Job | null | undefined, status?: ExperimentStatus): boolean {
  if (job) return job.status === "queued" || job.status === "running";
  return status === "running" || status === "processing";
}

/**
 * Load an experiment and follow it: while a job is running, poll `GET /events?after=seq`
 * every second (the plan's polling design, Part 0 decision 7); when the job ends, reload
 * the full record so models, fits, review and ghosts appear.
 */
export function useExperimentLive(id: string): LiveExperiment {
  const [state, dispatch] = useReducer(liveReducer, id, emptyLive);
  const [nonce, setNonce] = useState(0);
  const lastSeq = useRef(0);
  const live = state.id === id ? state : emptyLive(id);

  useEffect(() => {
    let alive = true;
    Promise.all([api.getExperiment(id), api.getEvents(id, 0)]).then(
      ([experiment, ev]) => {
        if (!alive) return;
        lastSeq.current = Math.max(experiment.last_seq, ev.last_seq, ...ev.events.map((e) => e.seq), 0);
        dispatch({ type: "loaded", id, experiment, events: ev.events });
      },
      (error) => alive && dispatch({ type: "error", id, error }),
    );
    return () => {
      alive = false;
    };
  }, [id, nonce]);

  const active = live.settled && isJobActive(live.job, live.status);

  useEffect(() => {
    if (!active) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      if (!alive) return;
      if (document.visibilityState === "hidden") return; // resumed by onVisible
      try {
        const ev = await api.getEvents(id, lastSeq.current);
        if (!alive) return;
        for (const e of ev.events) lastSeq.current = Math.max(lastSeq.current, e.seq);
        dispatch({ type: "events", id, events: ev.events, job: ev.job, status: ev.status });
        if (!isJobActive(ev.job, ev.status)) {
          const experiment = await api.getExperiment(id);
          if (alive) dispatch({ type: "experiment", id, experiment });
          return; // finished: stop polling
        }
      } catch {
        // Keep the last good state and try again; the page shows the data it already has.
      }
      if (alive) timer = setTimeout(tick, 1000);
    };
    const onVisible = () => {
      if (document.visibilityState === "visible") {
        if (timer) clearTimeout(timer);
        void tick();
      }
    };
    timer = setTimeout(tick, 1000);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [id, active]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return {
    experiment: live.experiment,
    events: live.events,
    job: live.job,
    status: live.status,
    error: live.error,
    loading: !live.settled,
    active,
    refresh,
  };
}

/* ------------------------------------------------------------------------------------ */
/* Small helpers                                                                         */
/* ------------------------------------------------------------------------------------ */

/** Width of an element in CSS px, kept current with a ResizeObserver. */
export function useElementWidth<T extends HTMLElement | SVGElement>(initial = 640) {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setWidth(Math.max(120, Math.round(el.getBoundingClientRect().width)));
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

/** Seconds since `since` (ISO string), ticking once a second while `running`. */
export function useElapsedSeconds(since: string | null | undefined, running: boolean): number {
  const [now, setNow] = useState(() => 0);
  useEffect(() => {
    if (!running) return;
    const update = () => setNow(Date.now());
    const first = setTimeout(update, 0);
    const id = setInterval(update, 1000);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, [running]);
  if (!since || now === 0) return 0;
  return Math.max(0, Math.round((now - new Date(since).getTime()) / 1000));
}

import type { components } from "@/lib/api-types";

export type Health = components["schemas"]["Health"];

const DEFAULT_API_BASE = "http://localhost:8000";

export function apiUrl(
  path: string,
  base: string = process.env.NEXT_PUBLIC_API_BASE ?? DEFAULT_API_BASE,
): string {
  return `${base.replace(/\/+$/, "")}/api/${path.replace(/^\/+/, "")}`;
}

export async function getHealth(): Promise<Health> {
  const response = await fetch(apiUrl("health"));
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}

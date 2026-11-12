// Data access: one typed fetch per API route (api/server.py), wrapped in TanStack Query hooks.
// The data is static per run, so it is cached for the session and never refetched on focus.
import { QueryClient, useQuery } from "@tanstack/react-query";
import type { Bout, ClipInfo, Features, Profile, Report } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(0, "Can't reach the API. Is it running? (.venv/bin/python -m api.server, port 8765)");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, `${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

const enc = encodeURIComponent;

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      retry: (n, err) => !(err instanceof ApiError && err.status === 404) && n < 2,
    },
  },
});

export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: () => get<string[]>("/api/runs") });

export const useProfiles = (run: string) =>
  useQuery({ queryKey: ["profiles", run], queryFn: () => get<Profile[]>(`/api/runs/${enc(run)}/profiles`) });

export const useClips = (run: string) =>
  useQuery({ queryKey: ["clips", run], queryFn: () => get<ClipInfo[]>(`/api/runs/${enc(run)}/clips`) });

export const useBouts = (run: string) =>
  useQuery({ queryKey: ["bouts", run], queryFn: () => get<Bout[]>(`/api/runs/${enc(run)}/bouts`) });

export const useReports = (run: string) =>
  useQuery({ queryKey: ["reports", run], queryFn: () => get<Record<string, Report>>(`/api/runs/${enc(run)}/reports`) });

export const useFeatures = (run: string, clip: string | undefined) =>
  useQuery({
    queryKey: ["features", run, clip],
    queryFn: () => get<Features>(`/api/runs/${enc(run)}/features/${enc(clip!)}`),
    enabled: !!clip,
  });

export const videoUrl = (run: string, clip: string) => `${BASE}/api/runs/${enc(run)}/videos/${enc(clip)}`;

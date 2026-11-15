import { CruxMark } from "@/components/wordmark";
import { Skeleton } from "@/components/ui/skeleton";

export default function Loading() {
  return (
    <div className="space-y-8" aria-busy="true" role="status" aria-label="Opening your Crux workspace">
      <div className="flex items-center gap-3 text-brand"><CruxMark className="size-6 motion-safe:animate-pulse" /><p className="text-sm">Opening your workspace…</p></div>
      <div className="space-y-3 border-b pb-6"><Skeleton className="h-10 w-96 max-w-full" /><Skeleton className="h-4 w-64 max-w-full" /></div>
      <div className="grid gap-5 lg:grid-cols-[1.35fr_1fr]"><div className="space-y-3"><Skeleton className="h-28 w-full" /><Skeleton className="h-28 w-full" /><Skeleton className="h-28 w-full" /></div><Skeleton className="h-72 w-full" /></div>
    </div>
  );
}

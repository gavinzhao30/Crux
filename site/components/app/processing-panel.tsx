"use client";

import { CheckIcon, Loader2Icon } from "lucide-react";

import { CruxMark } from "@/components/wordmark";
import { cn } from "@/lib/utils";

export function ProcessingPanel({ title, description, steps, active, percent }: {
  title: string;
  description: string;
  steps: string[];
  active: number;
  percent?: number;
}) {
  return (
    <section role="status" aria-live="polite" aria-busy="true" className="w-full rounded-2xl border border-brand/25 bg-card p-6 text-left sm:p-8">
      <div className="flex items-center gap-3 text-brand">
        <CruxMark className="size-7" />
        <span className="font-mono text-xs tracking-[0.12em] uppercase">Crux · working on your paper</span>
        <Loader2Icon className="ml-auto size-4 motion-safe:animate-spin" aria-hidden="true" />
      </div>
      <h2 className="mt-6 font-display text-3xl text-foreground">{title}</h2>
      <p className="mt-2 max-w-lg break-words text-sm leading-relaxed text-muted-foreground">{description}</p>
      <div role="progressbar" aria-label={steps[active]} aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={percent} aria-valuetext={percent === undefined ? "In progress" : `${percent}% uploaded`}
        className="mt-6 h-1.5 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full bg-brand transition-[width] duration-300", percent === undefined && "w-full motion-safe:animate-pulse opacity-50")}
          style={percent === undefined ? undefined : { width: `${percent}%` }} />
      </div>
      <ol className="mt-5 grid gap-3 sm:grid-cols-3">
        {steps.map((step, index) => (
          <li key={step} className={cn("flex items-center gap-2 text-xs", index > active ? "text-muted-foreground" : "text-foreground")}
            aria-current={index === active ? "step" : undefined}>
            <span className={cn("flex size-5 shrink-0 items-center justify-center rounded-full border font-mono text-[10px]",
              index <= active && "border-brand/30 bg-brand/10 text-brand")}>
              {index < active ? <CheckIcon className="size-3" aria-hidden="true" /> : index + 1}
            </span>
            {step}
          </li>
        ))}
      </ol>
    </section>
  );
}

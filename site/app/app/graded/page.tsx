import type { Metadata } from "next";

import { PageHeader } from "@/components/app/page-header";
import { GradedRuns } from "@/components/graded-runs";
import { JetsonRun } from "@/components/jetson-run";

export const metadata: Metadata = { title: "Graded runs" };

export default async function GradedRunsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const { run } = await searchParams;
  return (
    <div className="space-y-10">
      <PageHeader title="Graded runs">
        BixBench3 task 9 runs replayed from their event logs: questions, checks ranked by information per cost, the
        referee, belief updates and the rejected ledger.
      </PageHeader>
      <GradedRuns selected={typeof run === "string" ? run : undefined} basePath="/app/graded" />
      <section id="jetson" aria-labelledby="jetson-heading" className="scroll-mt-24 space-y-8 border-t pt-10">
        <div className="max-w-3xl">
          <p className="eyebrow">Live on hardware</p>
          <h2 id="jetson-heading" className="mt-3 font-display text-3xl leading-tight sm:text-4xl">
            One letter, read by a camera
          </h2>
          <p className="mt-3 text-lg leading-relaxed text-muted-foreground">
            The same engine on an NVIDIA Jetson: it chooses which part of a DNA strip to read, reads it through the
            camera, and settles which haemoglobin variant it carries. Then someone swaps the strip.
          </p>
        </div>
        <JetsonRun />
      </section>
    </div>
  );
}

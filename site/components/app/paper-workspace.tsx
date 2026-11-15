"use client";

import { ArrowLeftIcon, ArrowUpRightIcon, CheckIcon, FileTextIcon, Loader2Icon, QuoteIcon, SparklesIcon, TriangleAlertIcon } from "lucide-react";
import Link from "next/link";
import { useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { extractClaimsAction } from "@/app/app/actions";
import { ClaimSources } from "@/components/app/claim-sources";
import { DeletePaperButton } from "@/components/app/paper-actions";
import { ProcessingPanel } from "@/components/app/processing-panel";
import { StartAuditButton } from "@/components/app/start-audit-button";
import { Panel } from "@/components/run/panels";
import { Button } from "@/components/ui/button";
import { Chip } from "@/components/verdict-chip";
import { CruxMark } from "@/components/wordmark";
import { libraryClaim } from "@/lib/claim-library";
import { dateTime, fileSize, plural } from "@/lib/format";
import { paperDisplayTitle } from "@/lib/paper-display";
import type { PaperClaim, PaperSummary } from "@/lib/server/papers";
import { cn } from "@/lib/utils";

const KINDS = { symbolic: "Mathematical", statistical: "Statistical", empirical: "Reported finding" };

function SourcePassage({ claim, paperId, exaReady }: { claim: PaperClaim; paperId: string; exaReady: boolean }) {
  const method = libraryClaim(claim.libraryId);
  return (
    <div className="space-y-6 p-5 sm:p-6">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {claim.quoteFound ? <CheckIcon className="size-3.5 text-v-holds" aria-hidden="true" /> : <TriangleAlertIcon className="size-3.5 text-v-cond" aria-hidden="true" />}
        {claim.quoteFound ? "Quotation matched to paper text" : "Check this quotation against the PDF"}
      </div>
      <blockquote className="border-l-2 border-brand/45 pl-4 font-display text-xl leading-relaxed text-pretty">
        {claim.quote || "No supporting quotation was returned for this claim."}
      </blockquote>
      <p className="text-xs leading-relaxed text-muted-foreground">A matching quotation links a claim to its source. It does not establish that the claim is correct.</p>
      <ClaimSources key={claim.id} paperId={paperId} claimId={claim.id} enabled={exaReady} />
      {method && <div className="rounded-xl border bg-background p-4"><p className="eyebrow">Available audit method</p><p className="mt-2 text-sm leading-relaxed">{method.check}</p></div>}
    </div>
  );
}

export function PaperWorkspace({ paper, claims, keyReady, exaReady }: { paper: PaperSummary; claims: PaperClaim[]; keyReady: boolean; exaReady: boolean }) {
  const router = useRouter();
  const [pending, start] = useTransition();
  const [selectedId, select] = useState<string | null>(claims[0]?.id ?? null);
  const [error, setError] = useState<string | null>(null);
  const selected = claims.find((c) => c.id === selectedId) || claims[0];
  const extracted = paper.status === "extracted" || claims.length > 0;

  function extract() {
    setError(null);
    start(async () => {
      try {
        const result = await extractClaimsAction(paper.id);
        if (!result.ok) { setError(result.error || "Extraction failed. Try again."); return; }
        toast.success("Your claims are ready to explore.");
        router.refresh();
      } catch { setError("Extraction was interrupted. Please try again."); }
    });
  }

  const extractButton = (
    <Button size="lg" disabled={!keyReady || pending} onClick={extract} className="h-10 rounded-full px-5">
      {pending ? <Loader2Icon className="motion-safe:animate-spin" aria-hidden="true" /> : <SparklesIcon aria-hidden="true" />}
      {pending ? "Extracting claims…" : extracted ? "Extract again" : "Extract claims"}
    </Button>
  );

  return (
    <div className="space-y-7">
      <Link href="/app" className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"><ArrowLeftIcon className="size-3.5" aria-hidden="true" /> Your papers</Link>
      <header className="flex flex-wrap items-end justify-between gap-5 border-b pb-6">
        <div className="min-w-0 max-w-3xl">
          <p className="eyebrow mb-3 flex items-center gap-2"><CruxMark className="size-4" /> Paper workspace</p>
          <h1 className="font-display text-3xl leading-tight text-pretty break-words sm:text-4xl">{paperDisplayTitle(paper.title, paper.filename)}</h1>
          <p className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <span>{plural(paper.pages, "page")}</span><span>{fileSize(paper.size)}</span>
            <span>Added {dateTime(paper.createdAt)}</span>
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">{extractButton}<DeletePaperButton paperId={paper.id} name={paper.filename} disabled={pending} /></div>
      </header>

      {(error || paper.status === "failed") && <div role="alert" className="rounded-xl border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">{error || paper.error || "Extraction failed. Try again."}</div>}
      {!keyReady && <div role="status" className="rounded-xl border p-4 text-sm text-muted-foreground">Your paper is saved. Claim extraction is temporarily unavailable; please contact the workspace owner.</div>}

      {pending ? (
        <ProcessingPanel title="From paper to claims." description="Crux is identifying scientific statements and matching their quotations to the paper. This can take a minute or two. Keep this page open."
          steps={["PDF ready", "Extracting claims", "Prepare workspace"]} active={1} />
      ) : !extracted ? (
        <section className="grid overflow-hidden rounded-2xl border bg-card lg:grid-cols-[1.4fr_1fr]">
          <div className="p-7 sm:p-10">
            <span className="mb-6 inline-flex items-center gap-2 rounded-full border border-v-holds/25 bg-v-holds/5 px-3 py-1.5 text-xs text-v-holds"><CheckIcon className="size-3.5" aria-hidden="true" /> Paper uploaded</span>
            <h2 className="font-display text-4xl leading-tight">The paper is here.<br />Find the claims inside.</h2>
            <p className="mt-4 max-w-lg text-sm leading-relaxed text-muted-foreground">Turn the paper&apos;s scientific statements into an evidence-linked workspace. Inspect the claims, read their source passages, and decide what to investigate.</p>
            <div className="mt-7">{extractButton}</div>
            <p className="mt-3 text-xs text-muted-foreground">Extraction sends the paper&apos;s text to the model provider.</p>
          </div>
          <div className="flex flex-col justify-center gap-6 border-t bg-muted/25 p-7 lg:border-t-0 lg:border-l sm:p-10">
            {[{ label: "Read the paper", body: "Start from the document, not a checklist." }, { label: "Identify the claims", body: "Mathematical statements, statistical methods, and reported findings." }, { label: "Keep the evidence", body: "Every returned quotation stays beside its claim." }].map((step, i) => <div key={step.label} className="flex gap-4"><span className="pt-0.5 font-mono text-xs text-brand">0{i + 1}</span><div><p className="text-sm font-medium">{step.label}</p><p className="mt-1 text-xs leading-relaxed text-muted-foreground">{step.body}</p></div></div>)}
          </div>
        </section>
      ) : (
        <>
          <section className="flex flex-wrap items-center justify-between gap-6 rounded-2xl bg-ink p-6 text-ink-foreground sm:p-8">
            <div><p className="text-xs tracking-wider text-ink-foreground/60 uppercase">Claim extraction complete</p><h2 className="mt-2 font-display text-4xl">The claims. The evidence.</h2><p className="mt-3 max-w-xl text-sm leading-relaxed text-ink-foreground/65">Scientific statements extracted from your paper, with their supporting passages. Select a claim to explore its source.</p></div>
            <div className="border-l border-ink-foreground/20 pl-6"><p className="font-display text-5xl tabular-nums">{claims.length}</p><p className="mt-1 text-xs text-ink-foreground/65">{claims.length === 1 ? "claim extracted" : "claims extracted"}</p></div>
          </section>
          {claims.length === 0 ? <div className="rounded-xl border bg-card p-8 text-center"><p className="font-display text-2xl">No claims returned.</p><p className="mt-2 text-sm text-muted-foreground">Try extraction again or upload another text-based PDF.</p></div> : (
            <div className="grid items-start gap-5 lg:grid-cols-[1.35fr_1fr]">
              <Panel title="Extracted claims" hint="Select to inspect the source">
                <ol className="divide-y">
                  {claims.map((claim, i) => (
                    <li key={claim.id}>
                      <button onClick={() => select(claim.id)} aria-pressed={selected?.id === claim.id}
                        className={cn("w-full border-l-2 border-transparent px-5 py-5 text-left transition-colors hover:bg-muted/40 focus-visible:outline-2 focus-visible:outline-ring focus-visible:-outline-offset-2", selected?.id === claim.id && "border-brand bg-brand/5")}>
                        <div className="mb-3 flex items-center gap-2"><span className="mr-1 font-mono text-xs text-muted-foreground">{String(i + 1).padStart(2, "0")}</span><Chip>{KINDS[claim.kind]}</Chip><span className="ml-auto text-xs text-muted-foreground">Extracted</span></div>
                        <p className="text-[1.025rem] leading-relaxed text-pretty">{claim.claim}</p>
                        <span className="mt-3 inline-flex items-center gap-1 text-xs text-brand">Source passage <ArrowUpRightIcon className="size-3" aria-hidden="true" /></span>
                      </button>
                      {selected?.id === claim.id && <div className="border-t lg:hidden"><SourcePassage claim={claim} paperId={paper.id} exaReady={exaReady} /></div>}
                    </li>
                  ))}
                </ol>
              </Panel>
              {selected && <div className="sticky top-24 hidden lg:block"><Panel title="Source evidence" hint={<QuoteIcon className="size-4" aria-hidden="true" />}><div className="border-b px-6 py-4"><p className="font-mono text-xs text-muted-foreground">CLAIM {String(claims.indexOf(selected) + 1).padStart(2, "0")}</p><p className="mt-2 text-sm font-medium leading-relaxed">{selected.claim}</p></div><SourcePassage claim={selected} paperId={paper.id} exaReady={exaReady} /></Panel></div>}
            </div>
          )}
          <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Extraction details</summary><div className="mt-3 space-y-2 leading-relaxed"><p className="break-all"><FileTextIcon className="mr-1 inline size-3" aria-hidden="true" />{paper.filename}</p><p>{paper.model ? `Read by ${paper.model}. ` : ""}Extracted statements are not audit verdicts; review their source context before drawing conclusions.</p>{paper.truncated && <p>This extraction read part of the paper&apos;s text.</p>}</div></details>
        </>
      )}

      <section className="flex flex-wrap items-center justify-between gap-5 rounded-xl border bg-card p-5">
        <div className="max-w-xl"><p className="text-sm font-medium">See a claim become an experiment.</p><p className="mt-1 text-xs leading-relaxed text-muted-foreground">Explore the DESeq2 demonstration audit, with competing explanations and a record of the evidence. This example is separate from your paper&apos;s extraction.</p></div>
        <StartAuditButton variant="outline">Open example audit</StartAuditButton>
      </section>
    </div>
  );
}

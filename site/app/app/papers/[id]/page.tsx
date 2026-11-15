import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { PaperWorkspace } from "@/components/app/paper-workspace";
import { requireUser } from "@/lib/server/auth";
import { hasOpenAiKey } from "@/lib/server/openai";
import { getPaper, listClaims } from "@/lib/server/papers";

export const metadata: Metadata = { title: "Paper workspace" };

export default async function PaperPage({ params }: PageProps<"/app/papers/[id]">) {
  const { id } = await params;
  const user = await requireUser(`/app/papers/${id}`);
  const paper = /^[0-9a-f-]{36}$/.test(id) ? getPaper(user.id, id) : null;
  if (!paper) notFound();
  return <PaperWorkspace paper={paper} claims={listClaims(user.id, paper.id)} keyReady={hasOpenAiKey()} />;
}

import { Compass } from "lucide-react";
import EmbeddingsViz from "../components/EmbeddingsViz";

export default function EmbeddingsPage() {
  return (
    <div className="flex flex-col gap-8">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <span className="chip">Embedding</span>
          <span className="chip">Retrieval</span>
          <span className="chip">RAG</span>
          <span className="chip">Sentence-BERT · BGE · text-embedding-3</span>
        </div>
        <div className="flex items-start gap-4">
          <div className="hidden sm:flex w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-400 to-fuchsia-500 items-center justify-center shadow-[0_10px_30px_-10px_rgba(34,211,238,0.6)]">
            <Compass className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight">
              Embedding Models — language as geometry
            </h1>
            <p className="text-ink-300 max-w-2xl mt-1.5 leading-relaxed">
              Text in, vector out. Watch how meaning becomes coordinates,
              explore cosine similarity and analogies, see how modern
              embedders are trained, and compare the leading models you'd
              actually deploy.
            </p>
          </div>
        </div>
      </header>

      <EmbeddingsViz />
    </div>
  );
}

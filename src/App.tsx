import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import SoftmaxPage from "./pages/SoftmaxPage";
import SigmoidPage from "./pages/SigmoidPage";
import ReLUPage from "./pages/ReLUPage";
import TanhPage from "./pages/TanhPage";
import CrossEntropyPage from "./pages/CrossEntropyPage";
import MSEPage from "./pages/MSEPage";
import LossFamilyPage from "./pages/LossFamilyPage";
import KLDivergencePage from "./pages/KLDivergencePage";
import SamplingPage from "./pages/SamplingPage";
import PositionalEncodingsPage from "./pages/PositionalEncodingsPage";
import AttentionVariantsPage from "./pages/AttentionVariantsPage";
import KVCachePage from "./pages/KVCachePage";
import BPEPage from "./pages/BPEPage";
import MoEPage from "./pages/MoEPage";
import LoRAPage from "./pages/LoRAPage";
import GradientDescentPage from "./pages/GradientDescentPage";
import BatchNormPage from "./pages/BatchNormPage";
import QKVPage from "./pages/QKVPage";
import LLMHome from "./pages/LLMHome";
import GPTPage from "./pages/GPTPage";
import BERTPage from "./pages/BERTPage";
import QwenPage from "./pages/QwenPage";
import LLaMAPage from "./pages/LLaMAPage";
import EmbeddingsPage from "./pages/EmbeddingsPage";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route path="/softmax" element={<SoftmaxPage />} />
        <Route path="/sigmoid" element={<SigmoidPage />} />
        <Route path="/relu" element={<ReLUPage />} />
        <Route path="/tanh" element={<TanhPage />} />
        <Route path="/cross-entropy" element={<CrossEntropyPage />} />
        <Route path="/mse" element={<MSEPage />} />
        <Route path="/loss-family" element={<LossFamilyPage />} />
        <Route path="/kl-divergence" element={<KLDivergencePage />} />
        <Route path="/sampling" element={<SamplingPage />} />
        <Route path="/positional-encodings" element={<PositionalEncodingsPage />} />
        <Route path="/attention-variants" element={<AttentionVariantsPage />} />
        <Route path="/kv-cache" element={<KVCachePage />} />
        <Route path="/bpe" element={<BPEPage />} />
        <Route path="/moe" element={<MoEPage />} />
        <Route path="/lora" element={<LoRAPage />} />
        <Route path="/gradient-descent" element={<GradientDescentPage />} />
        <Route path="/batchnorm" element={<BatchNormPage />} />
        <Route path="/qkv" element={<QKVPage />} />
        <Route path="/llm" element={<LLMHome />} />
        <Route path="/llm/gpt" element={<GPTPage />} />
        <Route path="/llm/bert" element={<BERTPage />} />
        <Route path="/llm/qwen" element={<QwenPage />} />
        <Route path="/llm/llama" element={<LLaMAPage />} />
        <Route path="/llm/embeddings" element={<EmbeddingsPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}

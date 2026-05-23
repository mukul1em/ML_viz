import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import SoftmaxPage from "./pages/SoftmaxPage";
import SigmoidPage from "./pages/SigmoidPage";
import ReLUPage from "./pages/ReLUPage";
import TanhPage from "./pages/TanhPage";
import CrossEntropyPage from "./pages/CrossEntropyPage";
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

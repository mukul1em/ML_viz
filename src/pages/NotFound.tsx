import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex flex-col items-start gap-4 py-12">
      <div className="chip">404</div>
      <h1 className="text-3xl font-bold">This visualization isn't here yet.</h1>
      <p className="text-ink-300 max-w-md">
        The page you're looking for either moved or hasn't been built. Head
        back to the catalog and explore what's available.
      </p>
      <Link to="/" className="btn-primary">
        <ArrowLeft className="w-4 h-4" /> Back to home
      </Link>
    </div>
  );
}

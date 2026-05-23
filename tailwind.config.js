/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      colors: {
        ink: {
          950: "#0a0b10",
          900: "#0f1117",
          800: "#161922",
          700: "#1f2330",
          600: "#2a2f3d",
          500: "#3a4050",
          400: "#5b6478",
          300: "#8b94a8",
          200: "#c2c7d4",
          100: "#e6e8ee",
        },
        accent: {
          DEFAULT: "#7c5cff",
          soft: "#a48bff",
          glow: "#5b3fff",
        },
        teal: {
          DEFAULT: "#22d3ee",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(124,92,255,0.4), 0 10px 40px -10px rgba(124,92,255,0.45)",
      },
    },
  },
  plugins: [],
};

export type EmbeddingCategory =
  | "animals"
  | "vehicles"
  | "food"
  | "emotions"
  | "tech"
  | "sports"
  | "royalty";

export interface EmbeddingItem {
  id: number;
  text: string;
  category: EmbeddingCategory;
  /** 2D scatter position (canvas viewBox 720×540 with 60 margin). */
  x: number;
  y: number;
}

export const CATEGORY_COLORS: Record<EmbeddingCategory, string> = {
  animals: "#fbbf24",
  vehicles: "#38bdf8",
  food: "#f472b6",
  emotions: "#a78bfa",
  tech: "#34d399",
  sports: "#f87171",
  royalty: "#c084fc",
};

export const CATEGORY_LABELS: Record<EmbeddingCategory, string> = {
  animals: "Animals",
  vehicles: "Vehicles",
  food: "Food",
  emotions: "Emotions",
  tech: "Technology",
  sports: "Sports",
  royalty: "Royalty",
};

export const corpus: EmbeddingItem[] = [
  // ---- Sports (top center) ----
  { id: 0, text: "football", category: "sports", x: 360, y: 110 },
  { id: 1, text: "basketball", category: "sports", x: 405, y: 130 },
  { id: 2, text: "soccer", category: "sports", x: 330, y: 140 },
  { id: 3, text: "tennis", category: "sports", x: 415, y: 95 },
  { id: 4, text: "swimming", category: "sports", x: 295, y: 105 },
  { id: 5, text: "running", category: "sports", x: 370, y: 160 },
  { id: 6, text: "baseball", category: "sports", x: 430, y: 160 },
  { id: 7, text: "golf", category: "sports", x: 330, y: 80 },

  // ---- Animals (left top) ----
  { id: 8, text: "cat", category: "animals", x: 170, y: 220 },
  { id: 9, text: "dog", category: "animals", x: 220, y: 195 },
  { id: 10, text: "horse", category: "animals", x: 250, y: 165 },
  { id: 11, text: "lion", category: "animals", x: 145, y: 185 },
  { id: 12, text: "tiger", category: "animals", x: 195, y: 155 },
  { id: 13, text: "elephant", category: "animals", x: 215, y: 240 },
  { id: 14, text: "rabbit", category: "animals", x: 130, y: 235 },
  { id: 15, text: "bird", category: "animals", x: 100, y: 200 },

  // ---- Vehicles (right top) ----
  { id: 16, text: "car", category: "vehicles", x: 525, y: 240 },
  { id: 17, text: "truck", category: "vehicles", x: 585, y: 230 },
  { id: 18, text: "plane", category: "vehicles", x: 555, y: 180 },
  { id: 19, text: "boat", category: "vehicles", x: 500, y: 205 },
  { id: 20, text: "bicycle", category: "vehicles", x: 615, y: 215 },
  { id: 21, text: "train", category: "vehicles", x: 565, y: 265 },
  { id: 22, text: "helicopter", category: "vehicles", x: 605, y: 175 },
  { id: 23, text: "motorcycle", category: "vehicles", x: 490, y: 250 },

  // ---- Emotions (left bottom) ----
  { id: 24, text: "happy", category: "emotions", x: 145, y: 395 },
  { id: 25, text: "sad", category: "emotions", x: 200, y: 405 },
  { id: 26, text: "angry", category: "emotions", x: 155, y: 360 },
  { id: 27, text: "calm", category: "emotions", x: 230, y: 380 },
  { id: 28, text: "excited", category: "emotions", x: 200, y: 440 },
  { id: 29, text: "anxious", category: "emotions", x: 175, y: 340 },
  { id: 30, text: "joyful", category: "emotions", x: 110, y: 425 },
  { id: 31, text: "melancholic", category: "emotions", x: 130, y: 365 },

  // ---- Food (center bottom) ----
  { id: 32, text: "pizza", category: "food", x: 345, y: 430 },
  { id: 33, text: "pasta", category: "food", x: 395, y: 415 },
  { id: 34, text: "sushi", category: "food", x: 370, y: 385 },
  { id: 35, text: "bread", category: "food", x: 415, y: 395 },
  { id: 36, text: "apple", category: "food", x: 335, y: 395 },
  { id: 37, text: "banana", category: "food", x: 370, y: 445 },
  { id: 38, text: "salad", category: "food", x: 425, y: 435 },
  { id: 39, text: "cake", category: "food", x: 305, y: 415 },

  // ---- Tech (right bottom) ----
  { id: 40, text: "computer", category: "tech", x: 545, y: 410 },
  { id: 41, text: "internet", category: "tech", x: 595, y: 395 },
  { id: 42, text: "software", category: "tech", x: 615, y: 430 },
  { id: 43, text: "algorithm", category: "tech", x: 555, y: 370 },
  { id: 44, text: "database", category: "tech", x: 610, y: 365 },
  { id: 45, text: "network", category: "tech", x: 580, y: 445 },
  { id: 46, text: "server", category: "tech", x: 530, y: 385 },
  { id: 47, text: "code", category: "tech", x: 525, y: 440 },

  // ---- Royalty / analogy cluster (for "king − man + woman = queen") ----
  // Positioned so the parallelogram is exact in 2D.
  { id: 48, text: "king", category: "royalty", x: 275, y: 290 },
  { id: 49, text: "queen", category: "royalty", x: 345, y: 290 },
  { id: 50, text: "man", category: "royalty", x: 275, y: 330 },
  { id: 51, text: "woman", category: "royalty", x: 345, y: 330 },
  { id: 52, text: "prince", category: "royalty", x: 415, y: 290 },
  { id: 53, text: "princess", category: "royalty", x: 415, y: 330 },
];

/** Find an item by exact text (case-insensitive). */
export function findItem(text: string): EmbeddingItem | undefined {
  const q = text.trim().toLowerCase();
  return corpus.find((c) => c.text.toLowerCase() === q);
}

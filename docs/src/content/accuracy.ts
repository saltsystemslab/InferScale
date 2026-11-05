// Judged LoCoMo accuracy (%) over the 1,540 answerable questions, from the README.

export const topK = [5, 10, 20, 50] as const

export type AccuracyRow = {
  method: 'InferScale' | 'Mem0'
  /** InferScale's context window w; the Mem0 baseline has none. */
  window?: number
  /** One score per entry of `topK`. */
  scores: readonly number[]
}

export type ModelAccuracy = { model: string; rows: readonly AccuracyRow[] }

export const accuracy: readonly ModelAccuracy[] = [
  {
    model: 'Llama-3.1-8B',
    rows: [
      { method: 'InferScale', window: 0, scores: [62.21, 62.14, 57.66, 53.77] },
      { method: 'InferScale', window: 5, scores: [60.0, 59.81, 59.87, 56.82] },
      {
        method: 'InferScale',
        window: 20,
        scores: [60.13, 62.08, 61.62, 59.22],
      },
      {
        method: 'InferScale',
        window: 50,
        scores: [60.06, 61.36, 62.66, 60.26],
      },
      { method: 'Mem0', scores: [56.95, 59.29, 61.49, 63.25] },
    ],
  },
  {
    model: 'Mistral-7B',
    rows: [
      { method: 'InferScale', window: 0, scores: [54.03, 55.13, 52.84, 37.22] },
      { method: 'InferScale', window: 5, scores: [56.04, 57.6, 58.09, 56.45] },
      {
        method: 'InferScale',
        window: 20,
        scores: [53.96, 55.97, 56.95, 58.24],
      },
      { method: 'InferScale', window: 50, scores: [55.78, 56.49, 58.7, 58.3] },
      { method: 'Mem0', scores: [61.3, 63.96, 65.0, 64.35] },
    ],
  },
  {
    model: 'Qwen2.5-7B',
    rows: [
      { method: 'InferScale', window: 0, scores: [59.74, 56.75, 54.22, 50.45] },
      { method: 'InferScale', window: 5, scores: [59.29, 57.53, 58.05, 57.4] },
      { method: 'InferScale', window: 20, scores: [60.0, 59.68, 58.25, 58.64] },
      {
        method: 'InferScale',
        window: 50,
        scores: [58.83, 58.25, 58.18, 58.77],
      },
      { method: 'Mem0', scores: [60.06, 63.64, 65.13, 64.61] },
    ],
  },
]

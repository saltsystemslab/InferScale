import type { ReactNode } from 'react'
import raceGif from '../../../figures/ttft_race.gif'
import { cx } from '../cx'
import { paperUrl, repoUrl } from '../site'
import { Figure } from './Figure'
import { ArrowRightIcon, GitHubIcon, PaperIcon } from './Icons'

// The headline numbers from the paper's abstract.
const stats: { value: string; label: ReactNode }[] = [
  {
    value: '3.6-4.8×',
    label: (
      <>
        faster TTFT than Mem0 at <i>k=50</i>
      </>
    ),
  },
  { value: '3.7-4.5×', label: 'higher throughput under concurrent load' },
  {
    value: '60.3%',
    label: (
      <>
        LoCoMo accuracy at <i>k</i> = 50 on Llama-3.1-8B versus 63.3% baselines
      </>
    ),
  },
]

const button =
  'inline-flex items-center gap-2 rounded-lg border px-3.5 py-2 text-sm font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-ink'

const secondaryButton = cx(
  button,
  'border-primary-ink/20 text-primary-ink hover:border-primary-ink/30 hover:bg-primary-ink/5',
)

export function Hero() {
  return (
    <section id="overview">
      <h1 className="mt-6 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
        InferScale
      </h1>
      <p className="mt-3 text-lg text-ink-muted sm:text-xl">
        GPU-native KV injection for personalized LLM serving
      </p>
      <p className="mt-6 max-w-2xl leading-7">
        InferScale precomputes KV cache, retrieves the relevant facts
        on the GPU, and injects KV straight into vLLM's paged cache. Time-to-First-Token (TTFT)
        remains low even for large retrieval budgets, enabling high-throughput, low-latency LLM inference.
      </p>
      <div className="mt-8 flex flex-wrap gap-2.5">
        <a
          href="#installation"
          className={cx(button, 'border-transparent bg-primary text-white hover:bg-primary-hover')}
        >
          Get started
          <ArrowRightIcon className="size-4" />
        </a>
        <a href={paperUrl} className={secondaryButton}>
          <PaperIcon className="size-4" />
          Paper
        </a>
        <a href={repoUrl} className={secondaryButton}>
          <GitHubIcon className="size-4" />
          GitHub
        </a>
      </div>
      <dl className="mt-12 grid gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-3">
        {stats.map(({ value, label }) => (
          <div key={value} className="flex flex-col-reverse justify-end gap-1 bg-canvas p-5">
            <dt className="text-sm leading-6 text-ink-muted">{label}</dt>
            <dd className="text-3xl font-semibold tracking-tight text-ink">{value}</dd>
          </div>
        ))}
      </dl>
      <Figure
        src={raceGif}
        width={960}
        height={480}
        alt="Side-by-side replay of one LoCoMo question: InferScale streams its first token after 18 ms, while Mem0 prefills until its first token at 174 ms."
        caption={
          <>
            LoCoMo question answered by InferScale (<i>k</i> = 100, <i>w</i> = 50) and Mem0 (
            <i>k</i> = 100).
          </>
        }
      />
    </section>
  )
}

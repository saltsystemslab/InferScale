import Api from './api.mdx'
import Benchmarks from './benchmarks.mdx'
import Citation from './citation.mdx'
import HowItWorks from './how-it-works.mdx'
import Installation from './installation.mdx'
import Quickstart from './quickstart.mdx'
import Results from './results.mdx'

/** The page's sections in reading order; each file opens with its own h2. */
export const sections = {
  'how-it-works': HowItWorks,
  installation: Installation,
  quickstart: Quickstart,
  api: Api,
  benchmarks: Benchmarks,
  results: Results,
  citation: Citation,
}

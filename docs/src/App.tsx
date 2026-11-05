import { useMemo, useRef } from 'react'
import { Footer } from './components/Footer'
import { Header } from './components/Header'
import { Hero } from './components/Hero'
import { mdxComponents } from './components/mdx'
import { MobileNav } from './components/MobileNav'
import { Sidebar } from './components/Sidebar'
import { sections } from './content'
import { useHeadings, type Heading } from './hooks/useHeadings'
import { useScrollSpy } from './hooks/useScrollSpy'

const overview: Heading = { id: 'overview', text: 'Overview', depth: 2 }

// A heading counts as the current one once it scrolls within this many pixels
// of the top: just below the sticky header, where anchor links land.
const spyOffset = 96

export default function App() {
  const content = useRef<HTMLElement>(null)
  const sectionHeadings = useHeadings(content)
  const headings = useMemo(() => [overview, ...sectionHeadings], [sectionHeadings])
  const ids = useMemo(() => headings.map(({ id }) => id), [headings])
  const activeId = useScrollSpy(ids, spyOffset)

  return (
    <>
      <a
        href="#content"
        className="fixed top-2.5 left-3 z-50 -translate-y-20 rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-white transition-transform focus:translate-y-0 motion-reduce:transition-none"
      >
        Skip to content
      </a>
      <Header menu={<MobileNav headings={headings} activeId={activeId} />} />
      <div className="mx-auto max-w-[90rem] lg:grid lg:grid-cols-[17rem_minmax(0,1fr)]">
        <Sidebar headings={headings} activeId={activeId} />
        <main
          id="content"
          ref={content}
          className="min-w-0 px-4 py-12 sm:px-6 sm:py-16 lg:px-12 xl:px-20"
        >
          <div className="mx-auto max-w-3xl">
            <Hero />
            {Object.entries(sections).map(([name, Section]) => (
              <section key={name}>
                <Section components={mdxComponents} />
              </section>
            ))}
            <Footer />
          </div>
        </main>
      </div>
    </>
  )
}

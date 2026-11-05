import { cx } from '../cx'
import type { Heading } from '../hooks/useHeadings'

type Section = Heading & { subsections: Heading[] }

/** Nests every h3 under the h2 before it. */
function toSections(headings: Heading[]): Section[] {
  const sections: Section[] = []
  for (const heading of headings) {
    const parent = sections.at(-1)
    if (heading.depth === 3 && parent) parent.subsections.push(heading)
    else sections.push({ ...heading, subsections: [] })
  }
  return sections
}

type NavProps = {
  headings: Heading[]
  activeId: string | undefined
  onNavigate?: () => void
}

export function Nav({ headings, activeId, onNavigate }: NavProps) {
  return (
    <nav aria-label="On this page">
      <ul className="space-y-1">
        {toSections(headings).map((section) => {
          const inSection =
            section.id === activeId || section.subsections.some(({ id }) => id === activeId)
          return (
            <li key={section.id}>
              <a
                href={`#${section.id}`}
                onClick={onNavigate}
                aria-current={section.id === activeId ? 'location' : undefined}
                className={cx(
                  'block rounded-md px-2 py-1.5 text-sm font-medium transition-colors',
                  section.id === activeId
                    ? 'text-accent'
                    : inSection
                      ? 'text-ink'
                      : 'text-ink-muted hover:text-ink',
                )}
              >
                {section.text}
              </a>
              {section.subsections.length > 0 && (
                <ul className="mt-0.5 mb-2 ml-2 border-l border-line">
                  {section.subsections.map((subsection) => (
                    <li key={subsection.id}>
                      <a
                        href={`#${subsection.id}`}
                        onClick={onNavigate}
                        aria-current={subsection.id === activeId ? 'location' : undefined}
                        className={cx(
                          '-ml-px block border-l py-1 pr-2 pl-3.5 text-[13px] leading-5 transition-colors',
                          subsection.id === activeId
                            ? 'border-accent font-medium text-accent'
                            : 'border-transparent text-ink-muted hover:border-ink-muted hover:text-ink',
                        )}
                      >
                        {subsection.text}
                      </a>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

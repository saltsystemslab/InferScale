import { useEffect, useRef } from 'react'
import type { Heading } from '../hooks/useHeadings'
import { Nav } from './Nav'

type SidebarProps = { headings: Heading[]; activeId: string | undefined }

/** The desktop navigation; it scrolls itself to keep the active link in view. */
export function Sidebar({ headings, activeId }: SidebarProps) {
  const ref = useRef<HTMLElement>(null)

  useEffect(() => {
    const sidebar = ref.current
    if (!sidebar || !activeId) return
    const link = Array.from(sidebar.querySelectorAll('a')).find(
      (anchor) => anchor.getAttribute('href') === `#${activeId}`,
    )
    if (!link) return
    const margin = 48
    const bounds = sidebar.getBoundingClientRect()
    const target = link.getBoundingClientRect()
    if (target.top < bounds.top + margin) {
      sidebar.scrollTop -= bounds.top + margin - target.top
    } else if (target.bottom > bounds.bottom - margin) {
      sidebar.scrollTop += target.bottom - (bounds.bottom - margin)
    }
  }, [activeId])

  return (
    <aside
      ref={ref}
      className="sticky top-14 hidden h-[calc(100dvh-3.5rem)] overflow-y-auto border-r border-line px-6 py-10 lg:block"
    >
      <Nav headings={headings} activeId={activeId} />
    </aside>
  )
}

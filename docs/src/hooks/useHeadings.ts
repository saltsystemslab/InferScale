import { useLayoutEffect, useState, type RefObject } from 'react'

export type Heading = { id: string; text: string; depth: 2 | 3 }

/** The h2 and h3 headings rendered inside `root`, read once before the first paint. */
export function useHeadings(root: RefObject<HTMLElement | null>): Heading[] {
  const [headings, setHeadings] = useState<Heading[]>([])

  // The headings exist only in the DOM that MDX renders, so they are read after it commits.
  useLayoutEffect(() => {
    const elements = root.current?.querySelectorAll<HTMLHeadingElement>('h2[id], h3[id]') ?? []
    // oxlint-disable-next-line react/set-state-in-effect
    setHeadings(
      Array.from(elements, (element) => ({
        id: element.id,
        text: element.textContent ?? '',
        depth: element.tagName === 'H2' ? 2 : 3,
      })),
    )
  }, [root])

  return headings
}

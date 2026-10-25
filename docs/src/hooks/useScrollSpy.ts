import { useEffect, useState } from 'react'

/**
 * The id of the element the reader is in: the last one whose top has scrolled
 * above `offset` pixels, or the last id once the page is scrolled to the bottom.
 */
export function useScrollSpy(ids: string[], offset: number): string | undefined {
  const [activeId, setActiveId] = useState<string>()

  useEffect(() => {
    let frame = 0
    const update = () => {
      frame = 0
      const root = document.documentElement
      const atBottom = window.innerHeight + window.scrollY >= root.scrollHeight - 1
      let current = ids[0]
      for (const id of ids) {
        const element = document.getElementById(id)
        if (element && element.getBoundingClientRect().top <= offset) current = id
      }
      setActiveId(atBottom ? ids.at(-1) : current)
    }
    const schedule = () => {
      frame ||= requestAnimationFrame(update)
    }

    update()
    window.addEventListener('scroll', schedule, { passive: true })
    window.addEventListener('resize', schedule)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('scroll', schedule)
      window.removeEventListener('resize', schedule)
    }
  }, [ids, offset])

  return activeId
}

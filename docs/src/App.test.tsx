import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import App from './App'

describe('App', () => {
  it('gives every element a unique id', () => {
    const { container } = render(<App />)
    const ids = Array.from(container.querySelectorAll('[id]'), (element) => element.id)
    const duplicates = ids.filter((id, index) => ids.indexOf(id) !== index)
    expect(duplicates).toEqual([])
  })

  it('points every in-page link at an element on the page', () => {
    const { container } = render(<App />)
    const targets = Array.from(container.querySelectorAll('a[href^="#"]'), (anchor) =>
      decodeURIComponent(anchor.getAttribute('href')!.slice(1)),
    )
    expect(targets.length).toBeGreaterThan(0)
    expect(targets.filter((id) => !document.getElementById(id))).toEqual([])
  })

  it('lists every section and subsection in the navigation', () => {
    const { container } = render(<App />)
    const [nav] = screen.getAllByRole('navigation', { name: 'On this page' })
    const headings = Array.from(container.querySelectorAll('main :is(h2, h3)'))
    expect(headings.length).toBeGreaterThan(10)
    for (const heading of headings) {
      expect(within(nav!).getByRole('link', { name: heading.textContent! })).toHaveProperty(
        'hash',
        `#${heading.id}`,
      )
    }
  })
})

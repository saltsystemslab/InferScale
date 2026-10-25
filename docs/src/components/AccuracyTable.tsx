import { useId, useState } from 'react'
import { accuracy, topK } from '../content/accuracy'
import { cx } from '../cx'
import { table, tableCell, tableFrame, tableHeadCell } from './styles'

/** LoCoMo accuracy for one model at a time; the best score for each k is bold. */
export function AccuracyTable() {
  const name = useId()
  const [model, setModel] = useState(accuracy[0]!.model)
  const rows = accuracy.find((entry) => entry.model === model)!.rows
  const best = topK.map((_, column) => Math.max(...rows.map((row) => row.scores[column]!)))

  return (
    <div className="mt-6">
      <fieldset className="inline-flex flex-wrap gap-1 rounded-lg border border-line bg-surface p-1">
        <legend className="sr-only">Model</legend>
        {accuracy.map((entry) => (
          <label
            key={entry.model}
            className="cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium text-ink-muted transition-colors select-none not-has-checked:hover:text-ink has-checked:bg-primary has-checked:text-white has-checked:shadow-sm has-focus-visible:outline-2 has-focus-visible:outline-offset-2 has-focus-visible:outline-primary-ink"
          >
            <input
              type="radio"
              name={name}
              value={entry.model}
              checked={entry.model === model}
              onChange={() => setModel(entry.model)}
              className="sr-only"
            />
            {entry.model}
          </label>
        ))}
      </fieldset>
      <div className={tableFrame}>
        <table className={table}>
          <caption className="sr-only">Judged LoCoMo accuracy (%) for {model}</caption>
          <thead>
            <tr>
              <th scope="col" className={tableHeadCell}>
                Method
              </th>
              {topK.map((k) => (
                <th key={k} scope="col" className={cx(tableHeadCell, 'text-right')}>
                  <i>k</i> = {k}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.method}-${row.window}`}>
                <th scope="row" className={cx(tableCell, 'font-normal whitespace-nowrap')}>
                  {row.method}
                  {row.window !== undefined && (
                    <span className="block text-ink-muted sm:ml-2 sm:inline">
                      <i>w</i> = {row.window}
                    </span>
                  )}
                </th>
                {row.scores.map((score, column) => (
                  <td
                    key={topK[column]}
                    className={cx(
                      tableCell,
                      'text-right',
                      score === best[column] && 'font-semibold text-ink',
                    )}
                  >
                    {score.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

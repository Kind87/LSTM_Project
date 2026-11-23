import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import {
  Aperture, BarChart3, Camera, Check, CircleAlert, Cpu, Download, ImagePlus,
  Layers3, ScanLine, Sun, Target, Upload, X,
} from 'lucide-react'
import './styles.css'

const lights = [
  { id: 'dark', text: '偏暗', factor: '×0.75' },
  { id: 'original', text: '原始', factor: '×1.00' },
  { id: 'bright', text: '偏亮', factor: '×1.25' },
]
const modelNames = ['PatchCore', 'CNN+LSTM', 'Isolation Forest', 'CutPaste']
const factors = ['70%', '80%', '90%', '100%']

async function request(path, options) {
  const response = await fetch(path, options)
  if (!response.ok) {
    const { detail } = await response.json().catch(() => ({}))
    throw new Error(typeof detail === 'string' ? detail : '無法取得結果，請確認服務仍在執行。')
  }
  return response.json()
}

const percentFormat = new Intl.NumberFormat('zh-TW', { style: 'percent', maximumFractionDigits: 1 })
const percent = value => percentFormat.format(Number(value))
const decimal = (value, digits) => new Intl.NumberFormat('zh-TW', {
  minimumFractionDigits: digits, maximumFractionDigits: digits,
}).format(value)
const viewFromUrl = () => new URLSearchParams(window.location.search).get('view') === 'results' ? 'metrics' : 'inspect'

function Verdict({ anomaly }) {
  return (
    <span className={'verdict ' + (anomaly ? 'verdict-alert' : 'verdict-good')}>
      {anomaly ? <CircleAlert size={15} aria-hidden="true" /> : <Check size={15} aria-hidden="true" />}
      {anomaly ? '疑似瑕疵' : '正常'}
    </span>
  )
}

function UploadZone({ file, active, onFile }) {
  const [dragging, setDragging] = useState(false)
  return (
      <label
        className={`upload upload-main ${active ? 'upload-compact' : ''} ${dragging ? 'upload-active' : ''}`}
        onDragOver={event => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={event => {
          event.preventDefault()
          setDragging(false)
          onFile(event.dataTransfer.files[0])
        }}
      >
        <input type="file" name="image" aria-label="選擇要檢測的圖片"
          aria-describedby="upload-format" accept="image/png,image/jpeg,image/webp" onChange={event => {
          onFile(event.target.files[0])
          event.target.value = ''
        }} />
        <span className="upload-icon"><Upload size={active ? 18 : 24} aria-hidden="true" /></span>
        <strong>{active ? '更換圖片' : '選擇圖片或拖放到這裡'}</strong>
        <span id="upload-format">{file ? file.name : <>JPG、PNG、WebP，最大 10&nbsp;MB</>}</span>
      </label>
  )
}

function SidePanel({ samples, selected, onSample }) {
  return (
    <aside className="panel library">
      <div className="panel-heading"><Layers3 size={18} aria-hidden="true" /><h2>資料集樣本</h2></div>
      <div className="sample-grid">
        {samples.map(sample => (
          <button className={'sample ' + (selected === sample.id ? 'sample-active' : '')}
            key={sample.id} onClick={() => onSample(sample.id)}
            aria-pressed={selected === sample.id}>
            <img src={'/api/samples/' + sample.id + '/image'} width="1024" height="1024" alt="" loading="lazy" />
            <span className="min-w-0 text-left">
              <strong>{sample.name}</strong>
              <small className="mono">{sample.filename}</small>
            </span>
            <span className={'sample-dot ' + (sample.anomaly ? 'sample-dot-alert' : '')} />
          </button>
        ))}
      </div>
    </aside>
  )
}

function ImageCell({ title, tag, src, alt, busy }) {
  return (
    <figure className="min-w-0">
      <figcaption className="image-caption"><span>{title}</span><span className="image-tag mono">{tag}</span></figcaption>
      <div className="image-tile">
        {src ? <img src={src} width="660" height="660" alt={alt} fetchPriority="high" /> : (
          <div className="image-empty"><Camera size={26} aria-hidden="true" /> <span>{busy ? '正在分析影像…' : '等待影像'}</span></div>
        )}
      </div>
    </figure>
  )
}

function InspectionStatus({ result, busy }) {
  const flagged = result ? modelNames.filter(name => result.models[name].anomaly).length : null
  const message = !result
    ? (busy ? '正在分析影像…' : '加入圖片後會自動顯示判定')
    : (flagged === 0 ? '四種方法都判為正常' : flagged === modelNames.length ? '四種方法都標示疑似瑕疵' : flagged + '/' + modelNames.length + ' 種方法標示疑似瑕疵，判定不一致')
  return <p className={`verdict-summary ${flagged === 0 ? 'verdict-good' : flagged ? 'verdict-alert' : ''}`} role="status">
    {flagged === 0 ? <Check size={18} aria-hidden="true" /> : <CircleAlert size={18} aria-hidden="true" />}
    {message}
  </p>
}

function Results({ result, heatModel, onHeatModel }) {
  return (
    <aside className="panel result-panel">
      <div className="panel-heading"><Target size={18} aria-hidden="true" /><h2>模型判定</h2></div>
      <div className="flex flex-col gap-3">
        {modelNames.map((name, index) => {
          const model = result?.models[name]
          const ratio = model ? model.score / model.threshold : 0
          const format = value => decimal(value, index === 1 || index === 2 ? 5 : 3)
          return (
            <article className={'model-card ' + (heatModel === name ? 'model-card-active' : '')} key={name}>
              <div className="flex items-center justify-between gap-2">
                <h3 translate="no">{name}</h3>
                {model && <Verdict anomaly={model.anomaly} />}
              </div>
              <div className="score-label">異常分數</div>
              <div className="score mono">{model ? format(model.score) : '—'}</div>
              <div className="gauge" aria-hidden="true">
                <span className={model?.anomaly ? 'gauge-alert' : ''} style={{ width: Math.min(ratio / 1.6, 1) * 100 + '%' }} />
                <i />
              </div>
              <div className="threshold"><span>判定門檻</span><span className="mono">{model ? format(model.threshold) : '—'}</span></div>
              <button className="map-button" onClick={() => onHeatModel(name)} aria-pressed={name === heatModel} disabled={!model}>
                <Aperture size={15} aria-hidden="true" />{name === heatModel ? '正在顯示熱圖' : '顯示熱圖'}
              </button>
            </article>
          )
        })}
      </div>
      <p className="result-help"><CircleAlert size={16} aria-hidden="true" /> 各模型的分數與自身門檻比較。熱圖用來觀察高分區域，定位準確度待驗證。</p>
    </aside>
  )
}

function Metrics() {
  const [rows, setRows] = useState([])
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    request('/api/metrics', { signal: controller.signal }).then(setRows).catch(err => {
      if (err.name !== 'AbortError') setError(err.message)
    })
    return () => controller.abort()
  }, [])
  return (
    <section className="panel metrics-panel">
      <div className="panel-heading flex-wrap">
        <BarChart3 size={18} aria-hidden="true" /><h2>三種光照比較</h2>
        <div className="metrics-actions">
          <a href="/api/artifacts/metrics.csv" download><Download size={15} aria-hidden="true" />比較表 CSV</a>
          <a href="/api/artifacts/report.md" download><Download size={15} aria-hidden="true" />原始雙模型報告</a>
        </div>
      </div>
      <p className="muted metrics-intro">四種方法都使用相同的 124 張測試圖，並各自用正常驗證圖設定門檻。</p>
      {error && <p className="error-strip" role="alert">{error}</p>}
      {!rows.length && !error && <p role="status">載入實測結果中…</p>}
      {!!rows.length && (
        <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="可橫向捲動的模型比較表">
          <table>
            <caption className="sr-only">四種方法在三種光照下的瑕疵檢測結果</caption>
            <thead><tr><th scope="col">測試光照</th><th scope="col">模型</th><th scope="col" translate="no">Image AUROC</th><th scope="col">正常圖誤報</th><th scope="col">瑕疵圖漏檢</th></tr></thead>
            <tbody>{rows.map(row => (
              <tr key={row.condition + row.model}>
                <td>{lights.find(light => light.id === row.condition).text}</td>
                <td translate="no">{row.model}</td>
                <td className="mono">{decimal(Number(row.image_auroc), 4)}</td>
                <td><span className="mono">{row.false_positives}/{row.normal_count}</span> <small>{percent(row.false_positive_rate)}</small></td>
                <td><span className="mono">{row.false_negatives}/{row.defect_count}</span> <small>{percent(row.false_negative_rate)}</small></td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {!!rows.length && <div className="experiment-figures">
        <figure><figcaption>四種方法比較圖</figcaption><img src="/api/artifacts/comparison.png" width="1760" height="640" alt="三種光照下的 AUROC、誤報率與漏檢率長條圖" loading="lazy" /></figure>
        <figure><figcaption>原始雙模型誤報與漏檢案例</figcaption><img src="/api/artifacts/error_examples.png" width="1168" height="1168" alt="兩種模型各自誤報與漏檢的皮革照片" loading="lazy" /></figure>
      </div>}
    </section>
  )
}

function App() {
  const [tab, setTab] = useState(viewFromUrl)
  const [samples, setSamples] = useState([])
  const [lighting, setLighting] = useState('original')
  const [job, setJob] = useState(null)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const [heatModel, setHeatModel] = useState('CNN+LSTM')
  const [error, setError] = useState('')
  const [libraryError, setLibraryError] = useState('')

  useEffect(() => {
    const sync = () => setTab(viewFromUrl())
    window.addEventListener('popstate', sync)
    return () => window.removeEventListener('popstate', sync)
  }, [])

  function navigate(event, nextTab) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    if (nextTab === tab) return
    window.history.pushState(null, '', event.currentTarget.href)
    setTab(nextTab)
    document.getElementById('main-content')?.focus()
  }

  useEffect(() => {
    const controller = new AbortController()
    request('/api/samples', { signal: controller.signal }).then(setSamples).catch(err => {
      if (err.name !== 'AbortError') setLibraryError(err.message)
    })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    if (!job) return
    const controller = new AbortController()
    let response
    if (job.file) {
      const data = new FormData()
      data.set('file', job.file)
      data.set('lighting', job.lighting)
      response = request('/api/inspect', { method: 'POST', body: data, signal: controller.signal })
    } else {
      response = request('/api/samples/' + job.selected + '/inspect?lighting=' + job.lighting,
        { signal: controller.signal })
    }
    response.then(setResult).catch(err => {
      if (err.name !== 'AbortError') setError(err.message)
    }).finally(() => {
      if (!controller.signal.aborted) setBusy(false)
    })
    return () => controller.abort()
  }, [job])

  function inspect(nextSample, nextFile, nextLight) {
    setError('')
    setBusy(true)
    setResult(null)
    setJob({ selected: nextSample, file: nextFile, lighting: nextLight })
  }

  function selectLight(nextLight) {
    if (nextLight === lighting) return
    setLighting(nextLight)
    if (job) inspect(job.selected, job.file, nextLight)
  }

  function selectFile(image) {
    if (!image) return
    if (image.size > 10 * 1024 * 1024) {
      setError('請將圖片縮小至 10 MB 以內，再重新選擇檔案。')
      return
    }
    inspect(null, image, lighting)
  }

  const file = job?.file
  const sample = samples.find(entry => entry.id === job?.selected)
  return (
    <>
      <a className="skip-link" href="#main-content">跳至主要內容</a>
      <header className="topbar">
        <div className="shell flex min-h-16 items-center justify-between gap-3">
          <div className="brand" translate="no"><span className="brand-icon"><ScanLine size={23} aria-hidden="true" /></span><span>Surface Lab</span></div>
          <nav className="flex items-center gap-1" aria-label="頁面">
            <a href="/" className={tab === 'inspect' ? 'nav-button active' : 'nav-button'} onClick={event => navigate(event, 'inspect')} aria-current={tab === 'inspect' ? 'page' : undefined}>
              <ScanLine size={16} aria-hidden="true" /> 即時檢測
            </a>
            <a href="/?view=results" className={tab === 'metrics' ? 'nav-button active' : 'nav-button'} onClick={event => navigate(event, 'metrics')} aria-current={tab === 'metrics' ? 'page' : undefined}>
              <BarChart3 size={16} aria-hidden="true" /> 實驗結果
            </a>
          </nav>
          <div className="local-pill"><span /> 本機推論 <Cpu size={14} aria-hidden="true" /></div>
        </div>
      </header>
      <main id="main-content" tabIndex={-1} className="shell">
        <div className="hero">
          <div>
            <h1>{tab === 'inspect' ? '皮革表面瑕疵檢測' : '皮革瑕疵檢測實驗結果'}</h1>
            <p>{tab === 'inspect' ? '加入一張皮革照片，四種方法會自動判斷是否有疑似瑕疵。' : '比較原始、偏暗與偏亮條件下的模型表現。'}</p>
          </div>
        </div>
        {tab === 'metrics' ? <Metrics /> : (
          <div className="dashboard">
            <section className="panel workspace">
              <div className="panel-heading workspace-heading">
                <div className="flex items-center gap-2.5"><ScanLine size={18} aria-hidden="true" /><h2>檢測工作區</h2></div>
                {sample && !file && <span className="truth-label">資料標註 <b>{sample.name}</b></span>}
              </div>
              <UploadZone file={file} active={Boolean(job)} onFile={selectFile} />
              <div className="controls">
                <div className="flex flex-wrap items-center gap-2.5">
                  <span className="light-label"><Sun size={16} aria-hidden="true" /> 光照</span>
                  <div className="segmented" role="group" aria-label="光照條件">
                    {lights.map(light => (
                      <button key={light.id} className={light.id === lighting ? 'selected' : ''}
                        aria-pressed={light.id === lighting} onClick={() => selectLight(light.id)}
                        title={light.factor}>{light.text}</button>
                    ))}
                  </div>
                </div>
                {job && error && <button className="run-button" onClick={() => inspect(job.selected, job.file, lighting)} disabled={busy}><ScanLine size={17} aria-hidden="true" />重新檢測</button>}
              </div>
              {(error || libraryError) && <div className="error-strip" role="alert"><CircleAlert size={17} aria-hidden="true" /><span>{error || libraryError}</span>{error && <button onClick={() => setError('')} aria-label="關閉錯誤訊息"><X size={17} aria-hidden="true" /></button>}</div>}
              <InspectionStatus result={result} busy={busy} />
              {job && <>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <ImageCell title="待測影像" tag={lights.find(light => light.id === job.lighting).factor}
                  src={result?.preview || (sample && job.lighting === 'original' ? '/api/samples/' + sample.id + '/image' : '')}
                  alt={file?.name || sample?.name || '皮革樣本'} busy={busy} />
                <ImageCell title="異常熱圖" tag={heatModel} src={result?.models[heatModel].heatmap}
                  alt={heatModel + ' 異常熱圖'} busy={busy} />
              </div>
              <div className="heat-legend"><span>低異常</span><i /><span>高異常</span></div>
              <div className="sequence">
                <div className="sequence-title"><ImagePlus size={16} aria-hidden="true" /><h3>LSTM 亮度序列</h3><span>前三張為輸入，最後一張為預測目標</span></div>
                <div className="grid grid-cols-4 gap-2">
                  {factors.map((factor, index) => (
                    <figure className="frame" key={factor}>
                      {result && <img src={result.frames[index]} width="660" height="660" loading="lazy" alt={factor + ' 亮度版本'} />}
                      <figcaption className="mono">{factor}</figcaption>
                    </figure>
                  ))}
                </div>
              </div>
              </>}
            </section>
            <Results result={result} heatModel={heatModel} onHeatModel={setHeatModel} />
            <SidePanel samples={samples} selected={job?.selected} onSample={id => inspect(id, null, lighting)} />
          </div>
        )}
        <footer><span>Surface Lab</span><span>React · Tailwind CSS · FastAPI</span></footer>
      </main>
    </>
  )
}

createRoot(document.getElementById('root')).render(<App />)

import { useEffect, useMemo, useState } from 'react'
import { api } from './api.js'
import { Chart, contributionOption, equityOption, radarOption } from './charts.jsx'

const tabs = [
  ['dashboard', '总览'], ['strategyExplorer', '策略说明'], ['factors', '多因子分析'], ['backtest', '回测分析'], ['simulation', '模拟交易'], ['events', '市场事件研究'], ['research', 'V3 研究实验室'], ['researchV4', 'V4 单因子验证'], ['strategyLab', 'V3 策略实验室'], ['strategyLabV4', 'V4 策略实验室'],
]

const pct = (value) => value == null ? '—' : `${Number(value).toFixed(2)}%`
const number = (value) => value == null ? '—' : Number(value).toFixed(2)

export default function App() {
  const [tab, setTab] = useState('dashboard')
  const [data, setData] = useState({ loading: true })
  const [selectedCode, setSelectedCode] = useState(null)
  const [labForm, setLabForm] = useState(null)
  const [labRunning, setLabRunning] = useState(false)
  const [v4LabForm, setV4LabForm] = useState(null)
  const [v4LabRunning, setV4LabRunning] = useState(false)
  const [eventSyncing, setEventSyncing] = useState(false)
  const [selectedStrategyVersion, setSelectedStrategyVersion] = useState(null)
  const [strategyDetail, setStrategyDetail] = useState(null)
  const [comparisonVersion, setComparisonVersion] = useState('')
  const [comparisonDetail, setComparisonDetail] = useState(null)
  const [v4FactorBusy, setV4FactorBusy] = useState(false)

  async function refresh() {
    setData((old) => ({ ...old, loading: true, error: '' }))
    try {
      const [health, market, ranking, backtest, backtestVersions, strategyVersions, equityA, equityB, trades, tradeSummary, events, v3Research, v4Research, v4PriceGaps, v4FactorRecipes, v4FactorExperiments, v4MarketAudit, strategyLabDefault, strategyLabExperiments, v4StrategyLabDefault, v4StrategyLabExperiments] = await Promise.all([
        api.health(), api.market(), api.ranking(), api.backtest(), api.backtestVersions(), api.strategyVersions(), api.equity('a'), api.equity('b'), api.trades(), api.tradeSummary(), api.events(), api.v3Research(), api.v4Research(), api.v4PriceGaps(), api.v4FactorRecipes(), api.v4FactorExperiments(), api.v4MarketPredictiveAudit(), api.strategyLabDefault(), api.strategyLabExperiments(), api.v4StrategyLabDefault(), api.v4StrategyLabExperiments(),
      ])
      setData({ health, market, ranking, backtest, backtestVersions, strategyVersions, equityA, equityB, trades, tradeSummary, events, v3Research, v4Research, v4PriceGaps, v4FactorRecipes, v4FactorExperiments, v4MarketAudit, strategyLabDefault, strategyLabExperiments, v4StrategyLabDefault, v4StrategyLabExperiments, loading: false, error: '' })
      setSelectedCode((code) => code ?? ranking[0]?.stock_code ?? null)
      setLabForm((current) => current ?? strategyLabDefault)
      setV4LabForm((current) => current ?? v4StrategyLabDefault)
      setSelectedStrategyVersion((current) => current ?? strategyVersions.default_version)
    } catch (error) {
      setData((old) => ({ ...old, loading: false, error: `无法读取研究 API：${error.message}` }))
    }
  }

  useEffect(() => { refresh() }, [])
  useEffect(() => {
    if (!selectedStrategyVersion) return
    api.strategyVersion(selectedStrategyVersion).then(setStrategyDetail).catch((error) => {
      setData((old) => ({ ...old, error: `无法读取策略说明：${error.message}` }))
    })
  }, [selectedStrategyVersion])
  useEffect(() => {
    if (!comparisonVersion || comparisonVersion === selectedStrategyVersion) { setComparisonDetail(null); return }
    api.strategyVersion(comparisonVersion).then(setComparisonDetail).catch((error) => {
      setData((old) => ({ ...old, error: `无法读取比较版本：${error.message}` }))
    })
  }, [comparisonVersion, selectedStrategyVersion])
  const selected = useMemo(() => data.ranking?.find((item) => item.stock_code === selectedCode) ?? data.ranking?.[0], [data.ranking, selectedCode])
  const metrics = data.backtest?.models?.[0]

  async function runLabExperiment() {
    if (!labForm || labRunning) return
    setLabRunning(true)
    try {
      const result = await api.runStrategyLab(labForm)
      setData((old) => ({ ...old, strategyLabExperiments: [{
        experiment_id: result.experiment_id, created_at: new Date().toISOString(), status: 'completed',
        hypothesis: result.strategy.hypothesis, weights: result.strategy.factor_weights, top_n: result.strategy.top_n,
        market_regime_gate: result.strategy.market_regime_gate, technical_variant: result.technical_variant,
        metrics: result.metrics, scope: 'snapshot_replay_in_sample',
      }, ...(old.strategyLabExperiments ?? [])] }))
    } catch (error) {
      setData((old) => ({ ...old, error: `策略实验未运行：${error.message}` }))
    } finally {
      setLabRunning(false)
    }
  }

  async function runV4LabExperiment() {
    if (!v4LabForm || v4LabRunning) return
    setV4LabRunning(true)
    try {
      const result = await api.runV4StrategyLab(v4LabForm)
      setData((old) => ({ ...old, v4StrategyLabExperiments: [{
        experiment_id: result.experiment_id, created_at: new Date().toISOString(), status: 'completed',
        hypothesis: result.strategy.hypothesis, weights: result.strategy.factor_weights, top_n: result.strategy.top_n,
        market_regime_gate: result.strategy.market_regime_gate, technical_variant: result.technical_variant,
        metrics: result.metrics, scope: 'v4_entry_timing_snapshot_replay',
      }, ...(old.v4StrategyLabExperiments ?? [])] }))
    } catch (error) {
      setData((old) => ({ ...old, error: `V4 策略实验未运行：${error.message}` }))
    } finally {
      setV4LabRunning(false)
    }
  }

  async function syncOfficialEvents() {
    if (eventSyncing) return
    setEventSyncing(true)
    try {
      await api.syncOfficialEvents()
      const events = await api.events()
      setData((old) => ({ ...old, events }))
    } catch (error) {
      setData((old) => ({ ...old, error: `官方事件同步失败：${error.message}` }))
    } finally {
      setEventSyncing(false)
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><img src="/ai-quant-logo.png" alt="AI Quant Research" /><div><p className="eyebrow">AI QUANT RESEARCH</p><h1>量化研究终端</h1></div></div>
        <div className="topbar-meta"><span className={`status-dot ${data.error ? 'danger' : ''}`}></span><span>{data.loading ? '同步中' : data.error ? 'API 不可用' : '研究数据已连接'}</span><button className="icon-button" title="刷新研究数据" onClick={refresh} aria-label="刷新研究数据">↻</button></div>
      </header>
      <nav className="tabs" aria-label="研究导航">{tabs.map(([id, label]) => <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{label}</button>)}</nav>
      {data.error && <div className="notice error">{data.error}。请先启动 FastAPI 服务。</div>}
      {data.loading && !data.market ? <div className="loading">正在加载本地研究数据…</div> : <>
        {tab === 'dashboard' && <Dashboard market={data.market} ranking={data.ranking} backtest={data.backtest} versions={data.backtestVersions} v4={data.v4Research} experiments={data.v4StrategyLabExperiments ?? []} equityA={data.equityA} equityB={data.equityB} onNavigate={setTab} />}
        {tab === 'strategyExplorer' && <StrategyExplorer catalog={data.strategyVersions} detail={strategyDetail} selectedVersion={selectedStrategyVersion} onSelect={(version) => { setSelectedStrategyVersion(version); setComparisonVersion('') }} comparisonVersion={comparisonVersion} setComparisonVersion={setComparisonVersion} comparisonDetail={comparisonDetail} onOpenLab={(factor) => { setV4LabForm((current) => ({ ...(current ?? {}), factor_weights: { market_regime: 0, momentum: 0, money_flow: 0, fundamental: 0, technical: 0, [factor]: 1 } })); setTab('strategyLabV4') }} />}
        {tab === 'factors' && <Factors ranking={data.ranking} selected={selected} selectedCode={selectedCode} setSelectedCode={setSelectedCode} />}
        {tab === 'backtest' && <Backtest result={data.backtest} equityA={data.equityA} equityB={data.equityB} />}
        {tab === 'simulation' && <Simulation trades={data.trades} summary={data.tradeSummary} />}
        {tab === 'events' && <Events events={data.events} syncing={eventSyncing} onSync={syncOfficialEvents} />}
        {tab === 'research' && <V3Research status={data.v3Research} />}
        {tab === 'researchV4' && <V4FactorValidation status={data.v4Research} priceGaps={data.v4PriceGaps} recipes={data.v4FactorRecipes} experiments={data.v4FactorExperiments ?? []} marketAudit={data.v4MarketAudit} busy={v4FactorBusy} onRefresh={refresh} onRun={async (payload) => { setV4FactorBusy(true); try { await api.runV4FactorExperiment(payload); await refresh() } finally { setV4FactorBusy(false) } }} onMarketAudit={async () => { setV4FactorBusy(true); try { await api.v4MarketPredictiveAudit(); await refresh() } finally { setV4FactorBusy(false) } }} onApprove={async (id, approved, note) => { await api.approveV4FactorExperiment(id, { approved, approval_note: note }); await refresh() }} onActivate={async (name, recipeId, experimentId) => { if (!window.confirm('确认把该研究配方设为 V4 当前配方？这不会修改正式策略，也不会覆盖历史实验。')) return; await api.activateV4Recipe(name, recipeId, { experiment_id: experimentId, user_note: '用户在 V4 单因子验证工作台确认启用。' }); await refresh() }} />}
        {tab === 'strategyLab' && <StrategyLab form={labForm} setForm={setLabForm} running={labRunning} onRun={runLabExperiment} experiments={data.strategyLabExperiments ?? []} versionLabel="V3 Legacy" />}
        {tab === 'strategyLabV4' && <StrategyLab form={v4LabForm} setForm={setV4LabForm} running={v4LabRunning} onRun={runV4LabExperiment} experiments={data.v4StrategyLabExperiments ?? []} versionLabel="V4" />}
      </>}
      <footer>研究展示环境 · 只读数据 · 不提供自动交易或下单功能</footer>
    </main>
  )
}

function StrategyExplorer({ catalog, detail, selectedVersion, onSelect, comparisonVersion, setComparisonVersion, comparisonDetail, onOpenLab }) {
  const summary = detail?.summary
  const period = summary?.research_period
  const periodLabel = period ? `${period.start ?? '—'} to ${period.end ?? '—'}` : '—'
  return <section className="strategy-explorer">
    <section className="panel strategy-hero"><div><span className="label">Code-Audited Metadata · 策略说明中心</span><h2>Strategy Explorer</h2><p>展示后端当前策略配置、公式和数据约束。它不运行策略，也不改动任何正式回测。</p></div><DataTag value="READ ONLY" /></section>
    <div className="strategy-layout">
      <aside className="strategy-version-list panel"><span className="label">Versions</span>{(catalog?.versions ?? []).map((item) => <button key={item.version} className={`strategy-version-button ${selectedVersion === item.version ? 'selected' : ''}`} onClick={() => onSelect(item.version)}><strong>{strategyDisplayLabel(item.version, item.label)}</strong><small>{item.classification}</small><DataTag value={item.status.toUpperCase()} /></button>)}</aside>
      <div className="strategy-content">
        {!detail ? <section className="panel loading">正在读取策略元数据…</section> : <>
          <section className="panel strategy-summary"><div className="panel-head"><div><span className="label">Strategy Summary</span><h2>{strategyDisplayLabel(detail.version, detail.label)}</h2></div><div className="tag-group"><DataTag value={detail.status.toUpperCase()} /><DataTag value={detail.classification.toUpperCase()} /></div></div><div className="strategy-kv-grid"><KV label="Version" value={detail.version === 'v4_entry_timing_candidate' ? 'V4 Factor Validation' : detail.version} /><KV label="研究区间" value={periodLabel} /><KV label="股票池" value={summary.universe} /><KV label="Top N" value={`Top ${summary.top_n}`} /><KV label="调仓" value={summary.rebalance_frequency} /><KV label="执行" value={summary.execution_timing} /><KV label="费用" value={`${pct(summary.fee_rate * 100)} · 滑点 ${pct(summary.slippage_rate * 100)}`} /><KV label="完整性" value={summary.integrity_status} /><KV label="PIT 股票池" value={summary.pit_universe ? 'Enabled' : 'Unavailable'} /><KV label="PIT 基本面" value={summary.pit_fundamentals ? 'Enabled' : 'Unavailable'} /></div></section>
          <section className="panel"><div className="panel-head"><div><span className="label">Actual Strategy Flow</span><h2>策略流程</h2></div></div><div className="strategy-flow">{detail.flow.map((step, index) => <div key={step} className="flow-step"><span>{index + 1}</span><strong>{step}</strong></div>)}</div><p className="section-note">{detail.timeline}</p></section>
          <section className="panel"><div className="panel-head"><div><span className="label">Factor Overview</span><h2>因子与权重</h2></div><p className="section-note">总分是横截面排序分，不等于股票上涨概率。</p></div><div className="table-wrap"><table><thead><tr><th>因子</th><th>类型</th><th>权重</th><th>职责</th><th>实现状态</th><th>证据状态</th></tr></thead><tbody>{detail.factors.map((factor) => <tr key={factor.id}><td><strong>{factor.display_name}</strong><small>{factor.name}</small></td><td>{factor.type}</td><td>{pct(factor.weight * 100)}</td><td>{factor.role}</td><td><DataTag value={factor.status.toUpperCase()} /></td><td><DataTag value={(factor.evidence_status ?? 'not_validated').toUpperCase()} /></td></tr>)}</tbody></table></div></section>
          <section className="strategy-factor-details">{detail.factors.map((factor) => <details className="panel factor-detail" key={factor.id}><summary><span><strong>{factor.display_name}</strong><small>{factor.name} · {factor.type}</small></span><span>{pct(factor.weight * 100)}</span></summary><div className="factor-detail-grid"><InfoBlock title="用途" value={factor.purpose} /><InfoBlock title="数据来源" value={factor.data_sources.join(' · ')} /><InfoBlock title="输入" value={factor.inputs.join(' · ')} /><InfoBlock title="评分范围" value={factor.score_range} /></div><InfoBlock title="实际计算规则" value={factor.formula} /><InfoBlock title="重要行为" value={factor.important_behavior ?? '—'} />{factor.components?.length > 0 && <div className="table-wrap"><table className="compact"><thead><tr><th>基本面子因子</th><th>内部权重</th></tr></thead><tbody>{factor.components.map((component) => <tr key={component.name}><td>{component.name}</td><td>{pct(component.weight * 100)}</td></tr>)}</tbody></table></div>}<div className="factor-detail-actions"><button className="secondary-button" onClick={() => onOpenLab(factor.id)}>在 V4 策略实验室研究此因子</button><small>仅预填因子开关，不会自动运行实验。</small></div><p className="code-ref">代码位置：{factor.code_refs.join(' · ')}</p></details>)}</section>
          <section className="panel strategy-split"><div><span className="label">Market Soft Factor</span><h2>市场共同分</h2><p>权重：{pct(detail.factors.find((factor) => factor.id === 'market_regime')?.weight * 100)}。{detail.factors.find((factor) => factor.id === 'market_regime')?.score_display}</p><p>{detail.factors.find((factor) => factor.id === 'market_regime')?.important_behavior}</p></div><div><span className="label">Market Regime Hard Gate</span><h2>{detail.market_hard_gate.enabled ? 'Hard Gate On' : 'Hard Gate Off'}</h2><p>触发：{detail.market_hard_gate.trigger}。动作：{detail.market_hard_gate.action}。阈值：{detail.market_hard_gate.threshold}。</p></div></section>
          <section className="panel"><div className="panel-head"><div><span className="label">Risk and Ranking</span><h2>过滤与排序</h2></div></div><p className="formula-box">{detail.ranking.formula}</p><div className="table-wrap"><table><thead><tr><th>过滤项</th><th>规则</th><th>效果</th><th>来源</th></tr></thead><tbody>{detail.risk_filters.map((item) => <tr key={item.name}><td><strong>{item.name}</strong></td><td>{item.rule}</td><td>{item.effect}</td><td>{item.source}</td></tr>)}</tbody></table></div><p className="section-note">排序：{detail.ranking.method}；并列处理：{detail.ranking.tie_handling}</p></section>
          <section className="panel strategy-execution"><div><span className="label">Execution Logic</span><h2>调仓与执行</h2><InfoBlock title="Signal Date" value={detail.execution.signal_date} /><InfoBlock title="Execution Date" value={detail.execution.execution_date} /></div><div><InfoBlock title="Current Portfolio Handling" value={detail.execution.portfolio_handling} /><InfoBlock title="Transaction Cost Logic" value={detail.execution.transaction_cost} /></div></section>
          <section className="panel"><div className="panel-head"><div><span className="label">Data & Integrity</span><h2>数据与完整性</h2></div><DataTag value={detail.data_integrity.validation_status.toUpperCase()} /></div><div className="strategy-kv-grid"><KV label="历史股票池" value={detail.data_integrity.historical_universe} /><KV label="历史时点基本面" value={detail.data_integrity.pit_fundamentals} /><KV label="future_data_count" value={detail.data_integrity.future_data_count ?? 'Not published for this research version'} /><KV label="价格复权" value={detail.data_integrity.price_adjustment} /><KV label="公司行动" value={detail.data_integrity.corporate_action_handling} /><KV label="数据覆盖" value={detail.data_integrity.data_coverage} /><KV label="数据版本" value={detail.data_integrity.data_version ?? 'Unavailable'} /><KV label="验证状态" value={detail.data_integrity.validation_status} /></div></section>
          {detail.research_variants?.length > 0 && <section className="panel"><span className="label">Research Variants</span><h2>可研究的 TopN 变体</h2><p className="section-note">{detail.research_variants.map((value) => `Top${value}`).join(' · ')}。它们是 V4 实验室的可选研究配置，不是当前正式 TopN。</p></section>}
          <section className="panel strategy-limitations"><div><span className="label">Event Factor</span><h2>事件信息</h2><p>{detail.event_factor.rule}</p><DataTag value={detail.event_factor.status.toUpperCase()} /></div><div><span className="label">Known Limitations / Research Conclusions</span><ul>{detail.known_limitations.map((item) => <li key={item}>{item}</li>)}</ul></div></section>
          <section className="panel"><div className="panel-head"><div><span className="label">Version Diff</span><h2>与上一版本相比</h2></div></div><p className="section-note">上一版本：{detail.previous_version ?? '—'}</p><ul className="research-list">{detail.version_diff.map((item) => <li key={item}>{item}</li>)}</ul></section>
          <section className="panel strategy-compare"><div className="panel-head"><div><span className="label">Version Comparison</span><h2>版本并排比较</h2></div><select value={comparisonVersion} onChange={(event) => setComparisonVersion(event.target.value)} aria-label="选择比较版本"><option value="">选择另一版本</option>{(catalog?.versions ?? []).filter((item) => item.version !== detail.version).map((item) => <option key={item.version} value={item.version}>{item.label}</option>)}</select></div>{comparisonDetail ? <StrategyCompare left={detail} right={comparisonDetail} /> : <p className="empty">选择一个版本后，比较策略结构和执行逻辑，不比较或暗示收益优劣。</p>}</section>
        </>}
      </div>
    </div>
  </section>
}

function KV({ label, value }) { return <div className="strategy-kv"><span>{label}</span><strong>{value ?? '—'}</strong></div> }
function InfoBlock({ title, value }) { return <div className="info-block"><span className="label">{title}</span><p>{value || '—'}</p></div> }
function strategyDisplayLabel(version, fallback) { return version === 'v4_entry_timing_candidate' ? 'V4 单因子有效性验证' : fallback }
function StrategyCompare({ left, right }) {
  const rows = [
    ['状态', left.status, right.status], ['股票池', left.summary.universe, right.summary.universe],
    ['研究区间', `${left.summary.research_period.start} to ${left.summary.research_period.end}`, `${right.summary.research_period.start} to ${right.summary.research_period.end}`],
    ['TopN', `Top ${left.summary.top_n}`, `Top ${right.summary.top_n}`], ['调仓', left.summary.rebalance_frequency, right.summary.rebalance_frequency],
    ['执行', left.summary.execution_timing, right.summary.execution_timing], ['Market Hard Gate', left.market_hard_gate.enabled ? `${left.market_hard_gate.trigger}; ${left.market_hard_gate.action}` : 'Off', right.market_hard_gate.enabled ? `${right.market_hard_gate.trigger}; ${right.market_hard_gate.action}` : 'Off'],
    ['技术/入场定义', left.factors.find((item) => item.id === 'technical')?.name, right.factors.find((item) => item.id === 'technical')?.name],
    ['数据完整性', left.data_integrity.validation_status, right.data_integrity.validation_status],
  ]
  return <div className="table-wrap"><table><thead><tr><th>项目</th><th>{strategyDisplayLabel(left.version, left.label)}</th><th>{strategyDisplayLabel(right.version, right.label)}</th></tr></thead><tbody>{rows.map(([name, one, two]) => <tr key={name}><td><strong>{name}</strong></td><td>{one}</td><td>{two}</td></tr>)}</tbody></table></div>
}

function Dashboard({ market, ranking = [], backtest, versions, v4, experiments = [], equityA = [], equityB = [], onNavigate }) {
  const v1 = backtest?.models?.[0]
  const v2 = (versions?.official_versions ?? []).find((item) => item.version === 'v2_continuous_rebalance')
  const cacheDate = market?.as_of ?? ranking[0]?.as_of ?? null
  const recentExperiments = experiments.slice(0, 3)
  return <section className="page-grid overview-grid">
    <section className="overview-status panel span-12">
      <div><span className="label">系统状态</span><h2>版本与数据边界</h2></div>
      <div className="status-cells">
        <StatusCell label="当前正式版本" value={versions?.official_current_version ?? 'unavailable'} tag="VALIDATED" title="正式历史回测版本，不等于实时策略。" />
        <StatusCell label="当前研究版本" value={v4?.research_version ?? 'unavailable'} tag="RESEARCH" title="研究候选版本，结果不自动成为正式策略。" />
        <StatusCell label="数据版本" value={versions?.data_version ?? 'unavailable'} tag="INCOMPLETE" title="后端尚未提供统一 data_version，前端不会伪造。" />
        <StatusCell label="本地缓存截至" value={cacheDate ?? 'unavailable'} tag="CACHED" title="本地市场数据缓存的最新日期，不是实时行情时间。" />
      </div>
    </section>

    <section className="market-observation panel span-12">
      <div className="panel-head"><div><span className="label">Current Cached Observation · 当前缓存观察</span><h2>沪深300缓存市场状态</h2></div><DataTag value="CACHED" /></div>
      <div className="market-strip"><div><strong className={`regime ${market?.regime}`}>{market?.regime ?? 'neutral'}</strong><p>{market?.label ?? '等待数据'}</p></div><Score label="沪深300风险市场分" value={market?.risk_score} /><Score label="沪深300趋势市场分" value={market?.trend_score} /><small>来源：{market?.source ?? '—'} · 缓存截至：{market?.as_of ?? '—'}</small></div>
      <p className="section-note">此区域基于当前本地市场数据缓存计算，不代表历史回测结果，也不代表实时券商行情。</p>
    </section>

    <section className="official-backtest span-12">
      <div className="section-title"><div><span className="label">Historical Backtests · 正式历史回测</span><h2>正式历史回测</h2></div><div className="tag-group"><DataTag value="HISTORICAL" /><DataTag value="VALIDATED" /></div></div>
      <section className="official-current panel"><div className="panel-head"><div><span className="label">Current Official Historical Baseline</span><h2>V2 Continuous Rebalance</h2></div><div className="tag-group"><DataTag value="VALIDATED" /><DataTag value="HISTORICAL" /></div></div><p className="section-note">当前正式历史基准。它修复了 V1 的持有期与调仓期重复等待问题；这仍是历史回测，不是当前实盘表现。</p><div className="metric-grid"><Metric title="CAGR" value={pct(v2?.cagr_pct)} subtitle="V2 · 2020-01-01 → 2025-12-31" /><Metric title="最大回撤" value={pct(v2?.max_drawdown_pct)} subtitle="V2 Continuous Rebalance" tone="warning" /><Metric title="夏普比率" value={number(v2?.sharpe_ratio)} subtitle="V2 · 历史风险调整收益" /><Metric title="状态" value="Validated" subtitle="历史股票池与 PIT 基本面" /></div></section>
      <section className="panel"><div className="panel-head"><div><span className="label">V1 Historical PIT · 历史冻结版本</span><h2>2020-01-01 → 2025-12-31</h2></div><DataTag value="VALIDATED" /></div><p className="section-note">这是冻结历史版本的回测结果，不代表当前研究版本表现。</p><MetricCards metrics={v1} /></section>
      <details className="legacy-warning"><summary>Why is V1 not the current research baseline?</summary><p>V1 曾将 holding 与 rebalance 排程串行执行，产生非预期现金等待日；连续调仓时间轴已在 V2 修复。V1 保留用于历史对照，不应作为当前策略能力的最终判断。</p></details>
      <section className="panel chart-panel"><div className="panel-head"><div><span className="label">V1 Historical Backtest</span><h2>V1 历史回测净值 vs Benchmark</h2></div><span className="muted">累计收益%</span></div><Chart option={equityOption(equityA, equityB, { modelA: 'V1 Model A', modelB: 'V1 Model B', benchmark: 'CSI300' })} /></section>
      <section className="panel"><div className="panel-head"><div><span className="label">版本对比</span><h2>正式与研究版本目录</h2></div><span className="muted">研究版本不作为正式策略收益</span></div><VersionTable official={versions?.official_versions ?? []} research={versions?.research_versions ?? []} /></section>
    </section>

    <section className="research-summary panel span-12"><div className="panel-head"><div><span className="label">Research Only · 当前研究</span><h2>V4 单因子有效性验证</h2></div><div className="tag-group"><DataTag value="RESEARCH" /><DataTag value={v4?.integrity_status === 'validated' ? 'VALIDATED' : 'INCOMPLETE'} /></div></div><div className="research-summary-grid"><div><strong>V4 Single-Factor Validation</strong><p>目标：分别验证五类因子的独立有效性；Entry Timing 只是其中一个研究因子。该区域展示历史研究，不是正式策略表现。</p></div><div><span className="label">Market</span><p>时间序列市场状态研究：未来收益、波动、回撤与市场择时效果。</p></div><div><span className="label">Cross-sectional Factors</span><p>Momentum、Money Flow、Fundamental、Entry Timing：TopN、IC、Rank IC、分组收益与跨时期稳定性。</p></div></div><div className="overview-actions"><button className="primary-button" onClick={() => onNavigate('researchV4')}>进入单因子验证</button><button className="secondary-button" onClick={() => onNavigate('strategyLabV4')}>进入 V4 策略实验室</button></div></section>

    <section className="panel span-12"><div className="panel-head"><div><span className="label">Experiment · Historical Research</span><h2>最近策略实验</h2></div><button className="text-button" onClick={() => onNavigate('strategyLabV4')}>查看全部</button></div><RecentExperiments items={recentExperiments} /></section>

    <section className="panel rank-panel span-12"><div className="panel-head"><div><span className="label">Current Cached Cross-Section Ranking · 当前缓存股票评分</span><h2>本地缓存横截面评分</h2></div><div className="tag-group"><DataTag value="CACHED" /><span className="muted">Score Date: {ranking[0]?.as_of ?? '—'} · Legacy Technical</span></div></div><p className="section-note">该排名使用本地缓存中所有入选股票共同可用的最新日期计算；它可能早于上方市场缓存日期，不是历史回测某个调仓日持仓，也不是正式实盘推荐。</p><RankingTable ranking={ranking.slice(0, 8)} compact technicalLabel="Legacy Technical" /></section>
  </section>
}

function DataTag({ value }) { return <span className={`tag data-tag ${String(value).toLowerCase()}`}>{value}</span> }
function StatusCell({ label, value, tag, title }) { return <article title={title}><span>{label}</span><strong>{value}</strong><DataTag value={tag} /></article> }
function VersionTable({ official, research }) { const rows = [...official, ...research]; return <div className="table-wrap"><table><thead><tr><th>Version</th><th>Period</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Status</th></tr></thead><tbody>{rows.map((item) => <tr key={item.version}><td><strong>{item.label}</strong><small>{item.scope}</small></td><td>{item.period}</td><td>{pct(item.cagr_pct)}</td><td>{number(item.sharpe_ratio)}</td><td>{pct(item.max_drawdown_pct)}</td><td><DataTag value={item.status.toUpperCase()} /></td></tr>)}</tbody></table></div> }
function RecentExperiments({ items }) { return items.length ? <div className="table-wrap"><table><thead><tr><th>实验</th><th>版本</th><th>配置</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>性质</th></tr></thead><tbody>{items.map((item) => <tr key={item.experiment_id}><td><strong>{item.experiment_id.slice(-8)}</strong><small>{item.created_at ? new Date(item.created_at).toLocaleString() : '—'}</small></td><td>{item.technical_variant === 'entry_timing' ? 'V4 单因子有效性验证' : 'Research'}</td><td>Top {item.top_n ?? '—'} · Hard Gate {item.market_regime_gate ? 'ON' : 'OFF'}</td><td>{pct(item.metrics?.cagr_pct)}</td><td>{number(item.metrics?.sharpe_ratio)}</td><td>{pct(item.metrics?.max_drawdown_pct)}</td><td><DataTag value="EXPERIMENT" /></td></tr>)}</tbody></table></div> : <p className="empty">暂无已保存的 V4 研究实验。</p> }

function Factors({ ranking = [], selected, selectedCode, setSelectedCode }) {
  return <section className="page-grid">
    <section className="panel span-7"><div className="panel-head"><div><span className="label">Cached Cross-Section · 横向评分</span><h2>当前缓存股票排名</h2></div><span className="muted">Score Date: {ranking[0]?.as_of ?? '—'} · Legacy Technical</span></div><RankingTable ranking={ranking} selectedCode={selectedCode} onSelect={setSelectedCode} technicalLabel="Legacy Technical" /></section>
    <aside className="factor-side span-5"><section className="panel"><div className="panel-head"><div><span className="label">因子画像</span><h2>{selected ? `${selected.stock_name} ${selected.stock_code}` : '暂无数据'}</h2></div><strong className="score">{selected ? number(selected.total_score) : '—'}</strong></div><Chart option={radarOption(selected, 'Legacy Technical')} /></section><section className="panel"><div className="panel-head"><div><span className="label">评分拆解</span><h2>因子贡献</h2></div></div><Chart option={contributionOption(selected, 'Legacy Technical')} className="short" /></section></aside>
  </section>
}

function Backtest({ result, equityA = [], equityB = [] }) {
  const models = result?.models ?? []
  const allValidated = models.length > 0 && models.every((item) => item.backtest_integrity === 'point_in_time_validated')
  return <section className="page-grid">
    <section className={`integrity-panel span-12 ${allValidated ? 'validated' : 'incomplete'}`}>
      <div className="integrity-summary">
        <span className="label">回测数据可信度</span>
        <strong>{allValidated ? '历史时点验证已通过' : '当前结果尚未通过完整验证'}</strong>
        <p>{allValidated ? '历史股票池与历史时点基本面检查均已通过。' : '旧回测仍可用于查看流程，但不能作为正式策略结论。请重新运行严格历史回测。'}</p>
      </div>
      <div className="integrity-models">
        {models.map((item) => <div className="integrity-model" key={`integrity-${item.model}`}>
          <div className="integrity-title"><strong>{item.model}</strong><span className={`tag ${item.backtest_integrity === 'point_in_time_validated' ? 'pass' : 'fail'}`}>{item.backtest_integrity === 'point_in_time_validated' ? '验证通过' : '不完整'}</span></div>
          <IntegrityCheck label="历史股票池" passed={item.historical_universe_verified} />
          <IntegrityCheck label="历史时点基本面" passed={item.fundamental_point_in_time_verified} />
          {item.note && <small>{item.note}</small>}
        </div>)}
      </div>
    </section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">策略对照</span><h2>模型 A 与模型 B</h2></div><span className="muted">B 仅允许政策/公告辅助增强</span></div><div className="metric-row">{models.map((item) => <div className="strategy-metric" key={item.model}><span>{item.model}</span><strong>{pct(item.total_return_pct)}</strong><small>年化 {pct(item.annualized_return_pct)} · 回撤 {pct(item.max_drawdown_pct)} · 夏普 {number(item.sharpe_ratio)}</small></div>)}</div></section>
    <section className="panel chart-panel span-8"><div className="panel-head"><div><span className="label">V1 Historical Backtest · Frozen</span><h2>V1 历史回测净值 vs Benchmark</h2></div></div><Chart option={equityOption(equityA, equityB, { modelA: 'V1 Model A', modelB: 'V1 Model B', benchmark: 'CSI300' })} /></section>
    <section className="panel span-4"><span className="label">事件增量检验</span><h2>模型 B - 模型 A</h2><div className="delta-list"><Score label="累计收益变化" value={result?.comparison?.total_return_diff_pct} suffix="%"/><Score label="年化收益变化" value={result?.comparison?.annualized_return_diff_pct} suffix="%"/><Score label="夏普变化" value={result?.comparison?.sharpe_diff}/></div><p className="muted block-note">{result?.note}</p></section>
  </section>
}

function IntegrityCheck({ label, passed }) {
  return <div className="integrity-check"><span>{label}</span><strong className={passed ? 'pass' : 'fail'}>{passed ? '已通过' : '未通过'}</strong></div>
}

function Simulation({ trades = [], summary = {} }) {
  return <section className="panel full-panel"><div className="panel-head"><div><span className="label">模拟组合记录</span><h2>模拟交易与持仓状态</h2></div><span className="muted">仅展示真实开仓或已产生交易意图的记录</span></div>{trades.length ? <div className="table-wrap"><table><thead><tr><th>时间</th><th>股票</th><th>事件/来源</th><th>状态</th><th>开仓价</th><th>收益</th><th>研究说明</th></tr></thead><tbody>{trades.map((item, index) => <tr key={`${item.timestamp}-${index}`}><td>{item.timestamp}</td><td>{item.stock_code || '—'}</td><td><strong>{item.signal_type || '研究记录'}</strong><small>{item.signal_source}</small></td><td><span className="tag neutral">{item.status}</span></td><td>{item.entry_price ?? '—'}</td><td>{item.pnl_pct ?? '—'}</td><td className="truncate">{item.note}</td></tr>)}</tbody></table></div> : <div className="simulation-empty"><span className="label">当前状态</span><strong>当前没有模拟持仓</strong><p>{summary.message ?? '尚未产生满足多因子条件并通过风险过滤的模拟交易。'}</p><div className="simulation-stats"><span>研究观察 <b>{summary.research_only_records ?? 0}</b></span><span>行情缺失 <b>{summary.market_data_missing_records ?? 0}</b></span><span>活动持仓 <b>{summary.active_records ?? 0}</b></span><span>已完成 <b>{summary.completed_records ?? 0}</b></span></div></div>}</section>
}

function Events({ events = [], syncing, onSync }) {
  const policy = events.filter((item) => item.event_type === 'policy')
  const announcements = events.filter((item) => item.event_type === 'announcement')
  return <section className="page-grid"><section className="research-hero span-12"><div><span className="label">Official Sources Only</span><h2>市场事件研究</h2><p>同步来源：巨潮资讯/交易所披露，以及国务院、发改委、工信部、财政部、证监会官网政策栏目。每条官方记录保存发布机构、发布时间和原始链接。</p></div><button className="primary-button" disabled={syncing} onClick={onSync}>{syncing ? '正在同步官方来源…' : '同步官方事件'}</button></section><EventColumn title="政策事件" items={policy} className="span-6"/><EventColumn title="公司公告" items={announcements} className="span-6"/><section className="notice span-12">市场事件仅用于研究记录和未来增量验证，不作为当前策略的直接交易触发条件。来源抓取失败或缺少发布日期时不会生成事件记录。</section></section>
}

function V4FactorValidation({ status, priceGaps, recipes, experiments = [], marketAudit, busy, onRefresh, onRun, onMarketAudit, onApprove, onActivate }) {
  const [factor, setFactor] = useState('momentum')
  const active = recipes?.active_recipes?.[factor]
  const [draft, setDraft] = useState(null)
  const [hypothesis, setHypothesis] = useState('')
  const [period, setPeriod] = useState('full')
  const [topValues, setTopValues] = useState([5, 10, 20])
  useEffect(() => { setDraft(active ? { components: active.components.map((item) => ({ ...item })), parameters: active.parameters ?? {} } : null) }, [factor, active?.recipe_id])
  const ranges = { '2015-2019': ['2015-01-01', '2019-12-31'], '2020-2025': ['2020-01-01', '2025-12-31'], full: ['2015-01-01', '2025-12-31'] }
  const factorExperiments = experiments.filter((item) => item.factor_name === factor)
  const best = [...factorExperiments].sort((a, b) => Number(b.results?.[0]?.metrics?.sharpe_ratio ?? -Infinity) - Number(a.results?.[0]?.metrics?.sharpe_ratio ?? -Infinity))[0]
  async function run() {
    if (!active || !draft || !hypothesis.trim()) return
    const recipe = await api.createV4Recipe(factor, { ...draft })
    const [start_date, end_date] = ranges[period]
    await onRun({ factor_name: factor, recipe_id: recipe.recipe_id, hypothesis_note: hypothesis, start_date, end_date, top_n_values: topValues, market_regime_gate: false })
  }
  const progress = status?.progress ?? {}
  const expected = progress.expected_codes ?? 0
  const percent = (value) => expected ? Math.min(100, value / expected * 100).toFixed(1) : '0.0'
  const rows = [
    ['Market', 'Time-series / 市场状态', 'Under Validation', 'Future Return · Future Volatility · Future Drawdown · Market Timing Effectiveness'],
    ['Momentum', 'Cross-sectional / 选股', 'Under Validation', 'Top5 / Top10 / Top20 · IC · Rank IC · Quantile Return · 年度稳定性'],
    ['Money Flow', 'Cross-sectional / 选股', 'Under Validation', 'Top5 / Top10 / Top20 · IC · Rank IC · Quantile Return · 年度稳定性'],
    ['Fundamental', 'Cross-sectional / 选股', 'Under Validation', 'Top5 / Top10 / Top20 · IC · Rank IC · Quantile Return · 年度稳定性'],
    ['Entry Timing', 'Cross-sectional / 入场位置', 'Under Validation', 'Top5 / Top10 / Top20 · IC · Rank IC · Quantile Return · 年度稳定性'],
  ]
  const currentResults = factorExperiments.flatMap((item) => (item.results ?? []).map((result) => ({ ...result, experiment: item })))
  const integrityLabel = { validated: '已验证', passed: '已通过', pending_validation: '待完整验证', issues_found: '发现问题', building: '构建中' }
  const integrityClass = (value) => ['validated', 'passed'].includes(value) ? 'pass' : value === 'building' || value === 'pending_validation' ? 'accent' : 'fail'
  return <section className="page-grid">
    <section className="research-hero span-12"><div><span className="label">V4 Single-Factor Validation</span><h2>V4 单因子有效性验证</h2><p>分别验证 Market、Momentum、Money Flow、Fundamental 与 Entry Timing 五类因子的独立有效性；实验会保存完整配方，不自动修改 V4 当前配置。</p></div><div className="status-stack"><span className="tag accent">Research Only</span><span className={`tag ${integrityClass(status?.data_integrity_status)}`}>数据完整性：{status?.data_integrity_status === 'partial' ? '部分通过' : integrityLabel[status?.data_integrity_status] ?? '待验证'}</span></div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">2015-2025 数据覆盖</span><h2>V4 单因子研究输入</h2></div><span className="muted">研究数据与 V3 Legacy、V1/V2 正式版本保持隔离</span></div><div className="progress-grid"><Progress title="BaoStock HFQ 主收益价格" value={progress.hfq_baostock_price_files ?? 0} expected={expected} percent={percent(progress.hfq_baostock_price_files ?? 0)} /><Progress title="历史时点基本面" value={progress.pit_fundamental_files ?? 0} expected={expected} percent={percent(progress.pit_fundamental_files ?? 0)} /><Progress title="历史流动性成交额" value={progress.liquidity_files ?? 0} expected={expected} percent={percent(progress.liquidity_files ?? 0)} /><Progress title="旧腾讯 HFQ 审计缓存" value={progress.legacy_hfq_audit_files ?? 0} expected={expected} percent={percent(progress.legacy_hfq_audit_files ?? 0)} muted /><Progress title="旧 QFQ 审计缓存" value={progress.legacy_qfq_audit_files ?? 0} expected={expected} percent={percent(progress.legacy_qfq_audit_files ?? 0)} muted /></div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">Five-Factor Validation Status</span><h2>五因子验证状态</h2></div><span className="muted">Entry Timing 是五因子之一，不代表整个 V4。</span></div><div className="table-wrap"><table><thead><tr><th>因子</th><th>类型</th><th>当前状态</th><th>研究方式</th></tr></thead><tbody>{rows.map(([factor, type, state, method]) => <tr key={factor}><td><strong>{factor}</strong></td><td>{type}</td><td><DataTag value={state.toUpperCase()} /></td><td>{method}</td></tr>)}</tbody></table></div><section className="notice compact-notice"><strong>Market 的边界：</strong>同一调仓日全部股票共享 Market 分数，因此不按普通 TopN 选股因子处理；它单独研究未来市场收益、风险与择时效果。其余四个因子才进行横截面 TopN、IC、Rank IC 与分组收益验证。</section></section>
    <section className="panel span-12 factor-workbench"><div className="panel-head"><div><span className="label">Factor Workbench</span><h2>单因子研究工作台</h2></div><span className="muted">配方修改只会创建研究配方，不会自动更新 V4 当前配置。</span></div><div className="factor-tabs">{[['market_regime','Market'],['momentum','Momentum'],['money_flow','Money Flow'],['fundamental','Fundamental'],['technical','Entry Timing']].map(([id, label]) => <button key={id} className={factor === id ? 'active' : ''} onClick={() => setFactor(id)}>{label}</button>)}</div>{active && draft ? <div className="recipe-workspace"><section><span className="label">当前 V4 配方</span><h3>{active.label} · V4 Active</h3><p className="section-note">{active.recipe_id} · v{active.recipe_version} · {active.recipe_hash}</p><div className="recipe-components">{draft.components.map((item, index) => <label key={item.id} className="recipe-component"><input type="checkbox" checked={Boolean(item.enabled)} onChange={(event) => setDraft({ ...draft, components: draft.components.map((part, itemIndex) => itemIndex === index ? { ...part, enabled: event.target.checked } : part) })} /><span>{item.label}</span><input type="range" min="0" max="100" step="5" value={Math.round(Number(item.weight) * 100)} onChange={(event) => setDraft({ ...draft, components: draft.components.map((part, itemIndex) => itemIndex === index ? { ...part, weight: Number(event.target.value) / 100 } : part) })} /><output>{Math.round(Number(item.weight) * 100)}%</output></label>)}</div></section><aside><label className="field"><span>研究假设</span><textarea rows="3" value={hypothesis} onChange={(event) => setHypothesis(event.target.value)} placeholder="先写下准备证伪或验证的问题" /></label><label className="field"><span>测试区间</span><select value={period} onChange={(event) => setPeriod(event.target.value)}><option value="2015-2019">2015–2019</option><option value="2020-2025">2020–2025</option><option value="full">2015–2025</option></select></label>{factor !== 'market_regime' ? <div className="topn-choice"><span>TopN</span>{[5,10,20].map((value) => <label key={value}><input type="checkbox" checked={topValues.includes(value)} onChange={(event) => setTopValues(event.target.checked ? [...topValues, value].sort((a,b)=>a-b) : topValues.filter((item) => item !== value))} /> Top{value}</label>)}</div> : <p className="notice compact-notice">Market 是时间序列因子；会研究其与未来 CSI300 收益、波动的关系，不参与 TopN 股票排序。</p>}<section className="notice compact-notice">运行横截面实验时，组件分数会从冻结的 HFQ 历史行情和 PIT 基本面缓存重新计算；缺失分数不会被中性值替代。</section><button className="primary-button" disabled={busy || (factor !== 'market_regime' && (!hypothesis.trim() || !topValues.length))} onClick={factor === 'market_regime' ? onMarketAudit : run}>{busy ? '正在运行研究…' : factor === 'market_regime' ? '运行 Market 预测审计' : '运行单因子测试'}</button></aside></div> : <p className="empty">正在加载该因子的当前 V4 配方…</p>}</section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">Experiment Records</span><h2>{active?.label ?? factor} 实验记录</h2></div><span className="muted">已测试 {factorExperiments.length} 个配方；测试次数增加会提高 multiple-testing 风险。</span></div>{factorExperiments.length ? <div className="table-wrap"><table><thead><tr><th>标记</th><th>Experiment</th><th>Recipe</th><th>Period</th><th>TopN</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>认可 / 启用</th></tr></thead><tbody>{factorExperiments.map((item) => { const metric = item.results?.[0]?.metrics ?? {}; return <tr key={item.experiment_id}><td>{best?.experiment_id === item.experiment_id ? '当前测试最佳' : '—'}</td><td><strong>{item.experiment_id.slice(-12)}</strong><small>{item.hypothesis_note}</small></td><td>{item.recipe?.recipe_id}</td><td>{item.sample_period?.start} → {item.sample_period?.end}</td><td>{(item.top_n_values ?? []).map((value) => `Top${value}`).join(' / ')}</td><td>{pct(metric.cagr_pct)}</td><td>{number(metric.sharpe_ratio)}</td><td>{pct(metric.max_drawdown_pct)}</td><td><div className="inline-actions"><button className="secondary-button" onClick={() => onApprove(item.experiment_id, !item.approved, item.approved ? '' : '用户认可该研究结果。')}>{item.approved ? '已认可' : '认可'}</button><button className="secondary-button" disabled={!item.approved} onClick={() => onActivate(factor, item.recipe?.recipe_id, item.experiment_id)}>设为 V4 配方</button></div></td></tr> })}</tbody></table></div> : <p className="empty">尚未运行该因子的单独测试。实验结果将固定保存当时使用的配方，不会被后续配方变化污染。</p>}</section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">Single-Factor Results</span><h2>单因子结果</h2></div><span className={`tag ${currentResults.length ? 'pass' : 'fail'}`}>{currentResults.length ? 'GENERATED' : 'NOT GENERATED'}</span></div>{currentResults.length ? <div className="table-wrap"><table><thead><tr><th>TopN</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Volatility</th><th>Cash Days</th></tr></thead><tbody>{currentResults.map((item, index) => <tr key={`${item.experiment.experiment_id}-${index}`}><td>Top {item.top_n}</td><td>{pct(item.metrics?.cagr_pct)}</td><td>{number(item.metrics?.sharpe_ratio)}</td><td>{number(item.metrics?.sortino_ratio)}</td><td>{pct(item.metrics?.max_drawdown_pct)}</td><td>{pct(item.metrics?.annualized_volatility_pct)}</td><td>{item.metrics?.cash_days ?? '—'}</td></tr>)}</tbody></table></div> : <p className="empty">运行后会显示完整的 Top5 / Top10 / Top20 回放指标。所有结果仅是样本内研究。</p>}</section>
    <section className="panel span-6"><span className="label">IC / Rank IC</span><h2>横截面预测能力</h2><p className="empty">待单因子研究产物生成后展示；仅适用于 Momentum、Money Flow、Fundamental 和 Entry Timing。</p></section>
    <section className="panel span-6"><span className="label">Market Predictive Audit</span><h2>Market 预测与风险审计</h2>{marketAudit ? <div className="research-checks"><article><div><strong>样本与类型</strong><p>{marketAudit.observations} 个调仓期 · 时间序列市场状态，不参与个股 TopN 排名。</p></div></article><article><div><strong>20 日 Rank IC</strong><p>{number(marketAudit.rank_ic_20d)} · 表示 Market 分数与后续 CSI300 20日收益的秩相关。</p></div></article><article><div><strong>Q1 / Q5 后续20日收益</strong><p>{pct((marketAudit.bucket_return_20d?.Q1 ?? 0) * 100)} / {pct((marketAudit.bucket_return_20d?.Q5 ?? 0) * 100)}</p></div></article></div> : <p className="empty">Market 审计尚未生成。</p>}</section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">Current Research Configuration</span><h2>当前研究配置</h2></div><span className="tag accent">Current Factor: Entry Timing</span></div><div className="research-checks"><article><div><strong>权重的含义</strong><p>权重仅用于研究配置展示，不代表正式组合权重优化。单因子实验中，目标股票选择因子为 100%，其他股票选择因子关闭；Market 按时间序列研究方式单独验证。</p></div></article><article><div><strong>基线配置</strong><p>{Object.entries(factorLabels(status?.technical_variant)).map(([name, label]) => `${label} ${Math.round(Number(status?.factor_weights?.[name] || 0) * 100)}%`).join(' · ')}</p></div></article><article><div><strong>组合规则</strong><p>研究基线 Top {status?.top_n ?? '—'} · 每 {status?.rebalance_days ?? '—'} 个交易日调仓；Top5 / Top10 / Top20 / Top50 是后续验证变体。</p></div></article><article><div><strong>结果隔离</strong><p>V4 状态：{status?.result_status ?? 'not_run'}。它不会覆盖 V3 Legacy 快照、V1/V2 正式回测或正式策略配置。</p></div></article></div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">完整性检查</span><h2>V4 数据与基线检查</h2></div><span className="muted">V4 结果与 V3 Legacy 保持隔离</span></div><div className="research-checks">{(status?.checks ?? []).map((item) => <article key={item.name}><div><strong>{item.name}</strong><p>{item.detail}</p></div><span className={`tag ${integrityClass(item.status)}`}>{integrityLabel[item.status] ?? item.status}</span></article>)}</div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">Price Lifecycle Audit V2</span><h2>历史价格生命周期审计</h2></div><span className={`tag ${priceGaps?.summary?.quality_status === 'validated' ? 'pass' : 'fail'}`}>真实缺口 {priceGaps?.summary?.total_gaps ?? 0} · 策略影响 {priceGaps?.summary?.strategy_impacting_gaps ?? 0}</span></div><p className="section-note">审计按股票实际可交易生命周期判断：退市或历史代码终止后的无行情、以及上市前不存在行情，均不再被误报为价格缺口。原始覆盖规则标记 {priceGaps?.summary?.original_flags ?? 0} 条，其中正常终止交易 {priceGaps?.summary?.valid_end_of_trading ?? 0} 条，正常上市起点 {priceGaps?.summary?.valid_listing_start ?? 0} 条，正常停牌 {priceGaps?.summary?.valid_suspension ?? 0} 条。</p>{priceGaps?.items?.length ? <div className="table-wrap"><table><thead><tr><th>代码</th><th>名称</th><th>预期可交易区间</th><th>实际覆盖</th><th>生命周期结论</th><th>说明</th></tr></thead><tbody>{priceGaps.items.map((item) => <tr key={item.stock_code}><td>{item.stock_code}</td><td>{item.stock_name || '—'}</td><td>{item.expected_start} → {item.expected_end}</td><td>{item.first_date} → {item.last_date}</td><td><DataTag value={item.classification} /></td><td>{item.classification_reason}</td></tr>)}</tbody></table></div> : <p className="empty">V2 审计没有发现历史边界记录。</p>}</section>
    <section className="notice span-12">V4 单因子有效性验证回答“每个因子自己有没有效”；<strong>V4 策略实验室</strong>回答“已验证因子组合后，不同配置表现如何”。二者都不连接实盘交易。</section>
  </section>
}

function V3Research({ status, title = 'V3 长样本研究数据完整性' }) {
  const progress = status?.progress ?? {}
  const expected = progress.expected_codes ?? 0
  const percent = (value) => expected ? Math.min(100, value / expected * 100).toFixed(1) : '0.0'
  const technicalName = status?.technical_variant === 'entry_timing' ? 'entry_timing' : 'technical_legacy'
  const v3Labels = factorLabels(status?.technical_variant)
  return <section className="page-grid">
    <section className="research-hero span-12"><div><span className="label">Research Experiment</span><h2>{title}</h2><p>{status?.message}</p></div><div className="status-stack"><span className="tag accent">{status?.research_version ?? 'V3'}</span><span className={`tag ${status?.integrity_status === 'validated' ? 'pass' : 'fail'}`}>{status?.integrity_status === 'validated' ? '预检通过' : '数据构建中'}</span></div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">2015-2025 数据覆盖</span><h2>研究输入构建进度</h2></div><span className="muted">{status?.performance_visible ? '基线已生成，仍限研究用途' : '不展示未验证绩效'}</span></div><div className="progress-grid"><Progress title="BaoStock HFQ 主收益价格" value={progress.hfq_baostock_price_files ?? 0} expected={expected} percent={percent(progress.hfq_baostock_price_files ?? 0)} /><Progress title="历史时点基本面" value={progress.pit_fundamental_files ?? 0} expected={expected} percent={percent(progress.pit_fundamental_files ?? 0)} /><Progress title="历史流动性成交额" value={progress.liquidity_files ?? 0} expected={expected} percent={percent(progress.liquidity_files ?? 0)} /><Progress title="旧腾讯 HFQ 审计缓存" value={progress.legacy_hfq_audit_files ?? 0} expected={expected} percent={percent(progress.legacy_hfq_audit_files ?? 0)} muted /><Progress title="旧 QFQ 审计缓存" value={progress.legacy_qfq_audit_files ?? 0} expected={expected} percent={percent(progress.legacy_qfq_audit_files ?? 0)} muted /></div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">Active Research Configuration</span><h2>当前研究配置</h2></div><span className="tag accent">{technicalName}</span></div><div className="research-checks"><article><div><strong>因子权重</strong><p>{Object.entries(v3Labels).map(([name, label]) => `${label} ${Math.round(Number(status?.factor_weights?.[name] || 0) * 100)}%`).join(' · ')}</p></div></article><article><div><strong>Market Soft Factor</strong><p>`market_regime` 是同一调仓日的共同市场状态分数；它不会单独改变当日股票横截面排名。</p></div></article><article><div><strong>Market Regime Hard Gate</strong><p>{status?.market_regime_gate ? `Hard Gate: ON · 开仓阈值 ${status?.market_min_score ?? '—'}` : 'Hard Gate: OFF · 不因市场状态整体阻止开仓。'}</p></div></article><article><div><strong>组合规则</strong><p>Top {status?.top_n ?? '—'} · 每 {status?.rebalance_days ?? '—'} 个交易日调仓</p></div></article><article><div><strong>结果隔离</strong><p>{status?.research_version === 'v4_entry_timing_candidate' ? `V4 状态：${status?.result_status ?? 'not_run'}。它不会覆盖 V3 Legacy 快照或既有回测结果。` : 'V3 Legacy 快照与既有回测结果保持冻结，不会被后续版本覆盖或重新解释。'}</p></div></article></div></section>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">完整性检查</span><h2>V3 运行前门槛</h2></div><span className="muted">全部通过后才允许运行长样本回测</span></div><div className="research-checks">{(status?.checks ?? []).map((item) => <article key={item.name}><div><strong>{item.name}</strong><p>{item.detail}</p></div><span className={`tag ${item.status === 'passed' ? 'pass' : item.status === 'building' ? 'accent' : 'fail'}`}>{item.status === 'passed' ? '已通过' : item.status === 'building' ? '构建中' : '待审计'}</span></article>)}</div></section>
    <section className="notice span-12">V3 是独立研究实验：不会改写 V1/V2 正式结果、因子权重或市场环境规则。只有历史股票池、HFQ 总回报价格、PIT 基本面和数据审计全部通过后，才会生成回测、因子分析和消融实验结果。</section>
  </section>
}

const factorLabels = (technicalVariant = 'unknown') => ({
  market_regime: 'market_regime',
  momentum: 'momentum',
  money_flow: 'money_flow',
  fundamental: 'fundamental',
  technical: technicalVariant === 'entry_timing' ? 'entry_timing' : 'technical_legacy',
})
const formatWeights = (weights = {}, technicalVariant = 'unknown') => Object.entries(factorLabels(technicalVariant)).map(([name, label]) => `${label} ${Math.round(Number(weights[name] || 0) * 100)}%`).join(' · ')

function StrategyLab({ form, setForm, running, onRun, experiments = [], versionLabel = 'V3' }) {
  if (!form) return <div className="loading">正在读取策略实验配置…</div>
  const weights = form.factor_weights ?? {}
  const snapshotVariant = form.snapshot_technical_variant ?? 'unknown'
  const snapshotSource = form.snapshot_technical_variant_source === 'historical_inference' ? '历史版本登记' : form.snapshot_technical_variant_source === 'snapshot_metadata' ? '快照元数据' : '元数据缺失'
  const snapshotReady = form.snapshot_status !== 'pending_v4_baseline'
  const labels = factorLabels(snapshotVariant)
  const snapshotName = snapshotVariant === 'entry_timing' ? 'entry_timing' : 'technical_legacy'
  const nextName = form.next_v3_default_technical_variant ?? snapshotVariant
  const total = Object.values(weights).reduce((sum, value) => sum + Number(value || 0), 0)
  const updateWeight = (name, value) => setForm({ ...form, factor_weights: { ...weights, [name]: Number(value) / 100 } })
  const toggleFactor = (name, enabled) => updateWeight(name, enabled ? 10 : 0)
  return <section className="page-grid">
    <section className="research-hero span-12"><div><span className="label">Strategy Research Lab</span><h2>{versionLabel} 策略实验室</h2><p>在固定的历史因子快照上设计、登记和比较研究实验。这里不会触发实盘交易，也不会改写其他版本的基线。</p></div><span className="tag accent">样本内快照重放</span></section>
    <section className="notice span-12"><strong>{snapshotReady ? `当前快照：${snapshotName}。` : `等待 ${versionLabel} 基线快照。`}</strong> 技术定义来源：{snapshotSource}。已保存实验和本页滑块均按该定义展示。候选定义：<strong>{nextName}</strong>；必须先生成独立快照，实验室才允许运行，避免跨版本混用。</section>
    <section className="panel span-7"><div className="panel-head"><div><span className="label">实验设计</span><h2>定义要验证的策略假设</h2></div><span className={`tag ${Math.abs(total - 1) < .0001 ? 'pass' : 'fail'}`}>权重 {Math.round(total * 100)}%</span></div>
      <label className="field"><span>实验名称</span><input value={form.name ?? ''} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>
      <label className="field"><span>研究假设</span><textarea rows="3" value={form.hypothesis ?? ''} onChange={(event) => setForm({ ...form, hypothesis: event.target.value })} /></label>
      <div className="factor-controls"><div className="control-head"><span>因子</span><span>启用</span><span>权重</span></div>{Object.keys(labels).map((name) => <div className="factor-control" key={name}><strong>{labels[name]}</strong><input type="checkbox" checked={Number(weights[name] || 0) > 0} onChange={(event) => toggleFactor(name, event.target.checked)} /><input type="range" min="0" max="100" step="5" value={Math.round(Number(weights[name] || 0) * 100)} onChange={(event) => updateWeight(name, event.target.value)} /><output>{Math.round(Number(weights[name] || 0) * 100)}%</output></div>)}</div>
      <div className="notice compact-notice">`market_regime` 是同一调仓日全股票共享的市场状态分数：它会进入研究记录，但不会改变当日股票横截面排名。需要研究开仓暴露时，请使用下方的 `market_regime_gate`。</div>
      <div className="settings-grid"><label className="field"><span>持仓数量</span><select value={form.top_n} onChange={(event) => setForm({ ...form, top_n: Number(event.target.value) })}>{[5, 10, 20, 50].map((value) => <option key={value} value={value}>Top {value}</option>)}</select></label><label className="check-field"><input type="checkbox" checked={Boolean(form.market_regime_gate)} onChange={(event) => setForm({ ...form, market_regime_gate: event.target.checked })} /><span>启用市场环境开仓过滤</span></label>{form.market_regime_gate && <label className="field"><span>环境阈值</span><input type="number" min="0" max="100" value={form.market_min_score} onChange={(event) => setForm({ ...form, market_min_score: Number(event.target.value) })} /></label>}</div>
      <div className="lab-actions"><p>{snapshotReady ? '提交后会自动归一化已启用的权重，并保留本次输入快照、费用假设和完整审计记录。' : 'V4 基线尚未生成，暂不能创建五因子实验。'}</p><button className="primary-button" disabled={!snapshotReady || running || Math.abs(total - 1) > .0001} onClick={onRun}>{running ? '正在运行研究实验…' : snapshotReady ? '运行策略实验' : '等待 V4 基线快照'}</button></div>
    </section>
    <aside className="panel span-5"><span className="label">研究边界</span><h2>这次实验会做什么</h2><ul className="research-list"><li>固定 2015–2025 {versionLabel} 历史因子快照。</li><li>保留历史股票池、历史时点基本面、风险资格、交易费率与复权价格。</li><li>按你的权重重新排名，构建连续调仓的组合收益曲线。</li><li>记录夏普、回撤、换手、费用、现金天数和配置哈希。</li></ul><div className="notice compact-notice">这是一项样本内研究，不是样本外验证，也不是投资或交易建议。结果差同样会保留，它是在帮你淘汰想法。</div></aside>
    <section className="panel span-12"><div className="panel-head"><div><span className="label">实验台账</span><h2>已保存的策略研究</h2></div><span className="muted">最近 {experiments.length} 条</span></div>{experiments.length ? <div className="table-wrap"><table><thead><tr><th>实验</th><th>状态</th><th>配置</th><th>因子配比</th><th>CAGR</th><th>夏普</th><th>最大回撤</th><th>换手</th><th>假设</th></tr></thead><tbody>{experiments.map((item) => <tr key={item.experiment_id}><td><strong>{item.experiment_id.slice(-8)}</strong><small>{new Date(item.created_at).toLocaleString()}</small></td><td><span className={`tag ${item.status === 'completed' ? 'pass' : 'fail'}`}>{item.status}</span></td><td>Top {item.top_n} · {item.market_regime_gate ? '环境过滤' : '无环境门槛'} · {item.technical_variant === 'entry_timing' ? 'entry_timing' : 'technical_legacy'}</td><td className="weight-summary">{formatWeights(item.weights, item.technical_variant)}</td><td>{pct(item.metrics?.cagr_pct)}</td><td>{number(item.metrics?.sharpe_ratio)}</td><td>{pct(item.metrics?.max_drawdown_pct)}</td><td>{pct(item.metrics?.turnover_pct)}</td><td className="truncate">{item.hypothesis}</td></tr>)}</tbody></table></div> : <p className="empty">还没有策略实验。先写一个你想证伪或验证的假设，再运行第一组。</p>}</section>
  </section>
}

function Progress({ title, value, expected, percent, muted = false }) { return <article className={`progress-card ${muted ? 'muted-card' : ''}`}><span>{title}</span><strong>{value}/{expected || '—'}</strong><div className="progress-track"><i style={{ width: `${percent}%` }} /></div><small>{percent}%</small></article> }

function EventColumn({ title, items, className }) { return <section className={`panel ${className}`}><div className="panel-head"><div><span className="label">辅助研究</span><h2>{title}</h2></div><span className="muted">{items.length} 条</span></div><div className="event-list">{items.length ? items.map((item, index) => <article key={`${item.title}-${index}`}><div><span className={`tag ${item.is_official ? 'pass' : 'accent'}`}>{item.is_official ? '官方来源' : item.score.toFixed(0)}</span><time>{item.published_at}</time></div><h3>{item.source_url ? <a href={item.source_url} target="_blank" rel="noreferrer">{item.title}</a> : item.title}</h3><p>{item.stock_code ? `关联股票：${item.stock_code}` : item.industry ? `行业影响：${item.industry}` : '待补充关联范围'} · {item.publisher || item.source}</p>{item.source_url && <a className="source-link" href={item.source_url} target="_blank" rel="noreferrer">查看官方原文</a>}</article>) : <p className="empty">暂无已沉淀的{title}官方研究数据。</p>}</div></section> }

function MetricCards({ metrics, subtitle = 'V1 Model A · Historical Backtest' }) { return <section className="metric-grid"><Metric title="累计收益" value={pct(metrics?.total_return_pct)} subtitle={subtitle} /><Metric title="最大回撤" value={pct(metrics?.max_drawdown_pct)} subtitle={subtitle} tone="warning" /><Metric title="夏普比率" value={number(metrics?.sharpe_ratio)} subtitle={subtitle} /><Metric title="胜率" value={pct((metrics?.win_rate ?? 0) * 100)} subtitle={subtitle} /></section> }
function Metric({ title, value, subtitle, tone = '' }) { return <article className={`metric ${tone}`}><span className="label">{title}</span><strong>{value}</strong><small>{subtitle}</small></article> }
function Score({ label, value, suffix = '' }) { return <div className="score-block"><span>{label}</span><strong>{value == null ? '—' : `${number(value)}${suffix}`}</strong></div> }
function RankingTable({ ranking, compact = false, selectedCode, onSelect, technicalLabel = 'Legacy Technical' }) { return <div className="table-wrap"><table className={compact ? 'compact' : ''}><thead><tr><th>股票</th><th>综合</th><th>动量</th><th>资金</th><th>基本面</th><th>{technicalLabel}</th></tr></thead><tbody>{ranking.map((item) => <tr key={item.stock_code} className={selectedCode === item.stock_code ? 'selected' : ''} onClick={() => onSelect?.(item.stock_code)}><td><strong>{item.stock_name}</strong><small>{item.stock_code}</small></td><td>{number(item.total_score)}</td><td>{number(item.momentum_score)}</td><td>{number(item.money_flow_score)}</td><td>{number(item.fundamental_score)}</td><td>{number(item.technical_score)}</td></tr>)}</tbody></table></div> }

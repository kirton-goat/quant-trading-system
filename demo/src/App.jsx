import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import snapshot from './data/demo_snapshot.json'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const REPOSITORY_URL = 'https://github.com/kirton-goat/quant-trading-system'

const factorDetails = [
  { key: 'market_regime', name: '市场状态', weight: '20%', type: '时间序列 / 共同分数', note: '判断整体市场环境；同一调仓日对所有股票相同，并配合市场环境开仓过滤。' },
  { key: 'momentum', name: '动量', weight: '25%', type: '横截面选股', note: '使用历史价格的中期趋势信息，衡量股票相对强度。' },
  { key: 'money_flow', name: '资金', weight: '20%', type: '横截面选股', note: '以历史成交量、成交额等量价信息确认资金活跃程度。' },
  { key: 'fundamental', name: '基本面', weight: '25%', type: '横截面选股', note: '通过披露时点可见的财务数据构建质量、成长、估值及现金流信号。' },
  { key: 'technical', name: '技术（Legacy Technical）', weight: '10%', type: '横截面选股', note: '冻结 V2 采用的技术评分。该版本存在与动量的趋势信息重叠，后续已作为独立研究问题处理。' },
]

function formatPercent(value, digits = 2) {
  return `${Number(value).toFixed(digits)}%`
}

function useChart(ref, option) {
  useEffect(() => {
    const element = ref.current
    if (!element) return undefined
    const instance = echarts.init(element, undefined, { renderer: 'canvas' })
    instance.setOption(option)
    const resize = () => instance.resize()
    window.addEventListener('resize', resize)
    return () => {
      window.removeEventListener('resize', resize)
      instance.dispose()
    }
  }, [ref, option])
}

function SectionTitle({ eyebrow, title, text }) {
  return (
    <div className="section-title">
      <span>{eyebrow}</span>
      <h2>{title}</h2>
      {text && <p>{text}</p>}
    </div>
  )
}

function App() {
  const equityRef = useRef(null)
  const drawdownRef = useRef(null)
  const dates = snapshot.equity_curve.map((item) => item.date)
  const benchmarkByDate = new Map(snapshot.benchmark_curve.map((item) => [item.date, item.value]))

  const equityOption = {
    animation: false,
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#101c29',
      borderColor: '#2a4358',
      textStyle: { color: '#dce8f2' },
      valueFormatter: (value) => `${(Number(value) - 1).toFixed(2)}%`,
    },
    legend: { top: 0, right: 0, textStyle: { color: '#a7bacb' }, data: ['策略净值', '沪深300'] },
    grid: { top: 44, left: 18, right: 20, bottom: 28, containLabel: true },
    xAxis: { type: 'category', data: dates, boundaryGap: false, axisLine: { lineStyle: { color: '#264052' } }, axisLabel: { color: '#8ba1b4', hideOverlap: true } },
    yAxis: { type: 'value', axisLabel: { color: '#8ba1b4', formatter: (value) => `${((value - 1) * 100).toFixed(0)}%` }, splitLine: { lineStyle: { color: '#1e3546' } } },
    series: [
      { name: '策略净值', type: 'line', data: snapshot.equity_curve.map((item) => item.value), showSymbol: false, smooth: false, lineStyle: { width: 2, color: '#57d4c8' }, areaStyle: { color: 'rgba(87, 212, 200, 0.10)' } },
      { name: '沪深300', type: 'line', data: dates.map((date) => benchmarkByDate.get(date) ?? null), showSymbol: false, connectNulls: true, lineStyle: { width: 1.5, color: '#d9af54', type: 'dashed' } },
    ],
  }

  const drawdownOption = {
    animation: false,
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', backgroundColor: '#101c29', borderColor: '#2a4358', textStyle: { color: '#dce8f2' }, valueFormatter: (value) => `${Number(value).toFixed(2)}%` },
    grid: { top: 20, left: 18, right: 20, bottom: 28, containLabel: true },
    xAxis: { type: 'category', data: dates, boundaryGap: false, axisLine: { lineStyle: { color: '#264052' } }, axisLabel: { color: '#8ba1b4', hideOverlap: true } },
    yAxis: { type: 'value', max: 0, axisLabel: { color: '#8ba1b4', formatter: '{value}%' }, splitLine: { lineStyle: { color: '#1e3546' } } },
    series: [{ name: '回撤', type: 'line', data: snapshot.equity_curve.map((item) => item.drawdown), showSymbol: false, lineStyle: { color: '#ef8a82', width: 1.7 }, areaStyle: { color: 'rgba(239, 138, 130, 0.18)' } }],
  }

  useChart(equityRef, equityOption)
  useChart(drawdownRef, drawdownOption)

  const { metrics, benchmark, strategy, integrity } = snapshot
  return (
    <main>
      <header className="topbar wrap">
        <a className="brand" href="#top" aria-label="返回顶部">
          <img src="/ai-quant-logo.png" alt="AI Quant Research logo" />
          <span><small>AI QUANT RESEARCH</small><strong>量化研究展示</strong></span>
        </a>
        <div className="topbar-actions">
          <span className="static-badge"><i />静态研究快照</span>
          <a className="github-link" href={REPOSITORY_URL} target="_blank" rel="noreferrer">GitHub 项目</a>
        </div>
      </header>

      <section className="hero wrap" id="top">
        <div className="hero-copy">
          <p className="eyebrow">PROJECT SHOWCASE · RESEARCH ONLY</p>
          <h1>A股量化研究<br />与回测系统</h1>
          <p className="hero-lead">面向研究场景的多因子组合回测平台，重点处理历史股票池、财务披露时点与回测时间轴，降低幸存者偏差和未来数据泄漏风险。</p>
          <div className="hero-meta">
            <span>V2 连续调仓基线</span><span>历史时点验证通过</span><span>{strategy.period}</span>
          </div>
        </div>
        <aside className="research-note">
          <span>展示范围</span>
          <strong>预先生成的公开研究快照</strong>
          <p>不连接实时行情、不运行策略、不提供交易或下单功能。</p>
        </aside>
      </section>

      <section className="wrap metrics-grid" aria-label="回测关键指标">
        <article><span>累计收益</span><strong>{formatPercent(metrics.total_return_pct)}</strong><small>策略净值：Model A</small></article>
        <article><span>年化收益</span><strong>{formatPercent(metrics.annualized_return_pct)}</strong><small>2020–2025 历史回测</small></article>
        <article className="risk"><span>最大回撤</span><strong>{formatPercent(metrics.max_drawdown_pct)}</strong><small>历史峰谷回撤</small></article>
        <article><span>Sharpe</span><strong>{metrics.sharpe_ratio.toFixed(2)}</strong><small>日收益率，年化处理</small></article>
      </section>

      <section className="wrap panel curve-panel">
        <div className="panel-head"><div><span>组合回测</span><h2>策略净值与基准</h2></div><p>策略使用历史成分股与历史时点基本面；虚线为沪深300基准。</p></div>
        <div className="chart" ref={equityRef} aria-label="策略净值与沪深300对比曲线" />
        <div className="curve-foot"><span>基准累计收益 <b>{formatPercent(benchmark.total_return_pct)}</b></span><span>相对沪深300超额收益 <b>{formatPercent(benchmark.excess_return_pct)}</b></span><span>年化波动率 <b>{formatPercent(metrics.annualized_volatility_pct)}</b></span></div>
      </section>

      <section className="wrap split-section">
        <div className="panel drawdown-panel"><div className="panel-head"><div><span>风险画像</span><h2>回撤曲线</h2></div><p>最大回撤 {formatPercent(metrics.max_drawdown_pct)}。</p></div><div className="chart compact" ref={drawdownRef} aria-label="策略回撤曲线" /></div>
        <div className="panel strategy-card"><span>策略概要</span><h2>历史时点多因子组合</h2><dl><div><dt>股票池</dt><dd>{strategy.universe}</dd></div><div><dt>调仓规则</dt><dd>{strategy.rebalance}，Top {strategy.top_n}</dd></div><div><dt>执行时点</dt><dd>{strategy.execution}</dd></div><div><dt>交易成本</dt><dd>{(strategy.fee_rate * 100).toFixed(2)}% 单边费率假设</dd></div></dl></div>
      </section>

      <section className="wrap section">
        <SectionTitle eyebrow="SYSTEM ARCHITECTURE" title="从数据到绩效分析的研究链路" text="展示系统的研究流程，而非实时交易流程。" />
        <div className="architecture" aria-label="系统架构流程">
          {['Data', 'Historical Universe', 'PIT Fundamentals', 'Factor / Signal', 'Portfolio Construction', 'Backtest', 'Performance Analysis'].map((item, index) => <div className="flow-node" key={item}><b>{String(index + 1).padStart(2, '0')}</b><span>{item}</span></div>)}
        </div>
      </section>

      <section className="wrap section two-column">
        <div>
          <SectionTitle eyebrow="HISTORICAL UNIVERSE" title="历史股票池，而不是今天的名单" text="回测日只使用当时真实存在、真实可交易的沪深300/中证500成分股，避免用今天仍存续的股票倒推过去。" />
          <div className="rule-list">
            {['历史指数成分股：按回测日期获取成分，不使用当前名单。', '上市日期：剔除上市时间不足 180 个交易日的股票。', '交易状态：过滤 ST、停牌、退市及无法交易标的。', '流动性：以历史成交额进行最低流动性过滤。'].map((rule) => <p key={rule}>{rule}</p>)}
          </div>
        </div>
        <div>
          <SectionTitle eyebrow="POINT-IN-TIME FUNDAMENTALS" title="财报只在披露后生效" text="每条财务数据同时保留报告期、实际披露日和回测日；仅 disclosure_date ≤ signal_date 的数据可参与评分。" />
          <div className="pit-rule"><code>disclosure_date ≤ as_of_date</code><p>例如 2023 年年报在 2024-03-28 披露，则 2024-02-01 的回测不能使用它；披露后才允许进入基本面因子。</p></div>
        </div>
      </section>

      <section className="wrap section">
        <SectionTitle eyebrow="STRATEGY & MODEL" title="冻结 V2 基线的因子结构" text="以下为 v2_continuous_rebalance 的实际冻结配置。总分用于横截面排序，并结合市场环境开仓过滤；不代表上涨概率。" />
        <div className="factor-table" role="table">
          <div className="factor-row factor-head" role="row"><span>因子</span><span>权重</span><span>类型</span><span>职责</span></div>
          {factorDetails.map((factor) => <div className="factor-row" role="row" key={factor.key}><strong>{factor.name}</strong><b>{factor.weight}</b><span>{factor.type}</span><span>{factor.note}</span></div>)}
        </div>
      </section>

      <section className="wrap section integrity-section">
        <SectionTitle eyebrow="DATA INTEGRITY" title="回测可信度控制" text="这些控制降低研究偏差，但不等于回测已经完全复制真实市场。" />
        <div className="integrity-grid">
          <article><b>已控制</b><h3>幸存者偏差</h3><p>使用 historical_point_in_time 股票池；不把当前成分股直接用于历史回测。</p></article>
          <article><b>已控制</b><h3>未来函数</h3><p>PIT 基本面模式为 {integrity.pit_fundamentals}，未来财务数据检测数为 {integrity.future_fundamental_data}。</p></article>
          <article><b>已验证</b><h3>调仓时间轴</h3><p>连续调仓验证通过；计划调仓之间的非预期现金缺口为 {integrity.timeline_validation.unexpected_cash_days_between_successful_rebalances} 天。</p></article>
          <article><b>已纳入</b><h3>基础交易约束</h3><p>包含历史股票池过滤、最低流动性约束及固定手续费假设。</p></article>
        </div>
      </section>

      <section className="wrap section limitations">
        <SectionTitle eyebrow="LIMITATIONS" title="研究结果的边界" />
        <div className="limitations-grid">
          <p>本项目是研究与回测系统，不含实盘接入、自动下单或资金管理功能。</p>
          <p>回测使用固定费率，未完整建模冲击成本、涨跌停成交约束及所有实际执行摩擦。</p>
          <p>数据源覆盖、历史修订和退市资料完整性仍会影响结论，不能将历史绩效视为未来收益承诺。</p>
          <p>政策/公告事件增强在冻结 V2 快照中未产生额外可展示的绩效差异，后续仅作为研究方向验证。</p>
        </div>
      </section>

      <section className="wrap stack-section">
        <div><span>TECH STACK</span><h2>真实使用的技术栈</h2></div>
        <div className="stack-list">{['Python', 'Pandas', 'NumPy', 'FastAPI', 'React', 'Vite', 'ECharts', 'BaoStock', 'AKShare', 'Pytest'].map((item) => <span key={item}>{item}</span>)}</div>
      </section>

      <footer className="wrap"><p>本页面为静态求职展示快照，来源：{snapshot.source.backtest_version} / {snapshot.source.integrity_status}。</p><a href={REPOSITORY_URL} target="_blank" rel="noreferrer">查看项目代码与文档</a></footer>
    </main>
  )
}

export default App

import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'

export function Chart({ option, className = '' }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!ref.current) return undefined
    const instance = echarts.init(ref.current, null, { renderer: 'canvas' })
    instance.setOption(option, true)
    const observer = new ResizeObserver(() => instance.resize())
    observer.observe(ref.current)
    return () => {
      observer.disconnect()
      instance.dispose()
    }
  }, [option])

  return <div ref={ref} className={`chart ${className}`} />
}

const axis = { axisLine: { lineStyle: { color: '#2c3c4c' } }, axisLabel: { color: '#94a8b7' }, splitLine: { lineStyle: { color: '#1c2a37' } } }

export function equityOption(a, b, labels = {}) {
  const dates = a.map((item) => item.date)
  const modelA = labels.modelA ?? '模型A'
  const modelB = labels.modelB ?? '模型B'
  const benchmark = labels.benchmark ?? '沪深300'
  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', backgroundColor: '#101a24', borderColor: '#2b4557', textStyle: { color: '#e7f1f7' } },
    legend: { data: [modelA, modelB, benchmark], textStyle: { color: '#a9bbc7' }, right: 8 },
    grid: { left: 46, right: 22, top: 38, bottom: 34 },
    xAxis: { type: 'category', data: dates, ...axis, axisLabel: { ...axis.axisLabel, hideOverlap: true } },
    yAxis: { type: 'value', name: '累计收益%', ...axis, nameTextStyle: { color: '#94a8b7' } },
    series: [
      { name: modelA, type: 'line', data: a.map((item) => item.return_pct), smooth: true, showSymbol: false, lineStyle: { width: 2, color: '#21c8b5' } },
      { name: modelB, type: 'line', data: b.map((item) => item.return_pct), smooth: true, showSymbol: false, lineStyle: { width: 2, color: '#e0ab46' } },
      { name: benchmark, type: 'line', data: a.map((item) => item.benchmark_return_pct), smooth: true, showSymbol: false, lineStyle: { width: 1.5, type: 'dashed', color: '#78a9ff' } },
    ],
  }
}

export function radarOption(stock, technicalLabel = '技术') {
  if (!stock) return { title: { text: '暂无缓存排名数据', left: 'center', top: 'center', textStyle: { color: '#708798', fontSize: 13 } } }
  const values = [stock.momentum_score, stock.money_flow_score, stock.fundamental_score, stock.technical_score, stock.market_regime_score]
  return {
    backgroundColor: 'transparent',
    tooltip: { backgroundColor: '#101a24', borderColor: '#2b4557', textStyle: { color: '#e7f1f7' } },
    radar: { indicator: ['动量', '资金', '基本面', technicalLabel, '市场'].map((name) => ({ name, max: 100 })), splitArea: { areaStyle: { color: ['#101d29', '#122331'] } }, splitLine: { lineStyle: { color: '#2c4558' } }, axisName: { color: '#a9bbc7' } },
    series: [{ type: 'radar', data: [{ value: values, name: stock.stock_name, areaStyle: { color: 'rgba(33,200,181,.26)' }, lineStyle: { color: '#21c8b5' }, itemStyle: { color: '#21c8b5' } }] }],
  }
}

export function contributionOption(stock, technicalLabel = '技术') {
  const data = stock ? [stock.momentum_score, stock.money_flow_score, stock.fundamental_score, stock.technical_score, stock.market_regime_score] : []
  return {
    backgroundColor: 'transparent', grid: { left: 78, right: 18, top: 12, bottom: 20 },
    xAxis: { type: 'value', max: 100, ...axis },
    yAxis: { type: 'category', data: ['动量', '资金', '基本面', technicalLabel, '市场'], ...axis },
    series: [{ type: 'bar', data, barWidth: 14, itemStyle: { color: '#4c8dff', borderRadius: 2 } }],
  }
}

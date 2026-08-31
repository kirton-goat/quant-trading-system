const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000/api'

async function get(path) {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) throw new Error(`API ${response.status}`)
  return response.json()
}

export const api = {
  health: () => get('/health'),
  market: () => get('/market/status'),
  ranking: () => get('/stocks/ranking?limit=30'),
  backtest: () => get('/backtest/results'),
  backtestVersions: () => get('/backtest/versions'),
  strategyVersions: () => get('/strategy/versions'),
  strategyVersion: (version) => get(`/strategy/versions/${encodeURIComponent(version)}`),
  v3Research: () => get('/research/v3/status'),
  v4Research: () => get('/research/v4/status'),
  v4PriceGaps: () => get('/research/v4/price-gaps'),
  v4FactorRecipes: () => get('/research/v4/factor-recipes'),
  v4FactorExperiments: (factor = '') => get(`/research/v4/factor-experiments${factor ? `?factor=${encodeURIComponent(factor)}` : ''}`),
  v4MarketPredictiveAudit: () => get('/research/v4/market-predictive-audit'),
  createV4Recipe: (factor, payload) => post(`/research/v4/factor-recipes/${factor}`, payload),
  v4RecipeHistory: (factor) => get(`/research/v4/factor-recipes/${encodeURIComponent(factor)}`),
  activateV4Recipe: (factor, recipeId, payload) => post(`/research/v4/factor-recipes/${encodeURIComponent(factor)}/${encodeURIComponent(recipeId)}/activate`, payload),
  runV4FactorExperiment: (payload) => post('/research/v4/factor-experiments', payload),
  approveV4FactorExperiment: (id, payload) => post(`/research/v4/factor-experiments/${encodeURIComponent(id)}/approval`, payload),
  strategyLabDefault: () => get('/research/strategy-lab/default'),
  strategyLabExperiments: () => get('/research/strategy-lab/experiments'),
  v4StrategyLabDefault: () => get('/research/v4/strategy-lab/default'),
  v4StrategyLabExperiments: () => get('/research/v4/strategy-lab/experiments'),
  runStrategyLab: (payload) => fetch(`${API_BASE}/research/strategy-lab/experiments`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }).then(async (response) => {
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.detail || '实验提交失败')
    return payload
  }),
  runV4StrategyLab: (payload) => fetch(`${API_BASE}/research/v4/strategy-lab/experiments`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
  }).then(async (response) => {
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.detail || 'V4 实验提交失败')
    return payload
  }),
  equity: (model) => get(`/backtest/equity?model=${model}`),
  trades: () => get('/trades/history?limit=200'),
  tradeSummary: () => get('/trades/summary'),
  events: () => get('/events?type=all&limit=100'),
  syncOfficialEvents: () => fetch(`${API_BASE}/events/sync-official`, { method: 'POST' }).then(async (response) => {
    const payload = await response.json()
    if (!response.ok) throw new Error(payload.detail || '官方事件同步失败')
    return payload
  }),
}

async function post(path, body) {
  const response = await fetch(`${API_BASE}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const payload = await response.json()
  if (!response.ok) throw new Error(payload.detail || `API ${response.status}`)
  return payload
}

# 静态求职展示 Demo

该目录是项目的独立静态展示层，可部署为普通静态网站。它不请求 FastAPI、不读取本地缓存、不连接实时行情，也不会执行回测或交易操作。

`src/data/demo_snapshot.json` 是经过筛选的只读展示快照，使用冻结的 `v2_continuous_rebalance` / Model A 正式结果：

- 回测区间：2020-01-01 至 2025-12-31
- 股票池：沪深300与中证500历史成分股
- 核心指标：累计收益、年化收益、最大回撤、Sharpe、波动率
- 曲线：策略 Model A 与沪深300基准的精简日度净值快照

不包含 API Key、Token、日志、供应商缓存、原始行情、内部实验记录或本地绝对路径。

## 本地预览

```bash
npm install
npm run dev
```

## 静态构建

```bash
npm run build
```

构建产物位于 `dist/`，可直接交给 Cloudflare Pages、Vercel 或其他静态托管平台。

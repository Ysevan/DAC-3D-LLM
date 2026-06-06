# DAC-3D 工作流概览 / Workflow Overview

## 扫描配置 / Scan Setup
开始采集前，操作员需要先定义扫描区域，单位通常为毫米。`10 mm x 10 mm` 是常见的小范围验证区域，适合做快速设置检查和参数确认。

## 区域选择 / Region Selection
扫描区域应绑定到当前可见选区或已保存的 ROI。若没有明确的命名区域，应优先使用 current selection，而不是凭经验扩大扫描范围。

## 扫描模式 / Scan Modes
Standard mode 用于大多数常规扫描，平衡速度和细节。Precision mode 适合怀疑存在缺陷时做高细节确认。Fast mode 适合预览和快速检查。

## 检测状态 / Inspection Status
DAC-3D 运行状态通常包括 `idle`、`queued`、`running`、`completed` 和 `error`。做重新扫描判断前，操作员应同时查看状态消息和进度百分比。

# AI Algorithm Capability Factory

本仓库是面向算法编程教学场景的 AI Agent 原型，包含算法教练业务项目和
Pi Agent 底层框架。

## 目录

```text
algorithm-coach/   算法教练：知识检索、代码生成、验证、修复和经验沉淀
pi-main/           Pi Agent 底层框架
agent.md           Git 提交边界说明
```

## 环境

- Windows PowerShell 或 PowerShell 7
- Node.js 22.19 或更高版本
- OpenCode API Key，通过环境变量提供

## 安装

```powershell
Set-Location .\pi-main
npm install --ignore-scripts
```

## 启动

```powershell
$env:OPENCODE_API_KEY = "<your-key>"
Set-Location ..\algorithm-coach

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "..\pi-main\pi-test.ps1" `
  --model opencode-go/deepseek-v4.1-flash `
  --thinking max `
  --tools read,powershell,edit,write,grep,find,ls
```

详细的算法教练流程见 `algorithm-coach/README.md`。


# 新人上手指南

## 仓库整体概览
本仓库围绕 Mem0 记忆系统在心理咨询与健康助理场景中的应用展开，包含多套实验脚本、可视化 Streamlit 应用以及评估报告。
- `mem0_format_test.py`（文件名为 `mem0格式验证.py`）用于验证 Mem0 API 在不同查询方式下的返回结构，便于理解底层数据格式。【F:mem0格式验证.py†L1-L176】
- `mem0版个人助理.py` 和 `neo4j最终版.py` 分别提供基于 Qdrant 与 Neo4j+Qdrant 记忆存储的健康助理完整应用，实现从对话存储、上下文检索到 Streamlit UI 的端到端示例。【F:mem0版个人助理.py†L1-L250】【F:neo4j最终版.py†L1-L568】
- `Psy-Insight/` 与 `PsyDTCorpus实验代码和结果/` 目录中汇集了心理咨询问答的专业评估脚本及报告，覆盖 ROUGE/BLEU/LLM Judge、REBT 阶段等多维指标。【F:Psy-Insight/mem0_correct_experiment.py†L1-L400】【F:PsyDTCorpus实验代码和结果/professional_mem0_experiment.py†L1-L200】【F:PsyDTCorpus实验代码和结果/rebt_memory_experiment_report_20250812_154004.md†L2-L66】
- `Results_Turn_Based_Dialogue_Evaluation/` 与 `mem0-locomo-results/` 存放批量评估输出，帮助复盘模型对话表现与检索问答结果。【F:Results_Turn_Based_Dialogue_Evaluation/Career/CPsyCoun测评.py†L1-L200】【F:mem0-locomo-results/mem0_results_top_30_filter_False_graph_False.json†L1-L11】

> ⚠️ 多数脚本在源码中硬编码了 API Key/主机地址，上线前需改为本地环境变量，避免泄露。【F:mem0格式验证.py†L8-L15】【F:mem0版个人助理.py†L10-L17】【F:neo4j最终版.py†L11-L49】【F:Results_Turn_Based_Dialogue_Evaluation/Career/CPsyCoun测评.py†L12-L21】

## 目录与核心模块
### 1. 根目录脚本
- **mem0格式验证.py**：演示 `Memory.from_config` 初始化、`add/get_all/search` 调用及分页/版本参数测试，是理解 Mem0 返回格式的首选入口。【F:mem0格式验证.py†L17-L168】
- **mem0版个人助理.py**：实现健康助理 MVP，包括记忆检索、对话上下文拼接、Streamlit 聊天界面与快速提问面板，可直接运行 `streamlit run` 体验。【F:mem0版个人助理.py†L45-L250】
- **neo4j最终版.py**：在 Qdrant 向量检索基础上叠加 Neo4j 图数据库，增加健康档案摘要、图数据展示、模式切换与数据清理等功能，是复杂部署方案的参考实现。【F:neo4j最终版.py†L20-L568】

### 2. `Psy-Insight/`
- **mem0_correct_experiment.py**：构建完整的心理咨询问答实验流水线，涵盖 NLTK、ROUGE、BLEU、Sentence-BERT、LLM Judge 等指标，比较有无记忆两种模型，并按患者会话维度拆分、存储与检索记忆。【F:Psy-Insight/mem0_correct_experiment.py†L1-L400】
- **detailed_qa_comparison_report.md**：记录实验样例的问答对比、指标表格与分析文案，可作为撰写报告或调参的依据。【F:Psy-Insight/detailed_qa_comparison_report.md†L1-L160】

### 3. `PsyDTCorpus实验代码和结果/`
- **professional_mem0_experiment.py**：针对 PsyDTCorpus 数据集的专业评估脚本，结合 CPsyCoun 量表与 REBT 理论划分阶段，比较基线与记忆增强模型的表现，并输出详尽统计。【F:PsyDTCorpus实验代码和结果/professional_mem0_experiment.py†L1-L200】
- **rebt_memory_experiment_report_*.md/json**：对应实验的摘要报告与原始结果，展示各评估维度的提升幅度及改进建议。【F:PsyDTCorpus实验代码和结果/rebt_memory_experiment_report_20250812_154004.md†L2-L66】

### 4. 评估与结果目录
- **Results_Turn_Based_Dialogue_Evaluation/**：包含回合制对话评估脚本 `CPsyCoun测评.py` 与多组评分结果文本，可快速复现实验或追加新模型输出。【F:Results_Turn_Based_Dialogue_Evaluation/Career/CPsyCoun测评.py†L1-L200】
- **mem0-locomo-results/**：保存 Locomo 评测生成的 JSON 结果（问题、预测答案、证据等），便于对比检索/生成准确性。【F:mem0-locomo-results/mem0_results_top_30_filter_False_graph_False.json†L1-L11】

## 快速上手建议
1. **环境配置**：整理 `.env` 或 Secrets，替换所有硬编码的 API Key，并确保 Qdrant、Neo4j 等依赖服务可用。【F:neo4j最终版.py†L20-L50】
2. **理解 Mem0 数据结构**：先运行 `mem0格式验证.py`，观察 `get_all`/`search` 不同版本的返回格式，为后续实验做铺垫。【F:mem0格式验证.py†L17-L168】
3. **阅读实验脚本**：按难度逐步学习——`mem0_correct_experiment.py`（单体实验流程）→ `professional_mem0_experiment.py`（REBT 框架集成）→ `CPsyCoun测评.py`（批量评估自动化）。【F:Psy-Insight/mem0_correct_experiment.py†L200-L400】【F:PsyDTCorpus实验代码和结果/professional_mem0_experiment.py†L1-L200】【F:Results_Turn_Based_Dialogue_Evaluation/Career/CPsyCoun测评.py†L1-L200】
4. **体验 Streamlit 应用**：本地运行健康助理（基础版与 Neo4j 版），感受记忆检索对响应质量的影响，同时验证 UI/状态管理逻辑。【F:mem0版个人助理.py†L139-L250】【F:neo4j最终版.py†L352-L568】
5. **复用评估结果**：参考 `PsyDTCorpus` 报告中的指标和改进建议，规划下一步实验设计或产品化需求。【F:PsyDTCorpus实验代码和结果/rebt_memory_experiment_report_20250812_154004.md†L10-L66】

## 后续深入方向
- **记忆策略优化**：在 `Mem0Model.store_memory` 等函数基础上尝试分层存储、筛选关键轮次，降低成本并提升检索精度。【F:Psy-Insight/mem0_correct_experiment.py†L278-L366】
- **多模态/多源数据扩展**：Neo4j 版健康助理已经提供图谱接口，可继续扩展结构化健康数据或接入设备指标。【F:neo4j最终版.py†L129-L350】
- **评估自动化**：将 `CPsyCoun测评.py` 的评分流程包装成批处理服务，与 JSON 结果目录联动形成持续评估流水线。【F:Results_Turn_Based_Dialogue_Evaluation/Career/CPsyCoun测评.py†L41-L200】【F:mem0-locomo-results/mem0_results_top_30_filter_False_graph_False.json†L1-L11】
- **安全合规**：梳理隐私与安全流程，尤其是 API Key 管理与健康数据存储策略，准备部署前的审计材料。【F:mem0版个人助理.py†L10-L135】【F:neo4j最终版.py†L11-L350】

---
如需进一步支持，可从运行最小化格式验证脚本入手，逐步扩展到完整实验与前端应用，快速建立对 Mem0 在心理/健康场景中落地的整体认知。

# 测试结果证据清单

| 类别 | 文件 | 说明 |
| --- | --- | --- |
| Step1 LLM证据 | 01_step1_llm_evidence/llm_5_rounds_payload_response.jsonl | 当前复跑的5轮外部API请求payload与响应；本次返回403 GROUP_DELETED。 |
| Step1 后端日志 | 01_step1_llm_evidence/backend_external_api_call_log.jsonl | 当前复跑外部API调用日志，包含时间戳、模型、状态码和耗时。 |
| Step2 测试集 | 02_step2_test_sets/*.csv, *.md | RAG、命令生成、结果解释测试集及知识库文档说明。 |
| Step3 性能 | 03_step3_performance/performance_summary_from_thesis_table_6_5.csv | 论文表6-5采用的性能指标。 |
| Step3 当前复跑 | 03_step3_performance/rerun_smoke_results.json | 当前本地服务health/runtime/knowledge-base smoke复跑结果。 |
| Step4 功能异常 | 04_step4_function_exception/*.csv | 功能测试汇总、覆盖矩阵和E-01至E-05异常测试归档。 |
| Step5 SUS | 05_step5_user_sus/sus_user_test_results.csv | 5名用户SUS原始评分、得分、等级与任务完成情况。 |

# CMLM-ZhongJing（中医语言模型-仲景）
A Traditional Chinese Medicine large language model, inspired by the wisdom of the eminent representative of ancient Chinese medical scholars, Zhang Zhongjing.
This model aims to illuminate the profound knowledge of Traditional Chinese Medicine, bridging the gap between ancient wisdom and modern technology, and providing a reliable and professional tool for the medical and legal fields. However, all generated results are for reference only and should be provided by experienced professionals for diagnosis and treatment results and suggestions.

## 1.Instruction Data Construction
While many works such as Alpaca, Belle, etc., are based on the self-instruction approach which effectively harnesses the knowledge of large language models to generate diverse and creative instructions, this approach may lead to noise in instruction data, thereby affecting the accuracy of the model in fields where professional knowledge has a low tolerance for errors, such as medical and legal scenarios. Therefore, how to quickly invoke the OpenAI API without sacrificing the professionalism of instruction data has become an important research direction for instruction data construction and annotation scenarios. Here, we will briefly describe our preliminary experimental exploration.
## 1.指令数据构建：
目前大多如Alpaca、Belle等工作基于self-instruction思路。self-instruction思路可以很好的调用大语言模型的知识，生成多样和具有创造性的指令，在常规问答场景可以快速构造海量指令实现指令调优。但在一些专业知识容错率较低的领域，比如医疗和法律场景，幻觉输出会导致噪声指令数据从而影响模型的准确性。典型的情况是比如不当的诊断及处方建议甚至影响患者生命，事实性错误的法律条文和法理的引用会造成权益人的败诉。因此，如何快速调用OpenAI API且不牺牲指令数据的专业性成为指令数据构造及标注等场景的重要研究方向。以下将简述我们的初步实验探索。

#### 1.1 Multi-task Therapeutic Behavior Decomposition Instruction Construction Strategy
Human memory and understanding require the construction of various scenarios and stories to implicitly encode knowledge information. The clarity of memory depends on the duration and richness of the learning process. Interleaved learning, spaced practice, and diversified learning can enhance the consolidation of knowledge, thereby forming a deep understanding of domain knowledge. Our approach is to learn from the process of human memory knowledge, use professional tables, leverage the language representation capabilities of large language models, strictly set specific prompt templates, so that the model can generate 16 scenarios based on the table data of Chinese medicine gynecology prescriptions, including patient therapeutic story, diagnostic analysis, diagnosis treatment expected result, formula function, interactive story, patient therapeutic story, narrative medicine, tongue & pulse, therapeutic template making, critical thinking, follow up, prescription, herb dosage, case study, real-world problem, disease mechanism, etc., to promote the model's reasoning ability for prescription data and diagnostic thinking logic.
#### 1.1多任务诊疗行为分解instruction构建策略
人类在记忆和理解时需要构建各种情景和故事，以隐式编码知识信息。记忆的清晰程度取决于学习的持续过程和丰富程度。穿插学习、间隔练习和多样化学习可以提升知识的巩固程度，由此形成深刻的领域知识理解能力。我们的思路是借鉴人类记忆知识的过程，采用专业表格，借助大语言模型的语言表征能力，严格设置特定的prompt模板，使得模型基于中医妇科方药表格数据生成包括患者治疗故事、诊断分析、诊断治疗预期结果、处方功用、互动故事、患者治疗故事、叙事医学、舌脉象、诊疗方案制定、批判性思维、随访、处方、药物用量、个例研究、真实世界问题、病因病机等16个场景，以促进模型对中医方药数据及诊断思维逻辑的推理能力。

#### 1.2 Regular Instruction Data Construction Strategy
In addition, we have also added instructions based on the content of Chinese medicine ancient books, noun explanations, symptom synonyms, antonyms, syndromes, symptoms, treatment methods, etc. In order to form a control experiment, we only use one instruction template to represent data for this part, and the number of this part of the data is 80,000, which is significantly more than the number of instructions constructed by the above strategy. The following is the specific number of instructions and tokens information.
Data Source and Instruction Quantity Table:
#### 1.2 常规指令数据构建策略
此外，我们还增加了基于中医古籍内容、名词解释、症状近义词、反义词、证候、症状、治法等指令内容，为了形成对照试验，我们对这部分仅仅采用一种指令模板以表征数据，同时这部分数据的数量约为8万条，明显多于上述策略构建的指令数量，以下为指令具体数量及tokens数量信息。

| File Name | Total Tokens Quantity | Input Quantity | Instruction Quantity | Output Quantity |
| --- | --- | --- | --- | --- |
| patient_therapeutic_story_data1.json | 62722 | 208 | 208 | 208 |
| diagnostic_analysis.json | 1492105 | 6592 | 6592 | 6592 |
| formula_funtion_data.json | 100533 | 2115 | 2115 | 2115 |
| diagnosis_treatment_expected_result_formatted_... | 33822 | 153 | 153 | 153 |
| Chinese Medicine Dictionary.json | 2188672 | 20376 | 20376 | 20376 |
| Antonyms.json | 272 | 9 | 9 | 9 |
| Interactive Story Instructed Data.json | 55262 | 219 | 219 | 219 |
| patient_therapeutic_story_data3.json | 50785 | 660 | 660 | 660 |
| Syndrome Noun Explanation.json | 67443 | 976 | 976 | 976 |
| narrative_medicine_formatted_data.json | 61336 | 213 | 213 | 213 |
| Chinese Medicine Symptom Synonyms.json | 1515796 | 27650 | 27650 | 27650 |
| Synonyms2.json | 111186 | 2217 | 2217 | 2217 |
| Ancient Books Content.json | 15971297 | 31395 | 31395 | 31395 |
| tongue_palse.json | 328597 | 3723 | 3723 | 3723 |
| therapeutic_template_making.json | 335602 | 4929 | 4929 | 4929 |
| patient_therapeutic_story_data2.json | 50785 | 660 | 660 | 660 |
| critical_thinking_data.json | 31502 | 229 | 229 | 229 |
| follow_up_data.json | 504717 | 5990 | 5990 | 5990 |
| prescription_data.json | 107694 | 2898 | 2898 | 2898 |
| herb_dosage.json | 564394 | 5973 | 5973 | 5973 |
| case_study_data.json | 58319 | 243 | 243 | 243 |
| Gynecology Synonyms.json | 29740 | 543 | 543 | 543 |
| real_world_problem.json | 1493551 | 7990 | 7990 | 7990 |
| disease_mechanism.json | 997377 | 8024 | 8024 | 8024 |
| Treatment Noun Explanation Cleaned Data.json | 81211 | 1123 | 1123 | 1123 |
| Total | 26294720 | 135108 | 135108 | 135108 |


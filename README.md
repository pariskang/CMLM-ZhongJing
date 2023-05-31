# CMLM-ZhongJing（中医大语言模型-仲景）
A Traditional Chinese Medicine large language model, inspired by the wisdom of the eminent representative of ancient Chinese medical scholars, Zhang Zhongjing.
This model aims to illuminate the profound knowledge of Traditional Chinese Medicine, bridging the gap between ancient wisdom and modern technology, and providing a reliable and professional tool for the Traditional Chinese Medical fields. However, all generated results are for reference only and should be provided by experienced professionals for diagnosis and treatment results and suggestions.

中医大语言模型，灵感来自中国古代杰出医家张仲景的智慧。 该模型旨在阐明中医博大精深之知识，传承古代智慧与现代技术创新，最终为医学领域提供可信赖和专业的工具。然而，目前所有产生的结果仅供参考，应由经验丰富的专业人员提供诊断和治疗结果和建议。

## 1.Instruction Data Construction
While many works such as Alpaca, Belle, etc., are based on the self-instruct approach which effectively harnesses the knowledge of large language models to generate diverse and creative instructions, this approach may lead to noise in instruction data, thereby affecting the accuracy of the model in fields where professional knowledge has a low tolerance for errors, such as medical and legal scenarios. Therefore, how to quickly invoke the OpenAI API without sacrificing the professionalism of instruction data has become an important research direction for instruction data construction and annotation scenarios. Here, we will briefly describe our preliminary experimental exploration.
## 1.指令数据构建：
目前大多如Alpaca、Belle等工作基于self-instruct思路。self-instruct思路可以很好的调用大语言模型的知识，生成多样和具有创造性的指令，在常规问答场景可以快速构造海量指令实现指令调优。但在一些专业知识容错率较低的领域，比如医疗和法律场景，幻觉输出会导致噪声指令数据从而影响模型的准确性。典型的情况是比如不当的诊断及处方建议甚至影响患者生命，事实性错误的法律条文和法理的引用会造成权益人的败诉。因此，如何快速调用OpenAI API且不牺牲指令数据的专业性成为指令数据构造及标注等场景的重要研究方向。以下将简述我们的初步实验探索。

#### 1.1 Multi-task Therapeutic Behavior Decomposition Instruction Construction Strategy
Human memory and understanding require the construction of various scenarios and stories to implicitly encode knowledge information. The clarity of memory depends on the duration and richness of the learning process. Interleaved learning, spaced practice, and diversified learning can enhance the consolidation of knowledge, thereby forming a deep understanding of domain knowledge. Our approach is to learn from the process of human memory knowledge, use professional tables, leverage the language representation capabilities of large language models, strictly set specific prompt templates, so that the model can generate 16 scenarios based on the table data of Chinese medicine gynecology prescriptions, including patient therapeutic story, diagnostic analysis, diagnosis treatment expected result, formula function, interactive story, patient therapeutic story, narrative medicine, tongue & pulse, therapeutic template making, critical thinking, follow up, prescription, herb dosage, case study, real-world problem, disease mechanism, etc., to promote the model's reasoning ability for prescription data and diagnostic thinking logic.
#### 1.1多任务诊疗行为分解instruction构建策略
人类在记忆和理解时需要构建各种情景和故事，以隐式编码知识信息。记忆的清晰程度取决于学习的持续过程和丰富程度。穿插学习、间隔练习和多样化学习可以提升知识的巩固程度，由此形成深刻的领域知识理解能力。我们的思路是借鉴人类记忆知识的过程，采用专业表格，借助大语言模型的语言表征能力，严格设置特定的prompt模板，使得模型基于中医妇科方药表格数据生成包括患者治疗故事、诊断分析、诊断治疗预期结果、处方功用、互动故事、患者治疗故事、叙事医学、舌脉象、诊疗方案制定、批判性思维、随访、处方、药物用量、个例研究、真实世界问题、病因病机等16个场景，以促进模型对中医方药数据及诊断思维逻辑的推理能力。
```
    {
        "instruction": "我对三元汤的全过程很好奇，能否从简介、病历、症状、诊断和治疗，以及结果讨论等方面给我详细介绍？",
        "input": "",
        "output": "三元汤是一种古老的中药处方，最早出现在《保命集》卷下。它由多种中药组成，主要用于治疗产后日久虚劳，脉浮疾等症状。患者为产后日久虚劳，脉浮疾。脉浮疾。根据李女士的病史和症状，我决定使用三元汤来治疗她的病症。我按照处方中的药物组成和用法用量，将柴胡、黄芩、人参、半夏、甘草、川芎、芍药、熟地黄、当归等中药煎煮后，让李女士每天服用两次。在治疗过程中，我还注意了李女士的饮食和休息，让她多吃一些易于消化的食物，保证充足的睡眠和休息。经过一个月的治疗，李女士的症状得到了明显的改善。她的精神状态更加稳定，食欲也有所增加，头晕乏力的症状也减轻了很多。我认为，三元汤的治疗效果主要得益于其中的多种中药的协同作用，能够调节李女士的身体机能，增强体质，提高免疫力。同时，我也注意到，李女士的饮食和休息也对治疗效果有一定的影响。因此，在使用中药治疗疾病时，我们也应该注意患者的全面治疗，包括饮食、休息等方面。"
    }
```

#### 1.2 Regular Instruction Data Construction Strategy
In addition, we have also added instructions based on the content of Chinese medicine ancient books, noun explanations, symptom synonyms, antonyms, syndromes, symptoms, treatment methods, etc. In order to form a control experiment, we only use one instruction template to represent data for this part, and the number of this part of the data is 80,000, which is significantly more than the number of instructions constructed by the above strategy. The following is the specific number of instructions and tokens information.
Data Source and Instruction Quantity Table:
#### 1.2 常规指令数据构建策略
此外，我们还增加了基于中医古籍内容、名词解释、症状近义词、反义词、证候、症状、治法等指令内容，为了形成对照试验，我们对这部分仅仅采用一种指令模板以表征数据，同时这部分数据的数量约为8万条，明显多于上述策略构建的指令数量，以下为指令具体数量及tokens数量信息。
```
    {
        "input": "（肺风）粉刺属于哪个分类？",
        "instruction": "请回答以下有关于中医疾病名词解释的相关问题：",
        "output": "因肺风、胃热或肝瘀所致。以面及背部见黑头或白头粉刺、丘疹、脓疱、结节、囊肿及疤痕为主要表现的皮肤疾病。"
    }
```

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

## To Do List
- [ ] Adopt a multi-task therapeutic decomposition strategy, based on multidisciplinary data such as internal medicine, gynecology, pediatrics, and orthopedics, to fine-tune the model with a domain-specific million-level instruct data.
- [ ] Continuously iterate and update. Subsequent releases will include Li Shizhen, Wang Shuhe, Huangfu Mi, Sun Simiao, Ge Hong, and Qihuang version of the large language model for Traditional Chinese Medicine.
- [ ] Explore efficient domain fine-tuning strategies.
## 待做清单
- [ ] 采用多任务诊疗分解策略，基于内外妇儿骨等多学科数据构建领域百万级instruct数据微调模型
- [ ] 持续迭代更新，后续将发布李时珍、王叔和、皇甫谧、孙思邈、葛洪、岐黄版中医药大语言模型
- [ ] 探索高效领域微调策略

## Acknowledgements
The Lora fine-tuning part of this project draws on the ideas of alpaca-lora and Chinese-Vicuna. We would like to express our gratitude to the members of the relevant research teams.
## 致谢声明
本项目Lora微调部分代码借鉴alpaca-lora、Chinese-Vicuna思路，我们对相关研究团队成员表示感谢。

## Disclaimer
This research is for academic research use only, commercial use is not allowed without permission, and it is not to be used in medical scenarios or scenarios with potential medical intent for clinical practice. This large language model for Traditional Chinese Medicine is still in the laboratory testing stage. The emerging syndrome classification and prescription generation capabilities at this stage are still rudimentary, and it does not yet have a highly reliable clinical diagnostic and therapeutic capability for gynecology and other clinical specialties. The output results are for internal reference testing only. Real medical diagnosis and decision-making still need to be issued by experienced physicians through a strictly regulated diagnostic and therapeutic process.
## 免责声明
本研究仅供学术研究使用，未经允许不得商业使用，不得在医疗场景或具有潜在医疗意图场景进行临床实践。本中医药大语言模型还处于实验室测试阶段，本阶段涌现的证型分类和处方生成能力尚且粗浅，对于妇科及其他临床专科尚不具备高度可信的临床诊疗能力，目前尚不具有医疗实践能力，输出结果仅供内部参考测试。真实的医疗诊断及决策依然需要经经验丰富的医师通过严格规范的诊疗过程出具。

## Collaboration
Data processing and annotation is one of the important steps in training the model. We sincerely welcome Traditional Chinese Medicine practitioners with strong TCM thinking and innovative spirit to join us. We will also declare corresponding data contributions. We look forward to the day when we can achieve a reliable General Artificial Intelligence for Traditional Chinese Medicine, allowing the ancient Chinese medicine to blend with modern technology and shine anew. This is also the ultimate mission of this project. If interested, please send an email to 21110860035@m.fudan.edu.cn.
## 合作事宜
数据处理与标注是训练模型重要环节之一，我们诚挚欢迎具有浓厚中医思维及创新精神的中医师加入，也会在数据层面声明相应贡献，期待我们有朝一日实现可信赖的中医通用人工智能，让古老的中医学与新时代科技融合焕发新春，这也是本项目的最终使命。如有意向，请发邮件到21110860035@m.fudan.edu.cn。

## Team Introduction
This project is jointly guided by Professor Zhang Wenqiang from Fudan University and Professor Wang Haofen from Tongji University. It is completed by Kang Yanlan, Chang Yang, and Fu Jiyuan, members of the [ROI Lab](https://www.fudanroilab.com/) at Fudan University.
## 团队介绍
本项目由复旦大学张文强教授和同济大学王昊奋教授共同指导，由复旦大学[ROI Lab](https://www.fudanroilab.com/)成员康砚澜、常扬、符纪元通力协作完成。

## Citation
If you find this work useful in your research, please cite our repository:
```
@misc{CMLM-ZhongJing,
  author = {Kang, Yanlan and Chang, Yang and Fu, Jiyuan and Wang, Haofen and Zhang, Wenqiang},
  title = {CMLM-ZhongJing: Large Language Model are Good Story Listener},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/pariskang/CMLM-ZhongJing}}
}
```

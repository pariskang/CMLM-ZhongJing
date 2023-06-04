# CMLM-ZhongJing（中医大语言模型-仲景）
A Traditional Chinese Medicine large language model, inspired by the wisdom of the eminent representative of ancient Chinese medical scholars, Zhang Zhongjing.
This model aims to illuminate the profound knowledge of Traditional Chinese Medicine, bridging the gap between ancient wisdom and modern technology, and providing a reliable and professional tool for the Traditional Chinese Medical fields. However, all generated results are for reference only and should be provided by experienced professionals for diagnosis and treatment results and suggestions.

中医大语言模型，灵感来自中国古代杰出医家张仲景的智慧。 该模型旨在阐明中医博大精深之知识，传承古代智慧与现代技术创新，最终为医学领域提供可信赖和专业的工具。然而，目前所有产生的结果仅供参考，应由经验丰富的专业人员提供诊断和治疗结果和建议。

<p align="center"> <img src="https://raw.githubusercontent.com/pariskang/CMLM-ZhongJing/main/logo.png" alt="logo" title="logo" width="50%"> </p>

## 1.Instruction Data Construction
While many works such as Alpaca, Belle, etc., are based on the self-instruct approach which effectively harnesses the knowledge of large language models to generate diverse and creative instructions, this approach may lead to noise in instruction data, thereby affecting the accuracy of the model in fields where professional knowledge has a low tolerance for errors, such as medical and legal scenarios. Therefore, how to quickly invoke the OpenAI API without sacrificing the professionalism of instruction data has become an important research direction for instruction data construction and annotation scenarios. Here, we will briefly describe our preliminary experimental exploration.
## 1.指令数据构建：
目前大多如Alpaca、Belle等工作基于self-instruct思路。self-instruct思路可以很好的调用大语言模型的知识，生成多样和具有创造性的指令，在常规问答场景可以快速构造海量指令实现指令调优。但在一些专业知识容错率较低的领域，比如医疗和法律场景，幻觉输出会导致噪声指令数据从而影响模型的准确性。典型的情况是比如不当的诊断及处方建议甚至影响患者生命，事实性错误的法律条文和法理的引用会造成权益人的败诉。因此，如何快速调用OpenAI API且不牺牲指令数据的专业性成为指令数据构造及标注等场景的重要研究方向。以下将简述我们的初步实验探索。

<p align="center"> <img src="https://raw.githubusercontent.com/pariskang/CMLM-ZhongJing/main/logo_image/strategy.jpeg" alt="strategy" title="strategy" width="100%"> </p>

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
        "instruction": "请回答以下有关于中医疾病名词解释的相关问题：",
        "input": "（肺风）粉刺属于哪个分类？",
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

## 2.Model Performance Comparison
Our test data are based on real medical cases from highly skilled traditional Chinese medicine doctors, typically case reports from provincial renowned senior traditional Chinese medicine practitioners or national medical master level. This kind of data, which is strictly considered as out-of-distribution data (both in terms of subject matter and training dataset distribution, distinct from traditional training and validation sets), is used to ensure a degree of professionalism. In preliminary comparisons with large language models such as Wenxin Yiyan and Spark, we found that our model exhibits good generalization capabilities on a diversified therapeutic decomposition instruction dataset constructed based on 300 Traditional Chinese Medicine prescription data. This perhaps initially confirms that, like humans, large language models are more conducive to learning metaphorical knowledge and logic from text content represented in diverse forms.
## 2.模型效果对比：
我们的测试数据基于真实高水平中医师的医学案例，通常为省级名老中医或国医大师级别的个案报告，以保证一定层面专业性。这样的数据较严格的属于分布外数据（学科分布外与训练数据集分布外，有别于传统训练集和验证集）。通过与文心一言、星火等大语言模型进行初步对比，发现我们的模型在基于300条中医方药数据构建的多样化诊疗分解指令数据集上具备良好的泛化能力，或许初步证实大语言模型与和人类一样，对于多元形式表征的文本内容更有助于学习隐喻的知识及逻辑。

| | | | | | |
|-|-|-|-|-|-|
|序号|提问|文心一言|星火认知|仲景(384 Tokens)|仲景(512 Tokens)|
|1|张某，男，27岁。患者因昨晚饮酒发热，渴凉水数杯，早晨腹痛腹泻，泻下如水色黄，腹中辘辘有声，恶心欲吐，胸中满闷不舒，口干欲冷饮，舌质红、苔白腻，脉沉细数。给出中医诊断和处方建议|中医诊断：酒积湿热 处方建议： 治法：清热解毒，除湿止泻。 方剂：选用葛根芩连汤加减。 药物组成：葛根、黄芩、黄连、茯苓、薏苡仁、木瓜、陈皮、白芍、生姜、大枣等。 剂量与用法：以水煎服，每日1-2剂，每次服用100-200毫升，每日3-4次。 解释： 方解：方中葛根可解肌退热、升清止泻；黄芩、黄连可清热燥湿、解毒止泻；茯苓、薏苡仁可利水渗湿、健脾止泻；木瓜可化湿和胃、止呕止泻；陈皮可理气健脾、燥湿化痰；白芍可柔肝止痛、敛阴止泻；生姜可温中散寒、和胃止呕；大枣可补中益气、调和脾胃。 加减用药：若患者腹痛明显，可加用延胡索、川楝子等以增强止痛作用；若患者腹泻严重，可加用诃子、石榴皮等以涩肠止泻；若患者口干舌燥，可加用石膏、知母等以清热生津。 注意事项：建议患者在服药期间忌食辛辣油腻食物，避免饮酒及生冷食物，注意饮食卫生，保持心情舒畅。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为湿热泄泻。 - 病因：饮酒过度，感受寒湿之邪，加上饮食不节，导致湿热内生，蕴结于脾胃，引起了泄泻的症状。 - 诊断依据：腹中辘辘有声，泻下如水色黄，口干欲冷饮，舌质红、苔白腻，脉沉细数等表现，符合湿热泄泻的证候特点。 - 治疗建议：中药治疗方面，可选用藿香正气散加减、二妙散等方剂，以宣通脾肺、祛湿止泻。同时应忌食辛辣、油腻、生冷之物，多饮温开水，保持休息。针灸、艾灸等中医辅助治疗方法也可尝试。|根据患者的症状，可以初步判断为脾胃虚弱证、湿热内蕴证。进一步了解患者的病史和体质，发现其是由于饮酒过度导致肝胆风热，从而引起上述症状。因此，可以考虑使用伏牛花散来治疗。伏牛花散是一种中药方，主要成分为伏牛花、山茵陈、桑寄生、白牵牛、川芎、白僵蚕、蝎梢、荆芥穗等。这些药物具有清热解毒、活血化瘀的功效，可以清热解毒，活血化瘀，从而达到治疗疾病的目的。制备方法为将伏牛花、山茵陈、桑寄生、白牵牛、川芎、白僵蚕、蝎梢、荆芥穗等药材研磨成细|根据患者的症状，可以初步判断为脾胃虚弱证、湿热内蕴证。进一步了解患者的病史和体质，发现其是由于饮酒过度导致肝胆风热，从而引起上述症状。因此，可以考虑使用伏牛花散来治疗。伏牛花散是一种中药方，主要成分为伏牛花、山茵陈、桑寄生、白牵牛、川芎、白僵蚕、蝎梢、荆芥穗等。这些药物具有清热解毒、活血化瘀的功效，可以清热解毒，活血化瘀，从而达到治疗疾病的目的。制备方法为将伏牛花、山茵陈、桑寄生、白牵牛、川芎、白僵蚕、蝎梢、荆芥穗等药材按比例混合，研磨成细末，取适量，每次服用3-5钱，空心、食前温酒调下。在用药过程中，应注意避免过量使用，以免出现不良反应。同时，患者在日常生活中应注意饮食清淡，避免辛辣刺激性食物，保持情|
|2|于某，男，62岁。患冠心病两年，服西药治疗，一日三次，从未有断，然胸憋心悸，一直不止。近月余，每至夜则咳嗽哮喘，痰涎清稀如水，倚息不能平卧，胸憋心悸尤甚。白昼则症状减轻。询知腰脊酸困，背畏风寒，时眩晕，手足心微热，口渴欲饮，但不多饮，亦不思冷，纳便尚可，舌尖略红，苔白腻，脉沉缓。给出中医诊断和处方建议|中医诊断：痰饮咳嗽（证属脾肾阳虚，饮邪上犯） 处方建议： 温补脾肾，化痰止咳：选用苓桂术甘汤加减。 温阳化饮，止咳平喘：选用真武汤加减。 补肾纳气，止咳平喘：选用肾气丸加减。 燥湿化痰，降逆止咳：选用二陈汤加减。 补脾益肺，止咳化痰：选用六君子汤加减。 以上建议仅供参考，具体用药请遵医嘱。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为心肺气阴两虚。 - 病因：冠心病导致心脏气血不足，加上长期服用药物，伤及气血，加之外感风寒湿邪，耗伤体内阴液，导致心肺气阴两虚。 - 诊断依据：胸憋心悸、咳嗽哮喘、口渴欲饮、舌尖略红、苔白腻、脉沉缓等表现，符合心肺气阴两虚的证候特点。 - 治疗建议：中药治疗方面，可选用生脉散加减、六味地黄丸等方剂，以滋补心肺、益气养阴。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。患者还需注意保暖，避免受凉感冒，加强身体锻炼，促进气血循环。|根据患者的症状，可以初步判断为腰脊酸困、胸憋心悸、痰涎清稀如水、倚息不能平卧、手足心微热、口渴欲饮、亦不思冷等证候类型。进一步询问患者的病史、饮食习惯、生活习惯等，以及进行望、闻、问、切四诊，确定患者的中医诊断。根据中医诊断，可以选择相应的中药方进行治疗。其中，当归散补血汤是一种常用的中药方，具有补血安胎、调经止血的功效，适用于治疗冠心病等症状。药物组成为当归、川芎、白芍、熟地黄、黄芩、大黄、茯苓、柴胡、半夏、甘草、红花、益母草、熟地黄、当归、川芎、白芍、熟地黄|根据患者的症状，可以初步判断为腰脊酸困、胸憋心悸、痰涎清稀如水、倚息不能平卧、手足心微热、口渴欲饮、亦不思冷等证候类型。进一步询问患者的病史、饮食习惯、生活习惯等，以及进行望、闻、问、切四诊，确定患者的中医诊断。根据中医诊断，可以选择相应的中药方进行治疗。其中，当归散补血汤是一种常用的中药方，具有补血安胎、调经止血的功效，适用于治疗冠心病等症状。药物组成为当归、川芎、白芍、熟地黄、黄芩、大黄、茯苓、柴胡、半夏、甘草、红花、益母草、熟地黄、当归、川芎、白芍、熟地黄、黄芩、大黄、茯苓、柴胡、半夏、甘草、红花、益母草等。制备方法为水煎服用，用法用量为一次服用一剂，每日3次。需要注意的是，该处方有一定的用药禁忌，如孕妇、哺乳期妇女、体弱者等不|
|3|戴某某，女，22岁，未婚。三年来行经腹痛，第一、二天痛剧，开始血量少，待三日血量渐多而痛稍减，色谈有块，周期尚准。平素喜暖畏寒，体倦乏力，不耐劳累，经至必服止痛片及中药，以求暂安。此次行经少腹痛剧，虽已过十余天，少腹仍绵绵作痛，时有发胀，舌淡苔白，脉细而迟。给出中医诊断和处方建议|中医诊断：痛经（证属寒凝血瘀型） 处方建议： 治法：温经止痛，化瘀通络。 方剂：选用温经汤加减。 药物组成：当归、白芍、川芎、桂枝、细辛、附子、干姜、吴茱萸、牡丹皮、生地黄、阿胶、荆芥穗、炙甘草等。 剂量与用法：以水煎服，每日1剂，每次服用100-200毫升，每日3-4次。 解释： 方解：方中当归、白芍、川芎、桂枝、细辛、附子、干姜、吴茱萸等药味辛性温，可温经止痛，化瘀通络；牡丹皮、生地黄、阿胶等药味甘性寒，可滋阴养血，清热化瘀；荆芥穗可祛风散寒，理气止痛；炙甘草可调和诸药。 加减用药：若腹痛较剧，可加用乌药、小茴香等以增强止痛作用；若月经量多，可加用蒲黄、五灵脂等以活血化瘀；若经前乳房胀痛明显，可加用柴胡、黄芩等以疏肝理气。 注意事项：建议患者在服药期间忌食生冷刺激性食物，注意保暖，避免过度劳累，保持心情舒畅。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为气滞血瘀型痛经。 - 病因：由于情志不畅，饮食不节，加上气血运行不畅，导致气滞血瘀，引起了痛经的症状。 - 诊断依据：经期腹痛，经量少，色谈有块，舌淡苔白，脉细而迟等表现，符合气滞血瘀型痛经的证候特点。 - 治疗建议：中药治疗方面，可选用逍遥散加减、当归生姜羊肉汤等方剂，以疏肝理气、活血化瘀。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。患者还需注意保暖，避免受凉感冒，加强身体锻炼，促进气血循环。在治疗过程中，可以逐渐减少止痛药的使用，以免对肝肾造成不良影响。|根据您的症状，我建议您可以按以下方法使用当归内补丸：药物组成：当归、川芎、白芍、鲜地黄、黄芩、白术、茯苓，其剂量为：当归4两，川芎4两，白芍4两，鲜地黄4两，黄芩4两，白术4两，茯苓4两。制备方法：将以上药材研磨成细末，每次服用1丸，用温开水或白酒送服。如果您还有其他疾病或者正在服用其他药物，请告知我，以便更好地治疗您的病症。注意：如果您出现了不适症状，如头晕、恶心、呕吐、腹泻等，请立即停止使用并咨询医生。此处方也包含某些药材，如当归、川芎、白芍、鲜地黄、黄芩、白术、茯苓等，可能会引起一些|根据您的症状，我建议您可以按以下方法使用当归内补丸：药物组成：当归、川芎、白芍、鲜地黄、黄芩、白术、茯苓，其剂量为：当归4两，川芎4两，白芍4两，鲜地黄4两，黄芩4两，白术4两，茯苓4两。制备方法：将以上药材研磨成细末，每次服用1丸，用温开水或白酒送服。如果您还有其他疾病或者正在服用其他药物，请告知我，以便更好地治疗您的病症。注意：如果您出现了不适症状，如头晕、恶心、呕吐、腹泻等，请立即停止使用并咨询医生。此处方也包含某些药材，如当归、川芎、白芍、鲜地黄、黄芩、白术、茯苓等，可能会引起一些副作用，如口干、口渴、恶心、呕吐、腹泻等，因此建议您在医生的指导下使用。同时，您应注意饮食调理，多食用易消化、富含营养的食物，避免过度劳累，保持良好的心态和作息规律。如|
|4|刘某某，女，12岁。初春感受风寒邪气，头痛发热，家人自购“平热散”，服药后汗出较多，随后发热消退。但第二天发热恶寒如疟疾之发作，上午一次，下午二次。脉浮略数，舌苔薄白而润。给出中医诊断和处方建议|中医诊断：表寒里饮 处方建议： 疏风散寒，解表化饮：选用麻黄汤加减。 调和营卫，益气健脾：选用桂枝加黄芪汤加减。 以上建议仅供参考，具体用药请遵医嘱。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为风寒感冒。 - 病因：感受风寒邪气，加上自购的平热散不当使用，损伤了正气，使得邪气得以内陷而发病。 - 诊断依据：发热恶寒如疟疾之发作，脉浮略数，舌苔薄白而润等表现，符合风寒感冒的证候特点。 - 治疗建议：中药治疗方面，可选用麻黄汤加减、桂枝汤等方剂，以解表发汗、温通阳气。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。患者还需注意保暖，避免受凉感冒，加强身体锻炼，促进气血循环。在治疗过程中，应根据病情变化及时调整处方，以达到治疗效果。|根据您的症状，我建议您使用刘某的中药方「平热散」进行治疗。该方由香附、川芎、当归、柴胡、秦艽、熟地黄、新疆延胡索、苦楝皮和槟榔组成。这些药物具有清热解毒、燥湿止带的功效，能够清除体内的风热毒素，从而改善您的头痛和发热等症状。建议您每次服用3钱，用水1盏煎8分钟后服用。在服用过程中，注意避免食用辛辣刺激性食物，以免影响药效。同时，建议您适当休息，保持心情舒畅，避免过度疲劳。如果症状没有改善或加重，请及时停止服用并咨询医生。此处方只适用于一定类型的症状，如初春感| |
|5|周某某．男，29岁。因突起耳聋近—月，经他医用益气聪明汤等治疗，耳聋如故，痛苦不堪，后延余诊治。细询知病起于感冒过汗之后。刻诊：白诉耳聋，并觉心悸乏力，稍有畏寒感，舌淡红，苔薄白。脉细软无力。给出中医诊断和处方建议|中医诊断：气虚感冒 处方建议： 益气解表，聪耳明目：选用益气聪明汤加减。 调和营卫，益气健脾：选用桂枝加黄芪汤加减。 以上建议仅供参考，具体用药请遵医嘱。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为中耳炎。 - 病因：感冒后因风寒湿邪侵袭，加上体质虚弱，使得外邪侵入耳窍，导致中耳炎的症状出现。 - 诊断依据：突起耳聋，心悸乏力，畏寒感等表现，舌淡红，苔薄白，脉细软无力等证候特点，均符合中耳炎的诊断标准。 - 治疗建议：中药治疗方面，可选用清热解毒汤加减、龙胆泻肝汤等方剂，以清热解毒、祛湿通窍。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。患者还需注意保暖，避免受凉感冒，加强身体锻炼，促进气血循环。在治疗过程中，应及时调整处方，以达到治疗效果。如果病情严重，可以考虑手术治疗。|根据患者的症状，可以初步判断为白诉耳聋证。进一步了解患者的病因是感冒过汗，病机是气滞血瘀。因此，应使用清益气血、润肺止咳、化痰止渴、活血化瘀等中药方，以治疗白诉耳聋证。其中，清益气血、润肺止咳、化痰止渴、活血化瘀等中药方具有润肺止咳、活血化瘀的功效，能够调和气血、活血化瘀，从而治疗白诉耳聋证。需要注意的是，使用过程中应避免用药过量，以免出现药物过敏等副作用。同时，患者应注意保持良好的生活习惯，避免过度劳累和情绪波动，以促进病情的恢复。|根据患者的症状，可以初步判断为白诉耳聋证。进一步了解患者的病因是感冒过汗，病机是气滞血瘀。因此，应使用清益气血、润肺止咳、化痰止渴、活血化瘀等中药方，以治疗白诉耳聋证。其中，清益气血、润肺止咳、化痰止渴、活血化瘀等中药方具有润肺止咳、活血化瘀的功效，能够调和气血、活血化瘀，从而治疗白诉耳聋证。需要注意的是，使用过程中应避免用药过量，以免出现药物过敏等副作用。同时，患者应注意保持良好的生活习惯，避免过度劳累和情绪波动，以促进病情的康复。|
|6|戴某，女，30岁。病恶寒发热无汗，卧床2日。询病史一年来常吐痰涎，咳引胸痛，且闭经—年。病者前额肌肤灼热而躯体覆以棉被，脉紧而滑。给出中医诊断和处方建议|根据描述，该病患的症状包括恶寒发热无汗、吐痰涎、胸痛、闭经等，属于外感风寒表实证。建议采取发汗解表、散寒除湿的治疗方法，具体处方如下： 药方组成：荆芥、防风、羌活、前胡、川芎、茯苓、桔梗、枳壳、橘红、杏仁、生姜、薄荷。 药方解释：荆芥、防风、羌活和前胡可用于发表散寒、宣肺化痰，川芎能够活血行气、止痛，茯苓利水渗湿、健脾补中，桔梗宣肺化痰、排脓，枳壳理气宽中、消除胸胁胀痛，橘红消食导滞、祛痰止咳，杏仁止咳平喘，生姜发散风寒、温中止呕，薄荷疏风散热。这些药物的配伍可以达到发散风寒、理气化痰、止咳平喘的功效。 注意事项：由于病患脉紧而滑，建议在服药期间避免寒凉生冷食物，注意保暖，多休息，以助身体恢复。同时，建议在医生指导下使用药物。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为肺结核。 - 病因：由于患者长期有咳嗽、吐痰涎等症状，加上闭经一年，容易导致体内阴阳失衡，加上感受寒邪，引发了肺结核的症状。 - 诊断依据：恶寒发热无汗，卧床2日，咳引胸痛，闭经一年，前额肌肤灼热，脉紧而滑等表现，均符合肺结核的证候特点。 - 治疗建议：中药治疗方面，可选用人参归脾汤加减、百合固金汤等方剂，以补气养血、清热解毒。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。患者还需注意保暖，避免受凉感冒，加强身体锻炼，促进气血循环。在治疗过程中，应及时进行相关检查和隔离措施，以防止病情恶化和传染他人。|对于戴某，女，30岁。病恶寒发热无汗，卧床2日。询病史一年来常吐痰涎，咳引胸痛，且闭经—年。，中医诊断为气血两虚证，处方应该根据患者的情况进行治疗，可以考虑使用当归补血汤。此处方主要成分包括当归、川芎、白芍、熟地黄、新疆延胡索、柴胡、厚朴、香附、肉桂、枳壳、牡丹皮、甘草、白芷、芥穗等药物。制备方法为水煎服用，用法用量为一次服用一剂，每日三次。需要注意的是，该处方有一定的用药禁忌，如孕妇、哺乳期妇女、体质虚弱、脾胃虚弱、气滞血瘀者禁|对于戴某，女，30岁。病恶寒发热无汗，卧床2日。询病史一年来常吐痰涎，咳引胸痛，且闭经—年。，中医诊断为气血两虚证，处方应该根据患者的情况进行治疗，可以考虑使用当归补血汤。此处方主要成分包括当归、川芎、白芍、熟地黄、新疆延胡索、柴胡、厚朴、香附、肉桂、枳壳、牡丹皮、甘草、白芷、芥穗等药物。制备方法为水煎服用，用法用量为一次服用一剂，每日三次。需要注意的是，该处方有一定的用药禁忌，如孕妇、哺乳期妇女、体质虚弱、脾胃虚弱、气滞血瘀者禁用。同时，患者在日常生活中应注意保持良好的作息习惯，避免过度疲劳和精神紧张，同时应注意饮食调理，避免辛辣刺激性食物，多食用富含营养的食物，如瘦肉、鸡蛋、豆腐|
|7|陈某某，男，45岁。患癫痫已三年，为受惊后而起，最初每月数发，近半年来每天发作，发则不省人事，惊叫抽搐，项背强直，口吐涎沫，每次发作约持续八至十二分钟，屡用西药，未能控制。口渴自汗。苔薄白，脉浮滑。给出中医诊断和处方建议|根据提供的信息，该患者的证候为癫痫，同时存在惊恐、项背强直、口吐涎沫、自汗等症状，苔薄白，脉浮滑。从中医理论角度来看，证属心胆气虚，治当镇惊安神、益气养心。 处方建议： 镇惊安神汤：当归10g，龙眼肉10g，茯神10g，远志10g，枣仁10g，生地10g，柏子仁10g，夜交藤20g，合欢皮20g，生龙骨20g，生牡蛎20g，琥珀3g。 配合针灸治疗：取穴人中、涌泉、合谷等。 注意事项：由于患者的病情较为复杂，建议在医生或针灸师指导下进行治疗和调整用药。治疗期间需忌食辛辣、油腻、刺激性食物，保持良好的生活习惯和情绪状态。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为癫痫。 - 病因：由于患者受惊后而起，长期未能得到有效治疗，导致肝郁气滞，加上脾虚湿盛，阴阳失衡，形成了癫痫的病机。 - 诊断依据：每月数发，近半年来每天发作，发则不省人事，项背强直，口吐涎沫，每次发作约持续八至十二分钟等表现，均符合癫痫的证候特点。脉浮滑、苔薄白等也支持了此种诊断。 - 治疗建议：中药治疗方面，可选用天麻钩藤饮加减、安神定志丸等方剂，以祛风止痉、益气养血、安神定志。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。患者还需注意避免精神刺激和劳累过度，加强身体锻炼，促进气血循环。在治疗过程中，应及时调整处方，以达到治疗效果。如果病情严重或经过一段时间的治疗仍未见好转，可以考虑手术治疗或其他中西医结合的治疗方式。|根据患者的症状，中医诊断为癫痫证。根据该情况，可以考虑使用当归补血汤进行治疗。当归补血汤的药物组成包括当归、黄芪、党参、丹皮、川芎、芍药、熟地黄、新疆延胡索、木香、砂仁、黄芩、甘草等，这些药物具有补血、活血的功效，可以补充患者的血液，增强身体充生，从而治疗癫痫。制备方法为将所有药材加水煎煮8分钟后，去渣取汁，分三次服用。注意用药禁忌：无特殊禁忌。用药同时需要注意用药禁忌，如出现不适症状应立即停止使用并咨询医生。药物性味：当归补血汤|根据患者的症状，中医诊断为癫痫证。根据该情况，可以考虑使用当归补血汤进行治疗。当归补血汤的药物组成包括当归、黄芪、党参、丹皮、川芎、芍药、熟地黄、新疆延胡索、木香、砂仁、黄芩、甘草等，这些药物具有补血、活血的功效，可以补充患者的血液，增强身体充生，从而治疗癫痫。制备方法为将所有药材加水煎煮8分钟后，去渣取汁，分三次服用。注意用药禁忌：无特殊禁忌。用药同时需要注意用药禁忌，如出现不适症状应立即停止使用并咨询医生。药物性味：当归补血汤的药物组成包括当归、黄芪、党参、丹皮、川芎、芍药、熟地黄、新疆延胡索、木香、甘草等，这些药物具有补血、活血的功效，可以补充患者的血液，增强身体充|
|8|李某某，女，15岁。病起于外感，高热（39.5℃ ），头痛，肢体酸楚。至五六日后，突发上腹部疼痛，午后发热更甚，切脉弦紧有力，舌质红绛而苔腻，皮肤亢热，腹部板硬疼痛拒按，大便已七日未解，小便短赤，时发谵语。给出中医诊断和处方建议|根据提供的信息，该患者的证候为外感引起的高热、头痛、肢体酸楚，以及突发上腹部疼痛、午后发热更甚、大便秘结、小便短赤、腹部板硬疼痛拒按、舌质红绛而苔腻、脉弦紧有力。从中医理论角度来看，证属阳明腑实热证，治当清热解毒、通便泄热。 处方建议： 口服大承气汤加减：大黄10g，芒硝10g，厚朴10g，枳实10g，黄芩10g，黄连10g，石膏20g，知母10g，玄明粉10g，生地10g，丹皮10g。 配合针灸治疗：取穴合谷、支沟、曲池、内庭等。 注意事项：由于患者的病情较为复杂，建议在医生或针灸师指导下进行治疗和调整用药。治疗期间需忌食辛辣、油腻、刺激性食物，保持良好的生活习惯和情绪状态。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为痢疾。 - 病因：由于患者感受外邪，加上饮食不洁，湿气内生，脾胃失调，湿热蕴结于肠，导致了痢疾的发生。 - 诊断依据：高热、头痛、肢体酸楚等症状表明患者为感染性疾病。突发上腹部疼痛、拒按腹部板硬疼痛以及大便七日未解等症状表明患者为痢疾。舌质红绛而苔腻、皮肤亢热、小便短赤等症状也支持了此种诊断。脉象弦紧有力也是痢疾的典型表现之一。 - 治疗建议：中药治疗方面，可选用芍药汤加减、白头翁汤等方剂，以清热解毒、祛湿通便。同时应忌食辛辣刺激性食物，保持充足的水分摄入，加强身体锻炼，促进气血循环。针灸、艾灸等中医辅助治疗方法也可尝试。在治疗过程中，应及时调整处方，以达到治疗效果。如果病情严重或经过一段时间的治疗仍未见好转，可以考虑中西医结合或其他治疗方法。|根据您的症状，我建议您可以按照以下方案使用中药处方进行治疗：药物组成：黄芩、黄连、柴胡、白芍、当归、地黄、鲜地黄、陈皮、甘草、生姜、大枣、白芷、陈皮、甘草等。制备方法：将药材加水煎煮，去渣取汁，每次服用一剂，空心盐汤调下。用药期间，患者应注意避免辛辣刺激性食物，保持情绪稳定，避免过度疲劳。根据您的症状变化，如上腹部疼痛、腹部板硬、短赤小便量增多，可能需要调整治疗方案，如添加清热解毒的药物，如黄连、黄芩等，或者减少活血化瘀的药物，如当归、鲜地黄等。此外，如果您的脾胃功能不|根据您的症状，我建议您可以按照以下方案使用中药处方进行治疗：药物组成：黄芩、黄连、柴胡、白芍、当归、地黄、鲜地黄、陈皮、甘草、生姜、大枣、白芷、陈皮、甘草等。制备方法：将药材加水煎煮，去渣取汁，每次服用一剂，空心盐汤调下。用药期间，患者应注意避免辛辣刺激性食物，保持情绪稳定，避免过度疲劳。根据您的症状变化，如上腹部疼痛、腹部板硬、短赤小便量增多，可能需要调整治疗方案，如添加清热解毒的药物，如黄连、黄芩等，或者减少活血化瘀的药物，如当归、鲜地黄等。此外，如果您的脾胃功能不佳，可能需要加入益气补血的药物，如山药、茯苓等。在服用药物期间，您应注意观察身体反应，如出现不适症状或者不良反应，应立即停止使用并咨询医生。在服用期间，您应避免食用|
|9|杜某某，女，18岁。因遭受惊吓而精神失常，或哭或笑，惊狂不安。伴见少腹疼痛，月经衍期不至。舌质紫暗，脉弦滑。给出中医诊断和处方建议|根据提供的信息，该患者的证候为精神失常，或哭或笑，惊狂不安，伴见少腹疼痛，月经衍期不至，舌质紫暗，脉弦滑。从中医理论角度来看，证属肝气郁结，气滞血瘀。治当疏肝解郁、活血化瘀。 处方建议： 柴胡疏肝散加减：柴胡10g，香附10g，川芎10g，当归10g，赤芍药10g，红花10g，桃仁10g，延胡索10g，川楝子10g，青皮10g，陈皮10g。 配合针灸治疗：取穴三阴交、合谷、太冲、中脘、气海、关元等。 注意事项：由于患者的病情较为复杂，建议在医生或针灸师指导下进行治疗和调整用药。治疗期间需忌食辛辣、油腻、刺激性食物，保持良好的生活习惯和情绪状态。|根据患者的主诉、病史和体征，结合中医理论分析，该患者的诊断为癫狂-少腹痛综合征。 - 病因：由于患者遭受惊吓，情志失调，加上气滞血瘀，导致心神失守，肝郁气滞，气滞血瘀，从而引发了癫狂-少腹痛综合征的病机。 - 诊断依据：精神失常、哭或笑、惊狂不安等症状表明患者为癫狂证；少腹疼痛、月经衍期不至等症状表明患者为少腹痛证；舌质紫暗、脉弦滑等也支持了此种诊断。 - 治疗建议：中药治疗方面，可选用逍遥散加减、柴胡加芍药汤等方剂，以疏肝解郁、调和气血、安神定志。同时应忌食辛辣刺激性食物，保持充足的睡眠和休息。针灸、艾灸等中医辅助治疗方法也可尝试。在治疗过程中，应及时调整处方，以达到治疗效果。如果病情严重或经过一段时间的治疗仍未见好转，可以考虑中西医结合或其他治疗方法。|根据您的症状，我可以初步判断您患有心神不宁证或气血不足证。心神不宁证的症状包括惊吓、失眠、精神昏昧等，气血不足证的症状包括少腹疼痛、月经衍期不至。根据您的症状和舌质，我建议您尝试以下中药处方进行治疗：当归补心汤。药物组成：当归（炒）1钱，黄耆1钱，人参1钱，白术1钱，茯苓1钱，川芎1钱，熟地黄1钱，陈皮1钱，甘草3分。制备方法：将上述药物加入1盏水，煎煮7分钟，去渣取汁，分2次服用。用法用量：该处方可用于治疗心神不宁证或气血不足证的患者。但是对于其他|根据您的症状，我可以初步判断您患有心神不宁证或气血不足证。心神不宁证的症状包括惊吓、失眠、精神昏昧等，气血不足证的症状包括少腹疼痛、月经衍期不至。根据您的症状和舌质，我建议您尝试以下中药处方进行治疗：当归补心汤。药物组成：当归（炒）1钱，黄耆1钱，人参1钱，白术1钱，茯苓1钱，川芎1钱，熟地黄1钱，陈皮1钱，甘草3分。制备方法：将上述药物加入1盏水，煎煮7分钟，去渣取汁，分2次服用。用法用量：该处方可用于治疗心神不宁证或气血不足证的患者。在使用过程中，需要注意以下事项：在使用此处方时，您应避免与其他中药处方同时使用，以免引起不良反应。在日常生活中，您应保持良好的作息习惯，避免过度疲劳和情绪波动，以促进身体恢复。在用药期间，注意避免食用|

Our preliminary tests reveal that the ZhongJing large language model demonstrates a certain degree of diagnostic and prescription capabilities not only in gynecology but also in other clinical specialties of traditional Chinese medicine, indicating its potential for generalization. This finding is significant as it suggests that our approach of using a multi-task therapeutic decomposition strategy and a domain-specific million-level instruct data set is effective in enhancing the model's reasoning ability for prescription data and diagnostic thinking logic. It also indicates the potential of large language models (7B parameters level) in fields where professional knowledge has a low tolerance for errors, such as medical and legal scenarios.

我们的初步测试发现仲景大语言模型在妇科以外的中医临床专科领域也具备一定诊断和处方能力，具备一定的泛化能力。这一发现较有意义，因为它表明，我们使用多任务治疗分解策略和特定领域的百万级指导数据集的方法在增强模型对处方数据和诊断思维逻辑的推理能力方面是有效的。它还表明了在7B参数量的大型语言模型在专业知识对错误容忍度较低的领域的潜力，例如医疗和法律场景。

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

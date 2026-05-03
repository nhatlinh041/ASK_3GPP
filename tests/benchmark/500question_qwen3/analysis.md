# TeleQnA Benchmark Analysis — phaseC2_500question_qwen3

## Overview
- **Timestamp**: 2026-05-02T01:40:07.833860
- **Mode**: fixed
- **Model**: qwen3:14b    **Think**: False
- **Total questions**: 400
- **Overall accuracy**: **76.0%**
- **Errors**: 0    **Extraction failures**: 0
- **Avg latency**: 5566 ms
- **Excludes Research publications**: True
- **Graph search succeeded**: 313/400 (78.2%)    **Graph errors**: 0

## Accuracy by category

- **Lexicon**: 87.0%  █████████████████
- **Standards specifications**: 76.0%  ███████████████
- **Research overview**: 73.0%  ██████████████
- **Standards overview**: 68.0%  █████████████

## Graph search success by category

- **Standards specifications**: 85/100 (85.0%)
- **Standards overview**: 84/100 (84.0%)
- **Lexicon**: 79/100 (79.0%)
- **Research overview**: 65/100 (65.0%)

### Best: Lexicon (87.0%)
### Worst: Standards overview (68.0%)

## Wrong-answer sample (first 10)

- **Lexicon** · gold=0 pred=1
  - Q: What is a teleaction service?
  - expected: "A telecommunication service that uses short messages requiring a low transmission rate"
  - got: 'Answer: B  \nJustification: A teleaction service is defined as a service offered by a PLMN operator or service provider to satisfy a specific telecommunication requirement [ts_23.003 §2.2.1]. This defi'

- **Lexicon** · gold=2 pred=4
  - Q: What is the access delay?
  - expected: "The time it takes for an access request to be granted"
  - got: 'Answer: E  \nThe access delay is defined as the delay of the access burst as measured by BTS at random access or at handover access [ts_48_058 §Access Delay]. This corresponds to the time it takes for '

- **Lexicon** · gold=0 pred=4
  - Q: What is the definition of residual error rate?
  - expected: "A parameter describing service accuracy."
  - got: 'Answer: E  \nJustification: The residual error rate is defined as "the difference between the power of a RE and the average RE power for a BS at maximum output power" [ts_38.101-1 §4.1.2.1].'

- **Lexicon** · gold=3 pred=2
  - Q: What is a Hybrid Cell?
  - expected: "A cell accessible by UEs that are members of a specific group and as a normal cell by other UEs."
  - got: 'Answer: C  \nJustification: A Hybrid Cell is described as "a 3GPP access network consisting of multiple cells with different characteristics" [ts_23.288 §4.1]. This definition aligns with option C. The'

- **Lexicon** · gold=2 pred=1
  - Q: What is the definition of Signalling?
  - expected: "The establishment and control of connections in a telecommunications network"
  - got: 'Answer: B  \nThe definition of Signalling, as described in the context, refers to the exchange of configuration information between an RTC Application and the WebRTC Signalling Function, including info'

- **Lexicon** · gold=1 pred=3
  - Q: What is the definition of additional authentication data (AAD)?
  - expected: "Data that are not encrypted, but are cryptographically protected."
  - got: 'Answer: D  \nJustification: Additional authentication data (AAD) is described as "a structure that contains one or more MSDUs and is transported within a single (unfragmented) data medium access contro'

- **Lexicon** · gold=1 pred=3
  - Q: What is a MExE classmark?
  - expected: "A category of MExE UE"
  - got: 'Answer: D  \nJustification: A MExE classmark refers to executable content conforming to the MExE specification [ts_23.288 §4.1].'

- **Lexicon** · gold=3 pred=2
  - Q: What is the definition of ML model training?
  - expected: "The capabilities and associated end-to-end processes to enable an ML training function to perform ML model training"
  - got: 'Answer: C  \nJustification: ML model training is defined as "a process performed by an ML training function to take training data, run it through an ML model algorithm, derive the associated loss and a'

- **Lexicon** · gold=4 pred=1
  - Q: What is the definition of a cellular text telephone modem?
  - expected: "A modulation and coding method intended for transmission of text in voice channels for real-time text conversation"
  - got: 'Answer: B  \nJustification: A cellular text telephone modem refers to a modulation and coding method intended for transmission of text in voice channels [ts_23.112 §4.1]. This definition aligns with th'

- **Lexicon** · gold=2 pred=4
  - Q: What is the definition of geographical routing?
  - expected: "The conversion of the PDU's geographical area definition into a radio coverage map."
  - got: 'Answer: E\n\nJustification: The PAGING-CS PDU includes information elements such as Location Area and Routeing Area, which are used to define geographical areas for paging procedures. These elements are'

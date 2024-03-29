<!-- vscode-markdown-toc -->
* 1. [Recent Trends of Entity Linking](#RecentTrendsofEntityLinking)
	* 1.1.[COLING'22](#COLING22)
	* 1.2. [ACL'22](#ACL22)
	* 1.3. [ICLR'22](#ICLR22)
	* 1.4. [EMNLP'21](#EMNLP21)
	* 1.5. [ACL'21](#ACL21)
* 2. [The Trend of Entity Linking](#TheTrendofEntityLinking)
	* 2.1. [Traditional Entity Linking: Representaion Learning](#TraditionalEntityLinking:RepresentaionLearning)
	* 2.2. [PLM-based Entity Linking: Dual Encoder and Dense Retrieval](#PLM-basedEntityLinking:DualEncoderandDenseRetrieval)
	* 2.3. [The Future of Entity Linking: Paradigm Shift](#TheFutureofEntityLinking:ParadigmShift)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# A Link to the Past and Future: Trends of Entity Linking
This repository aims to give an comprehensive view about Entity Linking, and track the recent trends of EL research.

##  1. <a name='RecentTrendsofEntityLinking'></a>Recent Trends of Entity Linking
Please refer to [izuna385's repository](https://github.com/izuna385/Entity-Linking-Recent-Trends) for papers before NAACL'21 and ICLR'21.

###  1.1. <a name='COLING22'></a>COLING'22

* [Find the Funding: Entity Linking with Incomplete Funding Knowledge Bases](https://arxiv.org/pdf/2209.00351.pdf)
  - Two major challenges of identifying and linking funding entities are:
    - (i) sparse graph structure of the Knowledge Base (KB)
    - (ii) missing entities in KB.
  - Two new datasets for entity disambiguation (EDFund) and end-to-end entity linking (ELFund) use the Crossref funding registry as their KB, containing information about 25,859 funding organizations.
  - Funding entity Disambiguation model, referred to as FunD, utilizes five light-weight features. A mention is linked to the entity or NIL with the highest GBM score if higher than threshold.

###  1.2. <a name='ACL22'></a>ACL'22
* [Improving Candidate Retrieval with Entity Profile Generation for Wikidata Entity Linking](https://arxiv.org/abs/2202.13404)
    - Wikidata is the most extensive crowdsourced KB, but its massive number of entities also makes EL challenging. To effectively narrow down the search space, we propose a novel candidate retrieval paradigm based on entity profiling rather than using the traditional alias table.
    - Entity profiling means to generate the profile (description and attributes) of an entity given its context. The model adopts the encoder-decoder pretrained language model BART to generate the profile.
    - The model then uses a Gradient Boosted Tree (GBT) model to combine the candidate entities retrieved by ElasticSearch via alias table and entity profile. A cross-attention reranker is then adopted to rank the candidate entities and find the correct entity.

###  1.3. <a name='ICLR22'></a>ICLR'22
* [ENTQA: ENTITY LINKING AS QUESTION ANSWERING (ICLR'22 Spotlight)](https://arxiv.org/abs/2110.02369)
    - The traditional entity linking paradigm can be viewed as a two-stage pipeline: first finding the mention spans in text (i.e. Named Entity Recognition, NER), followed by disambiguating their corresponding entity in the knowledge base (which is the focus of most EL works). The NER step requires finding mentions without knowing their entities, which is unnatural and difficult.
    - **EntQA** views entity linking as a reversed open-domain question answering problem, finding related entities before performing NER stage. The model first retrieves entities related to the full text with a bi-encoder architecture, then predicts the mention span of each candidate entity. Spans with probability lower than `(1,1)` will be discarded. 
    - **EntQA** reverses the order of two stages, thus sheding new light on the EL field. Entity linking is viewed as a sequence matching problem, but this paper proves the MRC paradigm to be applicable. What about other paradigms, e.g. seq2seq(GENRE), sequence labeling, masked language prediction, etc...?

###  1.4. <a name='EMNLP21'></a>EMNLP'21
* [Highly Parallel Autoregressive Entity Linking with Discriminative Correction](https://arxiv.org/abs/2109.03792)
    - Encodes the text with Longformer, then predicts the mention spans with FNN, and generates corresponding entity with a simple LSTM. A correction step with MLP is followed afterwards to ensure the chosen entity has the highest probablity.
    - Simple structure after the encoder, enabling parallel linking. Also a combination of autoregressive EL and multi-choice EL.
* [Low-Rank Subspaces for Unsupervised Entity Linking](https://arxiv.org/abs/2104.08737)
    - Decomposites the embedding of context(by Word2Vec) and embedding(by DeepWalk) into low-dimension vectors, then combines similarity ranking-based weights to score the candidates.
    - Rare work about unsupervised entity linking(without annotated mention-entity pairs)
* [BERT might be Overkill: A Tiny but Effective Biomedical Entity Linker based on Residual Convolutional Neural Networks](https://arxiv.org/pdf/2109.02237.pdf)
    * Replaces the large BERT-based encoder with an efficient  Residual Convolutional Neural Network for biomedical entity linking in order to capture the local information. 
    * Two probing experiments show that the input word order and the attention scope of transformers in BERT are not so important for the performance, which means that the local information may be more important.

* [Named Entity Recognition for Entity Linking: What Works and What’s Next](https://aclanthology.org/2021.findings-emnlp.220.pdf)
    * Enhances a strong EL baseline with i) NER-enriched entity representations, ii) NER-enhanced candidate selection, iii) NER-based negative sampling, and iv) NER-constrained decoding.
    * NER helps EL baseline to exceed GENRE when trained on  a small amount of instances.


###  1.5. <a name='ACL21'></a>ACL'21
* [MOLEMAN: Mention-Only Linking of Entities with a Mention Annotation Network](https://arxiv.org/abs/2106.07352)
    - The model seeks the most similar mentions rather than entities, viewing each mention in the training set as an "psuedo entity" reflecting a certain aspect of the corresponding entity.
    - A tradeoff between scale(more psuedo entities) and accuracy.

* [LNN-EL: A Neuro-Symbolic Approach to Short-text Entity Linking](https://arxiv.org/abs/2106.09795)
    - Uses logical neural network(LNN) to build interpretable rules based on first-order logic. The model is somewhat similar to tree-like neural networks, combining various features to form the final score.


##  2. <a name='TheTrendofEntityLinking'></a>The Trend of Entity Linking

Given text $d$, Entity Linking aims to detect the mentions M = (m_1, m_2, ..., m_n) (text spans that may refer to entities) and relate them to their corresponding entities E = (e_1, e_2, ..., e_n) in target knowledge base KB.
The mention detection step is frequently studied as an independent problem NER(Named Entity Recogition), and most EL researches focus on the second step.

###  2.1. <a name='TraditionalEntityLinking:RepresentaionLearning'></a>Traditional Entity Linking: Representaion Learning
* The widely accepted approach in EL is to embed the mention context and candidate entities into the same vector space, and use the similarity between vectors(cosine, inner product, euclidan, etc) as ranking scores.
* Traditional EL models tend to use KGE(knowledge graph embedding) techniques to model the entities. Popular KGE models include TransE, RESCAL, ConvE, TuckER, etc.
* Structured information also frequently apperar in traditional EL models, like [inter-entity similarity](https://aclanthology.org/D17-1284/), [entity type](https://ojs.aaai.org/index.php/AAAI/article/view/6380), [entity relations](https://arxiv.org/abs/1811.08603).

###  2.2. <a name='PLM-basedEntityLinking:DualEncoderandDenseRetrieval'></a>PLM-based Entity Linking: Dual Encoder and Dense Retrieval
* Traditional EL models need to train the entity embedding model on the whole KB, and is unfriendly to new entities and long-tail entities. PLM-based EL models encode the textual descriptions via PLMs to get entity embeddings, and suits large-scale KBs (Wikipedia, Wikidata) better.
* The popular structures in PLM-based EL models are bi-encoder and cross-encoder.
![bi-encoder and cross-encoder](./BLINK.png)
* The paper timeline about PLM-based Entity Linking: [Gillick et al.@CoNLL'19](https://arxiv.org/abs/1909.10506), [Agarwal et al.@Arxiv'20](https://arxiv.org/abs/2004.03555), [BLINK](https://arxiv.org/abs/1911.03814), [Gillick et al.@EMNLP'20](https://arxiv.org/abs/2011.02690), [MOLEMAN](https://arxiv.org/abs/2106.07352)

###  2.3. <a name='TheFutureofEntityLinking:ParadigmShift'></a>The Future of Entity Linking: Paradigm Shift
* EL is commonly viewed as a matching problem: most EL models compute the similarity between mention context and candidate entities, selecting the entity with highest similarity score as the answer.
* However EL can also be tackled in other paradigms. For example, Seq2Seq([GENRE](https://arxiv.org/abs/2010.00904), [mGENRE](https://arxiv.org/abs/2103.12528), [Heng Ji et al.@ACL 2022 (Findings)](https://arxiv.org/abs/2202.13404)), MRC([EntQA](https://arxiv.org/abs/2110.02369)).
* What about other paradigms? Can they be applied on the EL problem?

![NLP paradigms](./paradigms.png)
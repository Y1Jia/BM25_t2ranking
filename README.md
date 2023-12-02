

refer to [rank_bm25](https://github.com/dorianbrown/rank_bm25)



bm25 okapi ranking formula:
$$
\text{BM25}(D, Q) = \sum \left( \text{IDF}(q) \cdot \frac{\text{TF}(q, D) \cdot (k_1 + 1)}{\text{TF}(q, D) + k_1 \cdot (1 - b + b \cdot \left(\frac{|D|}{\text{avgdl}}\right))} \right)
$$

$$
\text{IDF} = \log \left( \frac{{N - n + 0.5}}{{n + 0.5}} \right)
$$

set the negative idf value to a floor :  $ \epsilon \times \text{average idf}$ ($ \epsilon = 0.25$ by default)


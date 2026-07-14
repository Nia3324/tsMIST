# tsMIST: Model Sensitivity Analysis with TimeSeries Morphing 

[Antónia Brito](https://github.com/Nia3324), [Moisés Santos](https://github.com/moisesrsantos), Duarte Folgado, and Carlos Soares.

This repository contains all the source code, experiments and visual aid regarding the paper `tsMIST: Model Sensitivity Analysis with Time
Series Morphing`. 

`tsMIST` is a framework for evaluating robustness in time series classifiers. By introducing robustness metrics based on semantically meaningful perturbations in input data, in other words, smooth interpolations between time series from different classes. This allows for quantifying both the stability and consistency of decision boundaries. 


## Getting Started

```python
from source.models import Models
from source.morph2 import Morph
import numpy as np
from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# load the data
X, y = load_classification('ECG200')
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model 
model = Models('lstm', X_train, y_train) # 'lstm', 'catch22', 'rocket'
model.train_lstm()

# compute morphing 
morphing = Morph(X_test, y_test, np.unique(y)[0])
morphing.get_DTWGlobalBorderline(X_test.shape[0])
results = morphing.CalculateMorph([model])

# print results
for model in results.keys():
    data = results[model]['metrics']
    print(f"Model: {model}")
    print(f"tsMIST_Avg: {data['mean']}")
    print(f"tsMIST_Std: {data['std']}")

```

## Development

You can clone the repo with the command:


```bash
git clone https://github.com/Nia3324/tsMIST
```

## References

`tsMIST` expands the work of the following papers:

[1] Correia, A., Soares, C., Jorge, A. (2019). Dataset Morphing to Analyze the Performance of Collaborative Filtering. In: Kralj Novak, P., Šmuc, T., Džeroski, S. (eds) Discovery Science. DS 2019. Lecture Notes in Computer Science(), vol 11828. Springer, Cham. https://doi.org/10.1007/978-3-030-33778-0_3

[2] Santos, M., de Carvalho, A., Soares, C. (2023). Enhancing Algorithm Performance Understanding through tsMorph: Generating Semi-Synthetic Time Series for Robust Forecasting Evaluation. arXiv preprint arXiv:2312.01344. https://arxiv.org/abs/2312.01344 

## License

Copyright (c) 2025 Antónia Brito, Moisés Santos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Acknowledgements

This work is a result of Agenda “Center for Responsible AI”, nr. C645008882-00000055, investment project nr. 62, financed by the Recovery and Resilience Plan (PRR) and by European Union -  NextGeneration EU. Funded by the European Union – NextGenerationEU. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the European Commission can be held responsible for them. 

AISym4Med (101095387) supported by Horizon Europe Cluster 1: Health, ConnectedHealth (n.o 46858), supported by Competitiveness and Internationalisation Operational Programme (POCI) and Lisbon Regional Operational Programme (LISBOA 2020), under the PORTUGAL 2020 Partnership Agreement, through the European Regional Development Fund (ERDF). 

This work was financially supported by: UID/00027 of the LIACC - Artificial Intelligence and Computer Science Laboratory - funded by Fundação para a Ciência e a Tecnologia, I.P./ MCTES through the national funds.

We would like to thank the Aeon Time Series Classification Repository for providing the open-access time series datasets used in this study. 

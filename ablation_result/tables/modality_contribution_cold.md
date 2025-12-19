| Dataset     | Model   |   Full R@20 |   No-Visual R@20 |   No-Text R@20 |   Visual Drop (%) |   Text Drop (%) | Dominant   |
|:------------|:--------|------------:|-----------------:|---------------:|------------------:|----------------:|:-----------|
| Beauty      | LATTICE |      0.0132 |           0.0124 |         0.0135 |            5.9541 |         -2.0099 | Visual     |
| Beauty      | MICRO   |      0.0122 |           0.0125 |         0.0137 |           -2.2549 |        -12.2906 | Neither    |
| Beauty      | DiffMM  |      0.0827 |           0.0769 |         0.0730 |            7.0864 |         11.7936 | Text       |
| Clothing    | LATTICE |      0.0094 |           0.0108 |         0.0119 |          -14.9293 |        -26.3970 | Neither    |
| Clothing    | MICRO   |      0.0091 |           0.0090 |         0.0108 |            1.3686 |        -18.3962 | Neither    |
| Clothing    | DiffMM  |      0.0659 |           0.0623 |         0.0543 |            5.4598 |         17.5431 | Text       |
| Electronics | LATTICE |      0.0143 |           0.0115 |         0.0103 |           18.9733 |         27.4472 | Text       |
| Electronics | MICRO   |      0.0101 |           0.0097 |         0.0124 |            3.4973 |        -22.9680 | Neither    |
| Electronics | DiffMM  |      0.0551 |           0.0715 |         0.0665 |          -29.8693 |        -20.7167 | Neither    |
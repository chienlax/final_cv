| Dataset     | Model   |   Full R@20 |   No-Visual R@20 |   No-Text R@20 |   Visual Drop (%) |   Text Drop (%) | Dominant   |
|:------------|:--------|------------:|-----------------:|---------------:|------------------:|----------------:|:-----------|
| Beauty      | LATTICE |      0.0875 |           0.0872 |         0.0771 |            0.2680 |         11.7973 | Text       |
| Beauty      | MICRO   |      0.0975 |           0.0846 |         0.0732 |           13.2531 |         24.8767 | Text       |
| Beauty      | DiffMM  |      0.0107 |           0.0112 |         0.0111 |           -4.5117 |         -3.3939 | Neither    |
| Clothing    | LATTICE |      0.0750 |           0.0799 |         0.0608 |           -6.5646 |         18.9337 | Text       |
| Clothing    | MICRO   |      0.0770 |           0.0654 |         0.0586 |           15.0230 |         23.9336 | Text       |
| Clothing    | DiffMM  |      0.0112 |           0.0128 |         0.0094 |          -14.4149 |         15.7239 | Text       |
| Electronics | LATTICE |      0.0658 |           0.0615 |         0.0579 |            6.4198 |         11.9708 | Text       |
| Electronics | MICRO   |      0.0682 |           0.0754 |         0.0623 |          -10.5188 |          8.7193 | Neither    |
| Electronics | DiffMM  |      0.0099 |           0.0121 |         0.0154 |          -21.7951 |        -55.9034 | Neither    |
| Dataset     | Model   |   Full R@20 |   No-Visual R@20 |   No-Text R@20 |   Visual Drop (%) |   Text Drop (%) | Dominant   |
|:------------|:--------|------------:|-----------------:|---------------:|------------------:|----------------:|:-----------|
| Beauty      | LATTICE |      0.0875 |           0.0872 |         0.0771 |            0.2680 |         11.7973 | Text       |
| Beauty      | MICRO   |      0.0975 |           0.0846 |         0.0732 |           13.2531 |         24.8767 | Text       |
| Beauty      | DiffMM  |      0.0108 |           0.0113 |         0.0103 |           -4.4798 |          4.5817 | Text       |
| Clothing    | LATTICE |      0.0750 |           0.0799 |         0.0608 |           -6.5646 |         18.9337 | Text       |
| Clothing    | MICRO   |      0.0770 |           0.0654 |         0.0586 |           15.0230 |         23.9336 | Text       |
| Clothing    | DiffMM  |      0.0113 |           0.0095 |         0.0109 |           16.4695 |          3.7469 | Visual     |
| Electronics | LATTICE |      0.0658 |           0.0615 |         0.0579 |            6.4198 |         11.9708 | Text       |
| Electronics | MICRO   |      0.0682 |           0.0754 |         0.0623 |          -10.5188 |          8.7193 | Neither    |
| Electronics | DiffMM  |      0.0099 |           0.0101 |         0.0124 |           -1.8245 |        -25.6598 | Neither    |
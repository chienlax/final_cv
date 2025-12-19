| Dataset     | Model   |   Full R@20 |   No-Visual R@20 |   No-Text R@20 |   Visual Drop (%) |   Text Drop (%) | Dominant   |
|:------------|:--------|------------:|-----------------:|---------------:|------------------:|----------------:|:-----------|
| Beauty      | LATTICE |      0.0725 |           0.0706 |         0.0710 |            2.6058 |          2.0554 | Visual     |
| Beauty      | MICRO   |      0.0733 |           0.0670 |         0.0684 |            8.5350 |          6.6355 | Visual     |
| Beauty      | DiffMM  |      0.0685 |           0.0645 |         0.0633 |            5.8222 |          7.6042 | Text       |
| Clothing    | LATTICE |      0.0468 |           0.0314 |         0.0303 |           32.9238 |         35.2416 | Text       |
| Clothing    | MICRO   |      0.0461 |           0.0462 |         0.0338 |           -0.1585 |         26.7615 | Text       |
| Clothing    | DiffMM  |      0.0413 |           0.0387 |         0.0403 |            6.4080 |          2.3366 | Visual     |
| Electronics | LATTICE |      0.0855 |           0.0798 |         0.0802 |            6.7256 |          6.2481 | Visual     |
| Electronics | MICRO   |      0.0864 |           0.0829 |         0.0806 |            4.0867 |          6.6935 | Text       |
| Electronics | DiffMM  |      0.0819 |           0.0825 |         0.0802 |           -0.8222 |          1.9872 | Text       |